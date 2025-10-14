from __future__ import annotations

import os
import re
import pickle
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "category_from_curie",
    "normalize_pred",
    "is_a",
    "allowed_subject_categories",
    "allowed_object_categories",
    "canonicalize_class",  # lightweight class canonicalizer (no RAG)
]

# ──────────────────────────────────────────────────────────────────────────────
# CURIE → category (authoritative if present; else prefix fallback)
# ──────────────────────────────────────────────────────────────────────────────

PREFIX2CAT: Dict[str, str] = {
    "CHEBI":     "biolink:ChemicalEntity",
    "DRUGBANK":  "biolink:ChemicalEntity",
    "PUBCHEM":   "biolink:ChemicalEntity",
    "CHEMBL":    "biolink:ChemicalEntity",
    "NCBIGene":  "biolink:Gene",
    "ENSEMBL":   "biolink:Gene",
    "HGNC":      "biolink:Gene",
    "UNIPROTKB": "biolink:Protein",
    "PR":        "biolink:Protein",
    "MONDO":     "biolink:Disease",
    "DOID":      "biolink:Disease",
    "EFO":       "biolink:Disease",
    "HP":        "biolink:PhenotypicFeature",
    "UBERON":    "biolink:AnatomicalEntity",
}

@lru_cache(maxsize=1)
def _load_curie2cat() -> Dict[str, str]:
    """
    Try to read a pickle with CURIE→category mapping from:
      - env CURIE2CAT_PKL (absolute path)
      - repo default: data/curie2cat.pkl
    Return {} if absent.
    """
    candidates = []
    env_p = os.getenv("CURIE2CAT_PKL")
    if env_p:
        candidates.append(Path(env_p))
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "curie2cat.pkl")

    for p in candidates:
        try:
            if p and p.exists():
                with p.open("rb") as fh:
                    return pickle.load(fh)
        except Exception:
            pass
    return {}

def category_from_curie(curie: str) -> str:
    if not curie:
        return "biolink:NamedThing"
    cat = _load_curie2cat().get(curie)
    if cat:
        return cat
    # fallback by prefix
    prefix = curie.split(":", 1)[0]
    return PREFIX2CAT.get(prefix, "biolink:NamedThing")

# ──────────────────────────────────────────────────────────────────────────────
# Predicate normalization
# ──────────────────────────────────────────────────────────────────────────────

PRED_SYNONYM: Dict[str, str] = {
    # expression / localization families (NEW)
    "expressed in":   "biolink:expressed_in",
    "expression in":  "biolink:expressed_in",
    "expressed":      "biolink:expressed_in",   # good enough for NL queries
    "located in":     "biolink:located_in",
    "localized to":   "biolink:located_in",

    # interaction family
    "interacts with": "biolink:physically_interacts_with",
    "interacts":      "biolink:physically_interacts_with",
    "interact":       "biolink:physically_interacts_with",
    "binds":          "biolink:physically_interacts_with",
    "binds to":       "biolink:physically_interacts_with",
    "targets":        "biolink:physically_interacts_with",
    "has target":     "biolink:physically_interacts_with",
    "physically interacts with": "biolink:physically_interacts_with",

    # generic relatedness
    "related to":     "biolink:related_to",
    "related":        "biolink:related_to",
    "associated with":"biolink:related_to",
    "associates with":"biolink:related_to",

    # clinical-ish
    "treats":                 "biolink:treats",
    "contraindicated":        "biolink:contraindicated_for",
    "contraindicated for":    "biolink:contraindicated_for",
}

# order matters: earlier rules have priority
_PRED_REGEX_RULES = [
    # NEW: catch expressed/express/expressing → expressed_in
    (r"\bexpress\w*\b", "expressed_in"),
    # NEW: catch locate/located/localization → located_in
    (r"\blocat\w*\b",   "located_in"),

    # existing rules
    (r"\binteract\w*\b",      "physically_interacts_with"),
    (r"\bbind\w*\b",          "physically_interacts_with"),
    (r"\btarget\w*\b",        "physically_interacts_with"),
    (r"\bassociat\w*\b",      "related_to"),
    (r"\brelat\w*\b",         "related_to"),
    (r"\btreat\w*\b",         "treats"),
    (r"\bcontraindicat\w*\b", "contraindicated_for"),
]

def _clean(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()

def normalize_pred(text: str) -> str:
    if not text:
        return "biolink:related_to"
    t = text.strip()
    if t.lower().startswith("biolink:"):
        right = t.split(":", 1)[1].strip().replace(" ", "_")
        right = re.sub(r"_+", "_", right)
        return f"biolink:{right}"

    cleaned = _clean(t)
    mapped = PRED_SYNONYM.get(cleaned)
    if mapped:
        return mapped
    for pat, canonical in _PRED_REGEX_RULES:
        if re.search(pat, cleaned):
            return f"biolink:{canonical}"
    token = re.sub(r"_+", "_", cleaned.replace(" ", "_"))
    return f"biolink:{token}"

# ──────────────────────────────────────────────────────────────────────────────
# Minimal Biolink “graph” from YAML (for is_a + domain/range checks)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_biolink() -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      classes_is_a:   class -> set(parents)       (transitive closure computed lazily)
      pred_domain:    predicate_key -> class_key  (subject domain)
      pred_range:     predicate_key -> class_key  (object range)
    If BIOLINK_YAML is unset/unreadable, returns empty dicts (callers must tolerate).
    """
    p = os.getenv("BIOLINK_YAML")
    if not p:
        return {}, {}, {}
    try:
        data = yaml.safe_load(open(p, "r"))
    except Exception:
        return {}, {}, {}

    classes_is_a: Dict[str, Set[str]] = {}
    pred_domain: Dict[str, str] = {}
    pred_range: Dict[str, str] = {}

    # classes
    for k, v in (data.get("classes") or {}).items():
        parents = set(v.get("is_a") or []) | set(v.get("mixins") or [])
        # normalize “biolink:Class” keys to just “Class”
        classes_is_a[k] = {p.split(":")[-1] for p in parents}

    # slots/predicates
    for k, v in (data.get("slots") or {}).items():
        if v.get("slot_uri", "").startswith("biolink:"):
            pred_domain[k] = (v.get("domain") or "").split(":")[-1] or ""
            pred_range[k]  = (v.get("range")  or "").split(":")[-1] or ""

    return classes_is_a, pred_domain, pred_range

@lru_cache(maxsize=None)
def _isa_memo(child: str) -> Set[str]:
    """Transitive closure of is_a for a class key (without 'biolink:' prefix)."""
    classes_is_a, _, _ = _load_biolink()
    seen: Set[str] = set()
    stack = [child]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for p in classes_is_a.get(x, ()):
            stack.append(p)
    return seen

def is_a(child: str, parent: str) -> bool:
    """
    Return True if biolink:Child is a (transitively) biolink:Parent.
    Accepts either 'biolink:Class' or bare 'Class'.
    """
    if not child or not parent:
        return False
    c = child.split(":")[-1]
    p = parent.split(":")[-1]
    if c == p:
        return True
    return p in _isa_memo(c)

def _domain_range_for_predicate(pred_curie: str) -> Tuple[str, str]:
    """
    Map a predicate CURIE to (domain_class, range_class) keys (no 'biolink:' prefix).
    If unknown, returns ("","").
    """
    _, pred_domain, pred_range = _load_biolink()
    key = pred_curie.split(":", 1)[-1]  # e.g., "physically_interacts_with"
    return pred_domain.get(key, ""), pred_range.get(key, "")

def allowed_subject_categories(pred_curie: str) -> Set[str]:
    """
    Subject classes allowed for this predicate (transitive closure).
    Returns full 'biolink:*' strings.
    If YAML not loaded, returns empty set (caller should skip check).
    """
    domain, _ = _domain_range_for_predicate(pred_curie)
    if not domain:
        return set()
    # All subclasses of domain are allowed
    return {f"biolink:{c}" for c in _isa_memo(domain)}

def allowed_object_categories(pred_curie: str) -> Set[str]:
    """Object classes allowed for this predicate (transitive closure)."""
    _, range_ = _domain_range_for_predicate(pred_curie)
    if not range_:
        return set()
    return {f"biolink:{c}" for c in _isa_memo(range_)}

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight class canonicalizer for "via X" constraints (no RAG)
# ──────────────────────────────────────────────────────────────────────────────

# Small, safe allowlist of entity-like categories that make sense as intermediates
_ALLOWED_INTERMEDIATE: Set[str] = {
    "Gene", "Protein",
    "Disease", "PhenotypicFeature",
    "Drug", "ChemicalEntity", "SmallMolecule",
    "Pathway", "BiologicalProcess", "MolecularActivity",
    "AnatomicalEntity", "Cell", "CellularComponent",
}

_SYNONYM_TO_CLASS: Dict[str, str] = {
    # genes / proteins
    "gene": "Gene", "genes": "Gene",
    "protein": "Protein", "proteins": "Protein",
    # drugs / chemicals
    "drug": "Drug", "drugs": "Drug",
    "chemical": "ChemicalEntity", "chemicals": "ChemicalEntity",
    "compound": "ChemicalEntity", "compounds": "ChemicalEntity",
    "small molecule": "SmallMolecule", "small molecules": "SmallMolecule",
    # diseases / phenotypes
    "disease": "Disease", "diseases": "Disease",
    "phenotype": "PhenotypicFeature", "phenotypes": "PhenotypicFeature",
    # pathways / processes / activities
    "pathway": "Pathway", "pathways": "Pathway",
    "process": "BiologicalProcess", "processes": "BiologicalProcess",
    "biological process": "BiologicalProcess",
    "activity": "MolecularActivity", "activities": "MolecularActivity",
    # anatomy
    "anatomy": "AnatomicalEntity", "tissue": "AnatomicalEntity", "tissues": "AnatomicalEntity",
    "cell": "Cell", "cells": "Cell",
    "cellular component": "CellularComponent", "organelle": "CellularComponent",
}

def canonicalize_class(term: str) -> Optional[str]:
    """
    Map a free-text mention like 'genes', 'a drug', 'pathways' → canonical Biolink class
    within a small allowlist. Returns 'biolink:ClassName' or None if not recognized.

    - Accepts already-canonical forms like 'biolink:Gene'
    - Uses a tiny synonym table
    - If BIOLINK_YAML is set, also honors exact class keys from that YAML
      (still filtered to the allowlist)
    """
    if not term:
        return None

    t = str(term).strip()

    # Already namespaced → check allowlist
    if t.lower().startswith("biolink:"):
        cls = t.split(":", 1)[1].strip()
        cls = cls[:1].upper() + cls[1:]  # TitleCase first char
        return f"biolink:{cls}" if cls in _ALLOWED_INTERMEDIATE else None

    # Synonym sweep
    key = " ".join(_clean(t).split())
    if key in _SYNONYM_TO_CLASS:
        cls = _SYNONYM_TO_CLASS[key]
        return f"biolink:{cls}" if cls in _ALLOWED_INTERMEDIATE else None

    # Optional: derive from loaded Biolink YAML class keys (if available)
    try:
        classes_is_a, _, _ = _load_biolink()
        if classes_is_a:
            classes = set(classes_is_a.keys())
            match = next((c for c in classes if c.lower() == key.lower()), None)
            if match and (match in _ALLOWED_INTERMEDIATE):
                return f"biolink:{match}"
    except Exception:
        pass

    return None
