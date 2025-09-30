#!/usr/bin/env python3
"""
biolink_utils.py

Utilities for normalizing Biolink categories and predicates.

- Pinned-node typing via CURIE→category (curie2cat.pkl), with prefix fallback
- Predicate normalization to canonical `biolink:*` identifiers
"""
from __future__ import annotations

import re
import pickle
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from ..config import settings

__all__ = ["category_from_curie", "normalize_pred"]

# Optional Biolink YAML (kept for extensions; not used directly here)
_BIOLINK_YAML_PATH: Path = Path(__file__).parent.parent / "biolink-model.yaml"
try:
    _YAML: Dict[str, Any] = yaml.safe_load(_BIOLINK_YAML_PATH.read_text())
except Exception:
    _YAML = {}

# Prefix → Category fallback (used only if CURIE not found in pickle)
PREFIX2CAT: Dict[str, str] = {
    "CHEBI":     "biolink:ChemicalEntity",
    "DRUGBANK":  "biolink:ChemicalEntity",
    "PUBCHEM":   "biolink:ChemicalEntity",
    "CHEMBL":    "biolink:ChemicalEntity",
    "NCBIGene":  "biolink:Gene",
    "ENSEMBL":   "biolink:Gene",
    "HGNC":      "biolink:Gene",
    "UNIPROTKB": "biolink:Protein",
    "PR":        "biolink:Protein",  # Protein Ontology
    "MONDO":     "biolink:Disease",
    "DOID":      "biolink:Disease",
    "EFO":       "biolink:Disease",
    "HP":        "biolink:PhenotypicFeature",
    "UBERON":    "biolink:AnatomicalEntity",
}

# Exact-phrase predicate synonyms (after cleaning)
PRED_SYNONYM: Dict[str, str] = {
    # interaction family
    "interacts with": "biolink:physically_interacts_with",
    "interacts":      "biolink:physically_interacts_with",
    "interact":       "biolink:physically_interacts_with",
    "binds":          "biolink:physically_interacts_with",
    "binds to":       "biolink:physically_interacts_with",
    "targets":        "biolink:physically_interacts_with",
    "has target":     "biolink:physically_interacts_with",
    "physically interacts with": "biolink:physically_interacts_with",
    "directly physically interacts with": "biolink:physically_interacts_with",
    "indirectly physically interacts with": "biolink:physically_interacts_with",
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

# Regex rules (order matters; first match wins)
_PRED_REGEX_RULES = [
    (r"\binteract\w*\b",      "physically_interacts_with"),
    (r"\bbind\w*\b",          "physically_interacts_with"),
    (r"\btarget\w*\b",        "physically_interacts_with"),
    (r"\bassociat\w*\b",      "related_to"),
    (r"\brelat\w*\b",         "related_to"),
    (r"\btreat\w*\b",         "treats"),
    (r"\bcontraindicat\w*\b", "contraindicated_for"),
]

@lru_cache(maxsize=1)
def _load_curie2cat() -> Dict[str, str]:
    """Load CURIE→category map once; return {} if missing/unreadable."""
    try:
        p = settings.CURIE2CAT_PKL
        if isinstance(p, str):
            p = Path(p)
        if isinstance(p, Path) and p.exists():
            with p.open("rb") as fh:
                return pickle.load(fh)
    except Exception:
        pass
    return {}

def category_from_curie(curie: str) -> str:
    """
    Resolve a Biolink category for a pinned-node CURIE.

    Preference:
      1) Exact lookup in curie2cat.pkl
      2) CURIE prefix fallback
      3) 'biolink:NamedThing'
    """
    if not curie:
        return "biolink:NamedThing"
    cat = _load_curie2cat().get(curie)
    if cat:
        return cat
    prefix = curie.split(":", 1)[0]
    return PREFIX2CAT.get(prefix, "biolink:NamedThing")

def _clean(text: str) -> str:
    """Lowercase and collapse to words/spaces."""
    return re.sub(r"\W+", " ", (text or "").lower()).strip()

def normalize_pred(text: str) -> str:
    """
    Normalize a free-text predicate into a canonical `biolink:*` identifier.

    Steps:
      • if already `biolink:*`, normalize spacing/underscores and return
      • lowercase & clean
      • exact-phrase synonym map
      • regex stems/inflections
      • fallback: underscore the cleaned phrase under biolink namespace
    """
    if not text:
        return "biolink:related_to"

    t = text.strip()

    # Already a CURIE? normalize the right side and return.
    if t.lower().startswith("biolink:"):
        right = t.split(":", 1)[1].strip().replace(" ", "_")
        right = re.sub(r"_+", "_", right)
        return f"biolink:{right}"

    cleaned = _clean(t)

    # Exact-phrase synonyms
    mapped = PRED_SYNONYM.get(cleaned)
    if mapped:
        return mapped

    # Regex stems/inflections
    for pat, canonical in _PRED_REGEX_RULES:
        if re.search(pat, cleaned):
            return f"biolink:{canonical}"

    # Fallback: cleaned phrase token under biolink namespace
    token = re.sub(r"_+", "_", cleaned.replace(" ", "_"))
    return f"biolink:{token}"

