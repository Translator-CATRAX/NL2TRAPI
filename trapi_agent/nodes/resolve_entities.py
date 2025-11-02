#!/usr/bin/env python3
"""
resolve_entities.py

LangGraph node: resolve free-text entities into TRAPI query_graph nodes.

General policy
--------------
1) If we can pin a CURIE:
   • Type it via biolink_utils.category_from_curie (uses curated map or prefix fallbacks).
   • Mark node as pinned=True.

2) If we cannot pin:
   • Leave a generic placeholder ("biolink:NamedThing").
   • ResolveSchema may type generic nodes later (unless the route skips schema).

Route-specific behavior
-----------------------
• route == "treats": if no pinned Disease is present after normal resolution,
  run a robust fallback that extracts a disease phrase and pins a Disease CURIE
  (covers simple cases like "Diabetes" as well as "drugs for migraine").

Generic safety-net (all routes)
-------------------------------
• If, after normal resolution (and any route-specific fallback), there are STILL zero
  pinned nodes, try to pin ANY resolvable entity from the question (NodeNorm → local
  resolver → n-gram scan). This prevents one-hop like
  "Which genes are associated with asthma?" from failing when the parser emits only
  a generic 'biolink:Gene' but no pinned node.

Category-hint stoplist
----------------------
• Words like "tissue(s)", "gene(s)", "protein(s)", "disease(s)", "pathway(s)", "phenotype(s)"
  are treated as schema hints rather than named entities, so they go to state['generic_types']
  and are NOT sent to NodeNorm (prevents spurious pins like “tissues” → MONDO disease).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Set, Tuple, List

from ..state_types import TRAPIState
from ..utils.node_norm import lookup as nn_lookup
from ..utils.node_resolver import NodeResolver

logger = logging.getLogger(__name__)

# Words that should be interpreted as Biolink class hints, not named entities.
GENERIC_CATEGORY_HINTS = {
    "tissue":   "biolink:AnatomicalEntity",
    "tissues":  "biolink:AnatomicalEntity",
    "cell":     "biolink:Cell",
    "cells":    "biolink:Cell",
    "process":  "biolink:BiologicalProcess",
    "processes":"biolink:BiologicalProcess",
    "gene":     "biolink:Gene",
    "genes":    "biolink:Gene",
    "protein":  "biolink:Protein",
    "proteins": "biolink:Protein",
    "disease":  "biolink:Disease",
    "diseases": "biolink:Disease",
    "pathway":  "biolink:Pathway",
    "pathways": "biolink:Pathway",
    "phenotype":"biolink:PhenotypicFeature",
    "phenotypes":"biolink:PhenotypicFeature",
}

# --- Category resolution (prefer your shared utility) --------------------------------
try:
    # Centralized mapping (curated map -> prefix fallback)
    from ..utils.biolink_utils import category_from_curie as _category_for_curie  # type: ignore
except Exception:
    # Safe internal prefix-only fallback (used only if utility is missing)
    _PREFIX2CAT: Dict[str, str] = {
        "CHEBI": "biolink:ChemicalEntity",
        "DRUGBANK": "biolink:ChemicalEntity",
        "PUBCHEM": "biolink:ChemicalEntity",
        "CHEMBL": "biolink:ChemicalEntity",
        "NCBIGene": "biolink:Gene",
        "ENSEMBL": "biolink:Gene",
        "HGNC": "biolink:Gene",
        "UNIPROTKB": "biolink:Protein",
        "PR": "biolink:Protein",
        "MONDO": "biolink:Disease",
        "DOID": "biolink:Disease",
        "EFO": "biolink:Disease",
        "HP": "biolink:PhenotypicFeature",
        "UBERON": "biolink:AnatomicalEntity",
    }
    def _category_for_curie(curie: str) -> str:
        return _PREFIX2CAT.get(curie.split(":", 1)[0], "biolink:NamedThing")

# --- Lazy local resolver (optional) ---------------------------------------------------
_RESOLVER: Optional[NodeResolver] = None
def _resolver() -> Optional[NodeResolver]:
    global _RESOLVER
    if _RESOLVER is None:
        try:
            _RESOLVER = NodeResolver()
        except Exception as e:
            logger.warning("Local NodeResolver unavailable: %s", e)
            _RESOLVER = None
    return _RESOLVER

# --- Tiny helpers ---------------------------------------------------------------------
def _count_pinned(nodes: Dict[str, Dict[str, Any]]) -> int:
    return sum(1 for meta in nodes.values() if meta.get("id"))

def _has_pinned_disease(nodes: Dict[str, Dict[str, Any]]) -> bool:
    for meta in nodes.values():
        if meta.get("id") and "biolink:Disease" in (meta.get("category") or []):
            return True
    return False

def _add_pinned_node(
    nodes: Dict[str, Dict[str, Any]],
    used: Set[str],
    curie: str,
    name: str,
    label_map: Dict[str, str] | None = None,
) -> None:
    if curie in used:
        return
    nid = f"n{len(nodes)}"
    nodes[nid] = {
        "id": curie,
        "name": name,
        "category": [_category_for_curie(curie)],
        "pinned": True,
    }
    used.add(curie)
    if label_map is not None and name:
        label_map[curie] = name

def _tokenize(q: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9’'\-]*", q or "")

# --- Treats-specific fallback logic ---------------------------------------------------
# 1) Named disease forms: "X disease/disorder/syndrome"
_DISEASE_PATTERNS = [
    r"\b([A-Z][a-zA-Z’' -]+?)\s+(?:disease|disorder|syndrome)\b",
    r"\b(?:with|of)\s+([A-Z][a-zA-Z’' -]+?)\s+(?:disease|disorder|syndrome)\b",
]

def _extract_disease_candidate(q: str) -> Optional[str]:
    q = (q or "").strip()
    for pat in _DISEASE_PATTERNS:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" -’'")
    # “treat <X> …” at end (covers “treat Diabetes?”)
    m = re.search(r"\btreat(?:s|ing)?\s+([A-Za-z0-9’' -]+?)[\.\?\!]*\s*$",
                  q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" ?!.:-'’")
    # “… for <X> …” near end (“drugs for Diabetes”)
    m = re.search(r"\bfor\s+([A-Za-z0-9’' -]+?)[\.\?\!]*\s*$",
                  q, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" ?!.:-'’")
    return None

def _try_pin_disease(name: str) -> Optional[Tuple[str, str]]:
    """Return (curie, label) if name resolves to a Disease; else None."""
    if not name:
        return None
    curie, label, _ = nn_lookup(name, mode="lookup")  # SRI NodeNorm first
    if curie and _category_for_curie(curie) == "biolink:Disease":
        return curie, (label or name)
    res = _resolver()                                 # local resolver next
    if res:
        hit = res.resolve(name)
        if hit and hit.get("id"):
            curie = hit["id"]
            if _category_for_curie(curie) == "biolink:Disease":
                return curie, hit.get("name", name)
    return None

def _pin_disease_from_ngrams(q: str, max_n: int = 5) -> Optional[Tuple[str, str]]:
    """Scan n-grams (1..max_n) and pin the first that resolves to a Disease CURIE."""
    toks = _tokenize(q)
    if not toks:
        return None
    n = min(max_n, len(toks))
    skip = {"what", "which", "drug", "drugs", "chemical", "chemicals",
            "treat", "treats", "treating", "are", "for", "help", "helpful"}
    for k in range(n, 0, -1):  # longer first
        for i in range(0, len(toks) - k + 1):
            phrase = " ".join(toks[i:i+k]).strip(" -’'")
            if phrase.lower() in skip:
                continue
            got = _try_pin_disease(phrase)
            if got:
                return got
    return None

# --- Generic (route-agnostic) fallback logic -----------------------------------------
def _try_pin_any(name: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to resolve 'name' to ANY CURIE via NodeNorm, then local resolver.
    Return (curie, label, category) or None.
    """
    if not name:
        return None
    curie, label, _ = nn_lookup(name, mode="lookup")  # NodeNorm first
    if curie:
        return curie, (label or name), _category_for_curie(curie)
    res = _resolver()
    if res:
        hit = res.resolve(name)
        if hit and hit.get("id"):
            curie = hit["id"]
            return curie, hit.get("name", name), _category_for_curie(curie)
    return None

def _pin_any_from_ngrams(q: str, max_n: int = 6) -> Optional[Tuple[str, str, str]]:
    """
    Scan n-grams (1..max_n) and pin the first that resolves to ANY CURIE.
    Useful when parser emitted only generic types and no pinned entities.
    """
    toks = _tokenize(q)
    if not toks:
        return None
    n = min(max_n, len(toks))
    # Common function words & route words to skip
    skip = {
        "what","which","who","how","are","is","the","a","an","of","for","to","with","in",
        "and","or","vs","by","via","through","contain","contains","containing","paths",
        "path","between","associated","related","genes","gene","proteins","protein",
        "drug","drugs","chemical","chemicals","treat","treats","treating","activity",
        "upregulated","downregulated"
    }
    for k in range(n, 0, -1):  # longer first
        for i in range(0, len(toks) - k + 1):
            phrase = " ".join(toks[i:i+k]).strip(" -’'")
            if phrase.lower() in skip:
                continue
            got = _try_pin_any(phrase)
            if got:
                return got
    return None

# --- LangGraph node -------------------------------------------------------------------
def node(state: TRAPIState) -> TRAPIState:  # noqa: C901
    """
    Resolve each text in state['entities'] into TRAPI nodes.
    Mutates/returns `state` with:
      - state['nodes']: { node_id: {id?, name, category[], pinned?} }
    """
    # Normalize state['nodes'] for downstream nodes
    nodes: Dict[str, Dict[str, Any]] = state.get("nodes") or {}
    state["nodes"] = nodes

    # Track used CURIEs to avoid duplicates
    used: Set[str] = {d.get("id") for d in nodes.values() if d.get("id")}
    curie_labels: Dict[str, str] = state.setdefault("curie_labels", {})

    # 1) Normal resolution for any extracted entities
    for text in state.get("entities", []):
        tok = (text or "").strip().lower()
        hint_cat = GENERIC_CATEGORY_HINTS.get(tok)
        if hint_cat:
            # Treat this word as a schema hint, not a named entity
            g = state.setdefault("generic_types", [])
            if hint_cat not in g:
                g.append(hint_cat)
            # Do NOT try to pin this via NodeNorm
            continue

        # Try NodeNorm, then local resolver
        curie, label, _ = nn_lookup(text, mode="lookup")
        if curie and curie not in used:
            _add_pinned_node(nodes, used, curie, label or text, curie_labels)
            continue

        res = _resolver()
        if res:
            hit = res.resolve(text)
            if hit and hit.get("id") and hit["id"] not in used:
                _add_pinned_node(nodes, used, hit["id"], hit.get("name", text), curie_labels)
                continue

        # Placeholder (typed later by ResolveSchema, unless route skips schema)
        nid = f"n{len(nodes)}"
        nodes[nid] = {"name": text, "category": ["biolink:NamedThing"]}

    # 2) Route-specific: ensure a pinned Disease for 'treats'
    if (state.get("route") == "treats") and not _has_pinned_disease(nodes):
        q = state.get("query", "") or ""
        pinned: Optional[Tuple[str, str]] = None

        cand = _extract_disease_candidate(q)   # explicit pattern match
        if cand:
            pinned = _try_pin_disease(cand)

        if not pinned:                         # n-gram sweep as last resort
            pinned = _pin_disease_from_ngrams(q)

        if pinned:
            curie, label = pinned
            nid = f"n{len(nodes)}"
            nodes[nid] = {
                "id": curie,
                "name": label,
                "category": ["biolink:Disease"],  # force explicit Disease typing
                "pinned": True,
            }
            used.add(curie)
            if label:
                curie_labels[curie] = label
            logger.info("Treats fallback pinned disease: %s → %s", label, curie)
        else:
            logger.warning("Treats fallback could not pin disease from query: %r", q)

    # 3) Generic safety-net: if NOTHING is pinned yet, pin *something* from the query
    if _count_pinned(nodes) == 0:
        q = state.get("query", "") or ""
        got = _pin_any_from_ngrams(q)
        if got:
            curie, label, cat = got
            nid = f"n{len(nodes)}"
            nodes[nid] = {
                "id": curie,
                "name": label,
                "category": [cat],
                "pinned": True,
            }
            used.add(curie)
            if label:
                curie_labels[curie] = label
            logger.info("Generic fallback pinned entity: %s → %s (category=%s)", label, curie, cat)
        else:
            logger.warning("Generic fallback could not pin any entity from query: %r", q)

    return state
