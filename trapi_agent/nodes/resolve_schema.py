

#!/usr/bin/env python3
"""
resolve_schema.py

LangGraph node for schema resolution.

This node:
  • Retrieves candidate predicates for the PARSE step via semantic Chroma lookup.
  • Converts generic Biolink types (e.g. "biolink:Protein") into concrete class nodes.
  • Avoids adding a class node that duplicates any pinned node’s category (case-insensitive).
  • Resolves the free-text predicate into a canonical Biolink predicate CURIE.
  • Uses the predicate’s domain/range to type any remaining UNPINNED node(s).

Skips work when state['skip_schema'] is True (e.g., Pathfinder, Treats).
"""
from __future__ import annotations

import logging
import re
from typing import Optional, List, Dict, Any, Set

from ..state_types import TRAPIState
from ..config import settings
from ..utils.chroma_client import get_collection
from ..utils.biolink_utils import (
    normalize_pred,                 # free-text → biolink:* CURIE
    allowed_subject_categories,     # predicate CURIE -> set[str]
    allowed_object_categories,      # predicate CURIE -> set[str]
    is_a,
    canonicalize_class,                           # subclass check: child, parent -> bool
)

logger = logging.getLogger(__name__)

# Chroma collection with Biolink classes & predicates
schema_col = get_collection(settings.YAML_SCHEMA_COLLECTION)

# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, drop leading 'biolink:'."""
    return re.sub(r"^biolink:", "", text or "", flags=re.I).strip().lower()

def _best_hit(term: str, kind: str) -> Optional[str]:
    """
    Return the best matching key of *kind* ("class" or "predicate") for *term*.

    Preference:
      1. Exact match on 'key' (case-insensitive).
      2. First neighbour of the right kind, requiring uppercase for classes.
    """
    if not term:
        return None
    try:
        res = schema_col.query(query_texts=[_normalize(term)], n_results=30)
    except Exception as e:
        logger.debug("Schema query failed: %s", e)
        return None

    metas = res.get("metadatas", [[]])[0] or []

    # Exact key match
    for meta in metas:
        key = (meta.get("key") or "").strip()
        if meta.get("category") == kind and key.lower() == _normalize(term):
            return key

    # First neighbour of requested kind
    for meta in metas:
        key = (meta.get("key") or "").strip()
        if meta.get("category") != kind:
            continue
        if kind == "class" and (not key or not key[:1].isupper()):
            continue
        return key
    return None

def _top_predicates(query: str, k: int = 10) -> List[str]:
    """Return up to k predicate keys most semantically similar to query."""
    try:
        res = schema_col.query(query_texts=[_normalize(query)], n_results=3 * k)
    except Exception as e:
        logger.debug("Predicate lookup failed: %s", e)
        return []
    metas = res.get("metadatas", [[]])[0] or []
    preds, seen = [], set()
    for meta in metas:
        if meta.get("category") != "predicate":
            continue
        key = (meta.get("key") or "").strip()
        if key and key not in seen:
            preds.append(key)
            seen.add(key)
        if len(preds) >= k:
            break
    return preds

def _prioritize_classes(candidates: list[str], query: str) -> list[str]:
    """
    Reorder to prefer entity-like targets; light lexical nudge from the question.
    Keep values as given (do not change casing).
    """
    ql = (query or "").lower()
    bias = {"biolink:Protein": 0} if re.search(r"\bprotein(s)?\b", ql) else {}

    order = [
        "biolink:Protein",
        "biolink:Gene",
        "biolink:ChemicalEntity",
        "biolink:Disease",
        "biolink:BiologicalProcess",
        "biolink:MolecularActivity",
        "biolink:Pathway",
        "biolink:AnatomicalEntity",
        "biolink:PhenotypicFeature",
        "biolink:NamedThing",
    ]
    # compare case-insensitively
    rank = {c.lower(): i for i, c in enumerate(order)}

    def key_fn(c: str) -> tuple[int, int]:
        return (bias.get(c, 1), rank.get(c.lower(), 999))

    # preserve first occurrence, then sort by rank/bias
    seen, uniq = set(), []
    for c in candidates:
        if c and c.lower() not in seen:
            uniq.append(c)
            seen.add(c.lower())
    return sorted(uniq, key=key_fn)

def _top_level(cat: str) -> str:
    """Very light top-level bucket used only for 'complementary class' bias."""
    c = cat.split(":", 1)[-1]
    if c.lower() in {"chemicalentity", "smallmolecule"}:
        return "ChemicalEntity"
    if c.lower() in {"gene", "protein", "geneorgeneproduct"}:
        return "GeneOrGeneProduct"
    if c.lower() in {"disease", "diseaseorphenotypicfeature"}:
        return "Disease"
    if c.lower() in {"biologicalprocess"}:
        return "BiologicalProcess"
    return c

def _pick_cat(generic_types: list[str], allowed: set[str], pinned_cats: set[str]) -> str:
    """
    For an UNPINNED node, choose a category preferring:
      1) a generic that fits predicate domain/range (is_a ok) and is NOT a duplicate of a pinned cat,
      2) a complementary family vs the pinned side (e.g., pick Protein if other side is ChemicalEntity),
      3) else first non-NamedThing generic,
      4) else NamedThing.

    All comparisons are case-insensitive; we return the original-cased value from generic_types.
    """
    pinned_l = {p.lower() for p in pinned_cats}
    allowed_l = {a.lower() for a in allowed}
    raw_map = {g.lower(): g for g in generic_types}
    pinned_top = {_top_level(p) for p in pinned_cats}

    def is_complement(g: str) -> bool:
        return _top_level(g) not in pinned_top

    # pass 1: allowed + complement + not duplicate of pinned
    for g in generic_types:
        gl = g.lower()
        if g == "biolink:NamedThing":
            continue
        if (gl in allowed_l or any(is_a(g, a) for a in allowed)) and gl not in pinned_l and is_complement(g):
            return raw_map[gl]

    # pass 2: allowed + not duplicate (even if same family)
    for g in generic_types:
        gl = g.lower()
        if g == "biolink:NamedThing":
            continue
        if (gl in allowed_l or any(is_a(g, a) for a in allowed)) and gl not in pinned_l:
            return raw_map[gl]

    # pass 3: first non-NamedThing that isn’t a duplicate
    for g in generic_types:
        gl = g.lower()
        if g != "biolink:NamedThing" and gl not in pinned_l:
            return raw_map[gl]

    return "biolink:NamedThing"

# ── node ──────────────────────────────────────────────────────────────────────

def node(state: TRAPIState) -> TRAPIState:
    """
    1) Populate candidate predicates for the PARSE step (schema RAG).
    2) Add a class node for the generic target (no dup of pinned categories).
    3) Resolve free-text predicate to canonical Biolink CURIE.
    4) Predicate-aware typing for remaining UNPINNED node(s).
    """
    # Route-level bypass (e.g., Pathfinder / Treats)
    if state.get("skip_schema"):
        if not (state.get("predicate") or "").startswith("biolink:"):
            state["predicate"] = "biolink:related_to"
        return state

    query = state.get("query", "")
    state.setdefault("nodes", {})
    nodes: Dict[str, Dict[str, Any]] = state["nodes"]

    # 1) candidate predicates (for parse step prompt)
    candidates = _top_predicates(query, k=10)
    state["candidate_preds"] = candidates
    logger.debug("Top-%d candidate predicates: %s", len(candidates), candidates)

    # 2) class node selection (case-insensitive duplicate checks)
    existing_cats: Set[str] = {cat for n in nodes.values() for cat in n.get("category", [])}
    pinned_cats: Set[str] = {
        cat for n in nodes.values() if "id" in n for cat in n.get("category", [])
    }

    raw_generics: list[str] = list(dict.fromkeys(state.get("generic_types", [])))
    raw_map = {g.lower(): g for g in raw_generics}  # keep original casing

    normalized: list[str] = []
    for g in raw_generics:
        key = _best_hit(g, "class")  # may return lowercase like "gene"
        if key:
            candidate = f"biolink:{key}"
            normalized.append(raw_map.get(candidate.lower(), candidate))
        else:
            normalized.append(g)

    # filter out anything that equals a pinned category (case-insensitive)
    pinned_l = {c.lower() for c in pinned_cats}
    filtered = [c for c in normalized if c.lower() not in pinned_l]

    prioritized = _prioritize_classes(filtered, query)

    # add at most ONE class node for single-hop targets
    existing_l = {c.lower() for c in existing_cats}
    for cls in prioritized:
        if cls.lower() in existing_l:
            continue
        node_id = f"n{len(nodes)}"
        nodes[node_id] = {"category": [cls], "name": ""}  # keep original casing
        logger.debug("Added class node %s as generic target", cls)
        break

    # 3) resolve predicate to canonical CURIE
    raw_pred = state.get("predicate", "")
    norm = normalize_pred(raw_pred)                    # e.g., "interacts with" → biolink:physically_interacts_with/related_to
    key = _best_hit(norm, "predicate") or norm
    key = key.replace(" ", "_").lstrip(":")
    if key.lower() in {"relation", "related", "related_to"}:
        key = "related_to"
    curie_pred = f"biolink:{key.split('biolink:')[-1]}"
    state["predicate"] = curie_pred
    logger.info("Resolved predicate '%s' → %s", raw_pred, curie_pred)

    # 4) Predicate-aware typing for any remaining UNPINNED placeholders
    try:
        gen = prioritized or normalized or ["biolink:NamedThing"]

        pinned_cats_now: Set[str] = set()
        for meta in nodes.values():
            if meta.get("id"):
                pinned_cats_now.update(meta.get("category") or [])

        allowed: Set[str] = set(allowed_subject_categories(curie_pred)) | set(
            allowed_object_categories(curie_pred)
        )

        for meta in nodes.values():
            if not meta.get("id"):  # UNPINNED placeholder
                chosen = _pick_cat(gen, allowed, pinned_cats_now)
                meta["category"] = [chosen]  # already correct casing
                logger.debug("Typed unpinned node to %s via predicate domain/range", chosen)
    except Exception as e:
        logger.debug("Predicate-aware typing skipped due to: %s", e)

    return state
