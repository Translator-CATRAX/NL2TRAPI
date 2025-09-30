
#!/usr/bin/env python3
"""
resolve_schema.py

LangGraph node for schema resolution.

This node:
  • Retrieves candidate predicates for the PARSE step via semantic Chroma lookup.
  • Converts generic Biolink types (e.g. "biolink:Protein") into concrete class nodes.
  • Avoids adding a class node that duplicates any pinned node’s category.
  • Resolves the free-text predicate into a canonical Biolink predicate CURIE.

Inputs (state):
  - state['query']: the user’s original question
  - state['generic_types']: list of Biolink class CURIEs (from parse_query)
  - state['nodes']: existing nodes dict (pinned entity likely present)
  - state['predicate']: free-text predicate string

Outputs (state):
  - state['candidate_preds']: List[str] of top-k schema predicate keys
  - state['nodes']: augmented with class nodes for generic types (filtered/prioritized)
  - state['predicate']: canonical Biolink predicate CURIE

Note: If state['skip_schema'] is True (e.g., Pathfinder), this node no-ops.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, List, Dict, Any, Set

from ..state_types import TRAPIState
from ..config import settings
from ..utils.chroma_client import get_collection
from ..utils.biolink_utils import normalize_pred

logger = logging.getLogger(__name__)

# Chroma collection containing classes + predicates from Biolink YAML
schema_col = get_collection(settings.YAML_SCHEMA_COLLECTION)

# --- helpers -------------------------------------------------------------------

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

    # exact key match
    for meta in metas:
        key = (meta.get("key") or "").strip()
        if meta.get("category") == kind and key.lower() == _normalize(term):
            return key

    # first neighbour of that kind
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

def _prioritize_classes(candidates: List[str], query: str) -> List[str]:
    """
    Reorder to prefer entity-like targets; light lexical nudge from the question.
    """
    ql = (query or "").lower()
    # lexical nudge
    if re.search(r"\bprotein(s)?\b", ql):
        bias = {"biolink:Protein": 0}
    else:
        bias = {}
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
    rank = {c: i for i, c in enumerate(order)}

    def key_fn(c: str) -> tuple[int, int]:
        return (bias.get(c, 1), rank.get(c, 999))

    return sorted(dict.fromkeys(candidates), key=key_fn)

# --- node ----------------------------------------------------------------------

def node(state: TRAPIState) -> TRAPIState:
    """
    1) Populate candidate predicates for the PARSE step (RAG over schema).
    2) Add a class node for a generic target (unless a pinned node already has that category).
    3) Resolve the free-text predicate to a canonical Biolink CURIE.

    If state['skip_schema'] is True (e.g., Pathfinder), this function no-ops.
    """
    # Route-level bypass: e.g., Pathfinder doesn't need schema work
    if state.get("skip_schema"):
        # still ensure predicate exists so downstream constructors have a safe default
        if not (state.get("predicate") or "").startswith("biolink:"):
            state["predicate"] = "biolink:related_to"
        return state

    query = state.get("query", "")
    state.setdefault("nodes", {})
    nodes: Dict[str, Dict[str, Any]] = state["nodes"]

    # 1) candidate predicates for the prompt
    candidates = _top_predicates(query, k=10)
    state["candidate_preds"] = candidates
    logger.debug("Top-%d candidate predicates: %s", len(candidates), candidates)

    # 2) add class node(s), but do NOT duplicate any pinned node's category
    existing_cats: Set[str] = {cat for n in nodes.values() for cat in n.get("category", [])}
    pinned_cats: Set[str] = {
        cat for n in nodes.values() if "id" in n for cat in n.get("category", [])
    }

    # normalize incoming generic types to true class keys via schema, when possible
    raw_generics: List[str] = list(dict.fromkeys(state.get("generic_types", [])))
    normalized: List[str] = []
    for g in raw_generics:
        key = _best_hit(g, "class")
        if key:
            class_key = key  # use exact Biolink casing
        else:
            suffix = g.split(":", 1)[-1].strip()
            class_key = suffix[:1].upper() + suffix[1:]  # preserve rest of casing
        normalized.append(f"biolink:{class_key}")

    # filter out anything that equals a pinned category
    filtered = [c for c in normalized if c not in pinned_cats]

    # prioritize (Protein > Gene > ChemicalEntity …), with a small lexical nudge
    prioritized = _prioritize_classes(filtered, query)

    # add at most ONE class node for single-hop targets (keeps graph minimal)
    for cls in prioritized:
        if cls in existing_cats:
            continue
        node_id = f"n{len(nodes)}"
        nodes[node_id] = {"category": [cls], "name": ""}
        logger.debug("Added class node %s as generic target", cls)
        break  # single generic target for 1-hop

    # 3) resolve predicate to canonical CURIE
    raw_pred = state.get("predicate", "")
    norm = normalize_pred(raw_pred)                      # e.g., "interacts with" → related_to
    key = _best_hit(norm, "predicate") or norm
    key = key.replace(" ", "_").lstrip(":")

    # force canonical relatedness
    low = key.lower()
    if low in {"relation", "related", "related_to"}:
        key = "related_to"

    curie_pred = f"biolink:{key.split('biolink:')[-1]}"
    state["predicate"] = curie_pred
    logger.info("Resolved predicate '%s' → %s", raw_pred, curie_pred)

    return state
