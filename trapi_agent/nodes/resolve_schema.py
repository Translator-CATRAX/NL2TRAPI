#!/usr/bin/env python3
"""
resolve_schema.py

LangGraph node for schema resolution:

This node:
  • Retrieves candidate predicates for the PARSE step via semantic Chroma lookup.
  • Converts generic Biolink types (e.g. "biolink:Protein") into concrete class nodes.
  • Ensures no duplicate or placeholder-only categories.
  • Resolves the free-text predicate into a canonical Biolink predicate CURIE.

Uses the embedded Biolink YAML in a Chroma collection defined by
settings.YAML_SCHEMA_COLLECTION.

Inputs (state):
  - state['query']: the user’s original question
  - state['generic_types']: list of free-text generic type labels
  - state['nodes']: existing partially-resolved nodes dict
  - state['predicate']: free-text predicate string

Outputs (state):
  - state['candidate_preds']: List[str] of top-k schema predicate keys
  - state['nodes']: augmented with class nodes for each generic type
  - state['predicate']: canonical Biolink predicate CURIE
"""

from __future__ import annotations
import logging
import re
from typing import Optional, List, Dict, Any, Set

from ..state_types import TRAPIState
from ..config import settings
from ..utils.chroma_client import get_collection
from ..utils.biolink_utils import normalize_pred

# Module-level logger
logger = logging.getLogger(__name__)

# Chroma collection containing classes + predicates from Biolink YAML
schema_col = get_collection(settings.YAML_SCHEMA_COLLECTION)


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

    res = schema_col.query(query_texts=[_normalize(term)], n_results=15)
    metas = res.get("metadatas", [[]])[0]
    if not metas:
        return None

    # exact key match
    for meta in metas:
        key = meta.get("key", "").strip()
        if meta.get("category") == kind and key.lower() == _normalize(term):
            return key

    # first neighbour
    for meta in metas:
        key = meta.get("key", "")
        if meta.get("category") != kind:
            continue
        if kind == "class" and not key[:1].isupper():
            continue
        return key

    return None


def _top_predicates(query: str, k: int = 10) -> List[str]:
    """Return up to k predicate keys most semantically similar to query."""
    res = schema_col.query(query_texts=[_normalize(query)], n_results=3 * k)
    metas = res.get("metadatas", [[]])[0]
    preds: List[str] = []
    seen: Set[str] = set()
    for meta in metas:
        if meta.get("category") != "predicate":
            continue
        key = meta.get("key", "")
        if key and key not in seen:
            preds.append(key)
            seen.add(key)
        if len(preds) >= k:
            break
    return preds


def node(state: TRAPIState) -> TRAPIState:
    """
    1. Populate state['candidate_preds'] for the ParseQuery prompt.
    2. Append one class node per generic type, avoiding duplicates.
    3. Prune standalone NamedThing if other classes exist.
    4. Resolve the free-text predicate to a canonical CURIE.
    """
    query = state.get("query", "")
    state.setdefault("nodes", {})

    # 1️⃣ Candidate predicates
    candidates = _top_predicates(query, k=10)
    state["candidate_preds"] = candidates
    logger.debug("Top-%d candidate predicates: %s", len(candidates), candidates)

    # 2️⃣ Add class nodes
    existing: Set[str] = {
        cat for node in state["nodes"].values() for cat in node.get("category", [])
    }
    for generic in set(state.get("generic_types", [])):
        raw_key = _best_hit(generic, "class") or "NamedThing"
        class_key = raw_key[0].upper() + raw_key[1:]
        curie = f"biolink:{class_key}"
        if curie in existing:
            continue
        node_id = f"n{len(state['nodes'])}"
        state['nodes'][node_id] = {"category": [curie]}
        existing.add(curie)
        logger.debug("Added class node %s for generic type '%s'", curie, generic)

    # 3️⃣ Prune bare NamedThing
    non_named = existing - {"biolink:NamedThing"}
    if non_named and "biolink:NamedThing" in existing:
        for nid, data in list(state['nodes'].items()):
            if data.get("category") == ["biolink:NamedThing"]:
                del state['nodes'][nid]
                logger.debug("Removed placeholder NamedThing node %s", nid)
        existing.discard("biolink:NamedThing")

    # 4️⃣ Resolve predicate
    raw = state.get("predicate", "")
    token = normalize_pred(raw)
    key = _best_hit(token, "predicate") or token
    key = normalize_pred(key).replace(" ", "_")
    curie_pred = f"biolink:{key.lstrip('biolink:')}"
    state['predicate'] = curie_pred
    logger.info("Resolved predicate '%s' → %s", raw, curie_pred)

    return state
