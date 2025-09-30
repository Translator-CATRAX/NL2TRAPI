#!/usr/bin/env python3
"""
validate_trapi.py

LangGraph node for final TRAPI validation (one-hop/edges flavor).

Checks:
  - query_graph exists and has ≥2 nodes
  - at least one edge exists
  - first edge has subject/object that reference existing nodes and are different
  - first edge has a non-empty 'predicates' list with at least one 'biolink:*' CURIE
  - TRAPI 1.4 node fields: 'ids' (if present) is a non-empty list of strings;
    'categories' (if present) is a list of 'biolink:*' strings

Outputs:
  - state['errors']: List[str] of validation failures
  - state['valid']: bool indicating overall validity
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from ..state_types import TRAPIState

logger = logging.getLogger(__name__)


def _is_biolink(s: Any) -> bool:
    return isinstance(s, str) and s.startswith("biolink:")


def node(state: TRAPIState) -> TRAPIState:
    errors: List[str] = []

    qg: Dict[str, Any] = (
        state.get("output_json", {})
             .get("message", {})
             .get("query_graph", {})
    )
    if not isinstance(qg, dict) or not qg:
        state["errors"] = ["Missing query_graph"]
        state["valid"] = False
        return state

    nodes = qg.get("nodes")
    edges = qg.get("edges")

    # Nodes
    if not isinstance(nodes, dict) or len(nodes) < 2:
        n = 0 if not isinstance(nodes, dict) else len(nodes or {})
        errors.append(f"Need ≥2 nodes; found {n}")

    # Edges
    if not isinstance(edges, dict) or not edges:
        errors.append("No edges in query_graph")
    else:
        e = next(iter(edges.values()))
        sub = e.get("subject")
        obj = e.get("object")

        if not isinstance(sub, str) or not isinstance(obj, str):
            errors.append("Edge subject/object must be strings")
        else:
            if isinstance(nodes, dict):
                if sub not in nodes:
                    errors.append(f"Edge.subject '{sub}' not in nodes")
                if obj not in nodes:
                    errors.append(f"Edge.object '{obj}' not in nodes")
            if sub == obj:
                errors.append("Edge subject and object must be different")

        preds = e.get("predicates")
        if not isinstance(preds, list) or not preds or not all(isinstance(p, str) for p in preds):
            errors.append("Edge predicates must be a non-empty list of strings")
        elif not any(_is_biolink(p) for p in preds):
            errors.append("At least one predicate must be a 'biolink:*' CURIE")

    # Optional: TRAPI 1.4 node shape checks
    if isinstance(nodes, dict):
        for nid, meta in nodes.items():
            if not isinstance(meta, dict):
                errors.append(f"Node {nid} must be an object")
                continue
            if "ids" in meta:
                ids = meta["ids"]
                if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
                    errors.append(f"Node {nid}.ids must be a non-empty list of CURIE strings")
            if "categories" in meta:
                cats = meta["categories"]
                if not isinstance(cats, list) or not all(_is_biolink(c) for c in cats):
                    errors.append(f"Node {nid}.categories must be a list of 'biolink:*' strings")

    state["errors"] = errors
    state["valid"] = not errors
    logger.info("Validation %s%s",
                "passed" if not errors else "failed",
                "" if not errors else f" with {len(errors)} error(s)")

    return state
