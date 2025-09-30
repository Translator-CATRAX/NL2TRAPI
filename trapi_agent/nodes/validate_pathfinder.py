#!/usr/bin/env python3
"""
validate_pathfinder.py

Validate the Pathfinder query_graph shape.

Checks:
  - exactly 2 nodes
  - each node has non-empty 'ids' list
  - 'paths' has p0 connecting the two nodes
  - p0.predicates contains 'biolink:related_to'
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List
from ..state_types import TRAPIState

logger = logging.getLogger(__name__)
REQ_PRED = "biolink:related_to"

def node(state: TRAPIState) -> TRAPIState:
    errors: List[str] = []
    qg: Dict[str, Any] = state.get("output_json", {}).get("message", {}).get("query_graph", {})

    nodes = qg.get("nodes", {})
    paths = qg.get("paths", {})

    if len(nodes) != 2:
        errors.append(f"Expected exactly 2 nodes; found {len(nodes)}")

    # collect node ids and check ids lists
    node_keys = list(nodes.keys())
    for nk in node_keys:
        ids = nodes.get(nk, {}).get("ids")
        if not isinstance(ids, list) or not ids or not ids[0]:
            errors.append(f"Node {nk} missing non-empty 'ids'")

    p0 = paths.get("p0")
    if not p0:
        errors.append("Missing paths.p0")
    else:
        sub = p0.get("subject"); obj = p0.get("object")
        if sub not in nodes or obj not in nodes:
            errors.append("paths.p0 subject/object must reference existing nodes")
        preds = p0.get("predicates", [])
        if REQ_PRED not in preds:
            errors.append(f"paths.p0.predicates must include '{REQ_PRED}'")

    state["errors"] = errors
    state["valid"] = not errors
    logger.info("Pathfinder validation %s", "passed" if not errors else f"failed: {errors}")
    return state
