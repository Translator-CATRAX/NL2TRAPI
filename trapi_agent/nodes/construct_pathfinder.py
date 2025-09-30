#!/usr/bin/env python3
"""
construct_pathfinder.py

Build a Pathfinder-style TRAPI query_graph:
  nodes: exactly two, both pinned → {"ids": ["CURIE"]}
  paths: single p0 with predicates=["biolink:related_to"]

Inputs:
  state['nodes'] : {nid: {"id"?, "pinned"?, ...}, ...}  (from ResolveEntities)

Writes:
  state['output_json']['message']['query_graph']
  state['path_nodes'] : ["nX","nY"]  (for validator)
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Tuple
from ..state_types import TRAPIState

logger = logging.getLogger(__name__)

DEF_PRED = "biolink:related_to"

# def _two_pinned(nodes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
#     """Return up to two (node_id, curie) for pinned nodes in insertion order."""
#     out: List[Tuple[str, str]] = []
#     for nid, meta in nodes.items():
#         curie = meta.get("id")
#         if curie:
#             out.append((nid, curie))
#         if len(out) == 2:
#             break
#     return out

def _two_pinned(nodes):
    seen = set(); out = []
    for nid, meta in nodes.items():
        curie = meta.get("id")
        if curie and curie not in seen:
            out.append((nid, curie))
            seen.add(curie)
        if len(out) == 2:
            break
    return out

def node(state: TRAPIState) -> TRAPIState:
    src_nodes: Dict[str, Dict[str, Any]] = state.get("nodes", {}) or {}
    pinned = _two_pinned(src_nodes)

    if len(pinned) < 2:
        # Let the validator surface a crisp error message
        logger.warning("Pathfinder needs 2 pinned nodes; found %d", len(pinned))

    # Re-key as n0/n1 in output for cleanliness
    qg_nodes: Dict[str, Dict[str, Any]] = {}
    path_nodes: List[str] = []
    for i, (_, curie) in enumerate(pinned[:2]):
        nid = f"n{i}"
        qg_nodes[nid] = {"ids": [curie]}
        path_nodes.append(nid)

    # Construct paths p0 only if we got two nodes
    qg_paths: Dict[str, Dict[str, Any]] = {}
    if len(path_nodes) == 2:
        qg_paths["p0"] = {
            "subject": path_nodes[0],
            "object":  path_nodes[1],
            "predicates": [DEF_PRED],
        }

    state["output_json"] = {
        "message": {
            "query_graph": {
                "nodes": qg_nodes,
                "paths": qg_paths
            }
        }
    }
    state["path_nodes"] = path_nodes
    logger.info("Constructed Pathfinder query_graph with %d node(s)", len(qg_nodes))
    return state
