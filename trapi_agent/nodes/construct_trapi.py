#!/usr/bin/env python3
"""
construct_trapi.py

LangGraph node for assembling a minimal TRAPI query_graph from resolved nodes and predicate.

This node:
  • Validates that at least two nodes exist.
  • Selects subject and object via a simple heuristic:
      - First node without an explicit 'id' (unresolved) → subject
      - Next available node → object
      - Falls back to the first two nodes if needed.
  • Ensures each node dict has a 'name' key (empty string if missing).
  • Builds an edge with a TRAPI-compliant 'predicates' list (v1.4+).

Inputs (state):
  - state['nodes']: Dict[node_id, {'id'? str, 'name'? str, 'category': List[str]}]
  - state['predicate']: str (biolink CURIE)

Outputs (state):
  - state['edges']: Dict[edge_id, {'subject': node_id, 'object': node_id, 'predicates': [str]}]
  - state['output_json']: { 'message': { 'query_graph': {...} }}
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

from ..state_types import TRAPIState

logger = logging.getLogger(__name__)


def node(state: TRAPIState) -> TRAPIState:
    """
    Assemble a TRAPI query_graph from resolved nodes and predicate.

    Args:
        state: The TRAPIState dict, expecting 'nodes' and 'predicate'.

    Returns:
        The updated state with 'edges' and 'output_json' set.
    """
    nodes_dict: Dict[str, Dict[str, Any]] = state.get("nodes", {})

    # Need at least two nodes to form an edge
    if len(nodes_dict) < 2:
        logger.warning("Insufficient nodes to construct edge: %d found", len(nodes_dict))
        state["output_json"] = {}
        return state

    # Subject/object selection
    subj_id: str | None = None
    obj_id: str | None = None
    for node_id, metadata in nodes_dict.items():
        if "id" not in metadata and subj_id is None:
            subj_id = node_id
        elif obj_id is None:
            obj_id = node_id
        if subj_id and obj_id:
            break
    # Fallback to first two nodes
    node_keys = list(nodes_dict.keys())
    subj_id = subj_id or node_keys[0]
    obj_id = obj_id or node_keys[1]
    logger.debug("Selected subject '%s', object '%s'", subj_id, obj_id)

    # Ensure 'name' field exists on each node (for UI convenience)
    for metadata in nodes_dict.values():
        metadata.setdefault("name", "")

    # Build the single edge
    predicate_curie: str = state.get("predicate", "biolink:related_to")
    edge_id = "e0"
    edges: Dict[str, Dict[str, Any]] = {
        edge_id: {
            "subject": subj_id,
            "object": obj_id,
            "predicates": [predicate_curie],
        }
    }

    # Write back to state
    state["edges"] = edges
    state["output_json"] = {
        "message": {
            "query_graph": {
                "nodes": nodes_dict,
                "edges": edges,
            }
        }
    }
    logger.info("Constructed query_graph with %d nodes and 1 edge", len(nodes_dict))
    return state
