#!/usr/bin/env python3
"""
validate_trapi.py

LangGraph node for final TRAPI validation:

Checks that the constructed query_graph has at least one node,
an edge, and that the edge carries a non-empty predicates list.

Inputs (state):
  - state['output_json']['message']['query_graph']

Outputs (state):
  - state['errors']: List[str] of validation failures
  - state['valid']: bool indicating overall validity
"""

import logging
from typing import Dict, Any, List

from ..state_types import TRAPIState

# Module-level logger
logger = logging.getLogger(__name__)


def node(state: TRAPIState) -> TRAPIState:
    """
    Validate the TRAPI query_graph in state['output_json'].

    Populates:
      - state['errors']: a list of descriptive validation messages
      - state['valid']: True if no errors, False otherwise
    """
    errors: List[str] = []

    # Safely navigate to the query_graph
    qg: Dict[str, Any] = (
        state.get("output_json", {})
             .get("message", {})
             .get("query_graph", {})
    )

    # Check for presence of nodes
    nodes = qg.get("nodes")
    if not nodes:
        errors.append("No nodes in query_graph")
        logger.debug("Validation failed: no nodes found")

    # Check for presence of edges
    edges = qg.get("edges")
    if not edges:
        errors.append("No edges in query_graph")
        logger.debug("Validation failed: no edges found")
    else:
        # Ensure at least one edge has predicates
        first_edge = next(iter(edges.values()), {})
        preds = first_edge.get("predicates")
        if not preds:
            errors.append("Edge missing predicates list")
            logger.debug("Validation failed: predicates missing in edge %s", first_edge)

    # Update state
    state["errors"] = errors
    state["valid"] = not errors
    logger.info("Validation %s with errors: %s", "passed" if not errors else "failed", errors)

    return state
