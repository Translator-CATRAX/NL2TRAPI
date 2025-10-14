#!/usr/bin/env python3
"""
router.py

Pick a route from an explicit override (CLI/UI) or simple keyword heuristics.

Outputs:
  - state['route']  (only)
NOTE:
  - Do NOT set state['skip_schema'] here; SetRouteFlags in agent_graph does that
    using the route registry (Route.skip_schema).
  - Do NOT set state['predicate'] here; parse_query/construct_* handle it.
"""
from __future__ import annotations
import logging
from ..state_types import TRAPIState

logger = logging.getLogger(__name__)

KEYWORDS = {
    # Two-CURIE path query
    "pathfinder": [
        "path", "paths", "pathway", "between", "connected",
        "connection", "connect", "link", "links", "shortest"
    ],
    # xDTD: Drug treats Disease
    "treats": [
        "treat", "treats", "treating", "treatment", "treatments",
        "therapy for", "drug", "drugs for"
    ],
    # Chemical ↔ Gene/Protein one-hop
    "chem_gene": ["gene", "protein"],
}

def node(state: TRAPIState) -> TRAPIState:
    # 1) Respect explicit route (e.g., CLI/UI dropdown)
    if state.get("route"):
        logger.info("Router picked route=%s", state["route"])
        return state

    # 2) Infer from query keywords
    q = (state.get("query") or "").lower()
    for route, toks in KEYWORDS.items():
        if any(tok in q for tok in toks):
            state["route"] = route
            logger.info("Router picked route=%s", route)
            return state

    # 3) Default
    state["route"] = "onehop"
    logger.info("Router picked route=onehop")
    return state
