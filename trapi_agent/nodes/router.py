#!/usr/bin/env python3
"""
router.py

Choose the route based on CLI override or keyword heuristics.
Outputs:
  - state['route']
  - state['skip_schema'] (True only for 'pathfinder')
"""
from __future__ import annotations
import logging
from ..state_types import TRAPIState

logger = logging.getLogger(__name__)

KEYWORDS = {
    "pathfinder": ["path", "paths", "pathway", "connected"],
    "treats": ["treat", "treats", "therapy for"],
    "chem_gene": ["gene", "protein"],
}

def node(state: TRAPIState) -> TRAPIState:
    # 1) Respect explicit route (e.g., CLI)
    if state.get("route"):
        state["skip_schema"] = (state["route"] == "pathfinder")
        logger.info("Router picked route=%s skip_schema=%s", state["route"], state["skip_schema"])
        return state

    # 2) Infer from query keywords
    q = (state.get("query") or "").lower()
    for route, toks in KEYWORDS.items():
        if any(tok in q for tok in toks):
            state["route"] = route
            state["skip_schema"] = (route == "pathfinder")
            logger.info("Router picked route=%s skip_schema=%s", state["route"], state["skip_schema"])
            return state

    # 3) Default
    state["route"] = "onehop"
    state["skip_schema"] = False
    logger.info("Router picked route=%s skip_schema=%s", state["route"], state["skip_schema"])
    return state
