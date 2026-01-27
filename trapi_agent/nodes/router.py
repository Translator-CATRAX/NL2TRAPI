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
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..state_types import TRAPIState

logger = logging.getLogger(__name__)

ROUTE_CHOICES = {
    "onehop",
    "pathfinder",
    "pathfinder_constrained",
    "treats",
    "chem_gene",
    "xcrg",
    "multihop",
}

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


@lru_cache(maxsize=1)
def _load_router_model() -> Optional[object]:
    """
    Load a pre-trained router model if available.
    Path order:
      1) env ROUTER_MODEL
      2) repo data/router_model.pkl
    """
    candidates = []
    env_path = os.getenv("ROUTER_MODEL")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "router_model.pkl")

    for p in candidates:
        try:
            if p.exists():
                with p.open("rb") as fh:
                    return pickle.load(fh)
        except Exception as e:
            logger.warning("Router model load failed (%s): %s", p, e)
    return None


def _predict_route(model: object, query: str) -> Tuple[str, float]:
    """
    Predict route and confidence using a scikit-learn style pipeline.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([query])[0]
        conf = float(np.max(probs))
    else:
        scores = model.decision_function([query])
        exps = np.exp(scores - np.max(scores))
        probs = exps / exps.sum()
        conf = float(np.max(probs))
    label = model.predict([query])[0]
    return str(label), conf


def node(state: TRAPIState) -> TRAPIState:
    # 1) Respect explicit route (e.g., CLI/UI dropdown)
    if state.get("route"):
        logger.info("Router picked route=%s", state["route"])
        return state

    # 2) Model-based routing (if available and confident)
    q = (state.get("query") or "").strip()
    model = _load_router_model()
    if model and q:
        try:
            label, conf = _predict_route(model, q)
            threshold = float(os.getenv("ROUTER_CONF_MIN", "0.55"))
            if label in ROUTE_CHOICES and conf >= threshold:
                state["route"] = label
                logger.info("Router picked route=%s (model, conf=%.2f)", label, conf)
                return state
            logger.info("Router model low confidence (%.2f), falling back to heuristics", conf)
        except Exception as e:
            logger.warning("Router model prediction failed: %s", e)

    # 3) Infer from query keywords
    q = (state.get("query") or "").lower()
    for route, toks in KEYWORDS.items():
        if any(tok in q for tok in toks):
            state["route"] = route
            logger.info("Router picked route=%s", route)
            return state

    # 4) Default
    state["route"] = "onehop"
    logger.info("Router picked route=onehop")
    return state
