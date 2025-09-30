#!/usr/bin/env python3
from __future__ import annotations

"""
fix_trapi.py — LLM-based repair of TRAPI query_graph (handles edge or pathfinder paths shapes).
"""

import json
import logging
from typing import Any, Dict, List

from ..state_types import TRAPIState
from ..prompts import FIX_TEMPLATE
from ..utils.llm import run_llm

logger = logging.getLogger(__name__)
DEF_PATHFINDER_PRED = "biolink:related_to"


def _extract_qg(obj: Any) -> Dict[str, Any]:
    """Extract a query_graph dict regardless of whether it is nested under message{} or top-level."""
    if not isinstance(obj, dict):
        return {}
    if "message" in obj and isinstance(obj.get("message"), dict):
        mq = obj["message"]
        if "query_graph" in mq and isinstance(mq["query_graph"], dict):
            return mq["query_graph"]
    if "query_graph" in obj and isinstance(obj["query_graph"], dict):
        return obj["query_graph"]
    return obj


def _ensure_edge_predicates(qg: Dict[str, Any], fallback_pred: str) -> None:
    """Normalize edge predicate fields and ensure a non-empty 'predicates' list."""
    edges = qg.get("edges") or {}
    for _, e in edges.items():
        if "predicate" in e and "predicates" not in e:
            e["predicates"] = [e.pop("predicate")]
        if not e.get("predicates"):
            e["predicates"] = [fallback_pred]


def _ensure_paths_p0(qg: Dict[str, Any], fallback_pred: str, path_nodes: List[str]) -> None:
    """Ensure paths.p0 exists and references two existing nodes with a predicate."""
    nodes = qg.get("nodes") or {}
    paths = qg.setdefault("paths", {})
    p0 = paths.get("p0")
    nids = path_nodes[:2] if len(path_nodes) >= 2 else list(nodes.keys())[:2]
    if len(nids) < 2:
        return
    if not p0:
        paths["p0"] = {"subject": nids[0], "object": nids[1], "predicates": [fallback_pred]}
    else:
        if p0.get("subject") not in nodes:
            p0["subject"] = nids[0]
        if p0.get("object") not in nodes:
            p0["object"] = nids[1]
        preds = p0.get("predicates")
        if not isinstance(preds, list) or not preds:
            p0["predicates"] = [fallback_pred]
        elif fallback_pred not in preds:
            preds.append(fallback_pred)


def node(state: TRAPIState) -> TRAPIState:
    """LLM repair step. Reads state['output_json'], writes back a corrected query_graph."""
    attempts = state.get("fix_attempts", 0) + 1
    state["fix_attempts"] = attempts
    logger.debug("Fix attempt #%d", attempts)

    errors_list = state.get("errors") or []
    broken = state.get("output_json", {})

    try:
        prompt = FIX_TEMPLATE.format(errors="\n".join(errors_list), json=json.dumps(broken, indent=2))
    except Exception:
        prompt = FIX_TEMPLATE.format(errors="\n".join(errors_list), json=str(broken))

    raw_output = run_llm(prompt, max_new_tokens=400, temperature=0.0)

    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        parsed = json.loads(raw_output[start:end])
        repaired_qg = _extract_qg(parsed)
    except Exception as err:
        logger.warning("LLM did not return valid JSON for fix: %s", err)
        repaired_qg = _extract_qg(broken)

    fallback_pred = state.get("predicate") or DEF_PATHFINDER_PRED
    if "edges" in repaired_qg:
        _ensure_edge_predicates(repaired_qg, fallback_pred)

    if "paths" in repaired_qg or (state.get("route") == "pathfinder"):
        path_nodes = state.get("path_nodes", [])
        _ensure_paths_p0(repaired_qg, DEF_PATHFINDER_PRED, path_nodes)

    state.setdefault("output_json", {}).setdefault("message", {})["query_graph"] = repaired_qg
    logger.info("Applied LLM repair; wrote corrected query_graph.")
    return state
