# trapi_agent/nodes/validate_treats.py
from __future__ import annotations
import logging
from typing import Dict, Any, List
from ..state_types import TRAPIState

logger = logging.getLogger(__name__)
REQ_PRED = "biolink:treats"

def node(state: TRAPIState) -> TRAPIState:
    errors: List[str] = []

    qg: Dict[str, Any] = state.get("output_json", {}).get("message", {}).get("query_graph", {})
    nodes = qg.get("nodes", {})
    edges = qg.get("edges", {})

    # Must have the exact node keys: sn (ChemicalEntity) and on (Disease)
    sn = nodes.get("sn")
    on = nodes.get("on")
    if not sn or not on:
        errors.append("Expected nodes 'sn' (ChemicalEntity) and 'on' (Disease).")

    # Categories check
    if sn and "biolink:ChemicalEntity" not in (sn.get("categories") or []):
        errors.append("Node 'sn' must have category 'biolink:ChemicalEntity'.")
    if on and "biolink:Disease" not in (on.get("categories") or []):
        errors.append("Node 'on' must have category 'biolink:Disease'.")

    # 'on' must be pinned with at least one id
    if on:
        ids = on.get("ids", [])
        if not isinstance(ids, list) or not ids or not ids[0]:
            errors.append("Node 'on' must have a non-empty 'ids' list (pinned disease).")

    # Edge shape and predicate
    t_edge = edges.get("t_edge")
    if not t_edge:
        errors.append("Missing edge 't_edge'.")
    else:
        if t_edge.get("subject") != "sn" or t_edge.get("object") != "on":
            errors.append("Edge 't_edge' must connect subject='sn' → object='on'.")
        preds = t_edge.get("predicates") or []
        if REQ_PRED not in preds:
            errors.append(f"Edge 't_edge' predicates must include '{REQ_PRED}'.")
        # Optional checks for boilerplate fields
        if t_edge.get("knowledge_type") != "inferred":
            errors.append("Edge 't_edge' must have knowledge_type='inferred'.")
        if not isinstance(t_edge.get("attribute_constraints", []), list):
            errors.append("Edge 't_edge' must include 'attribute_constraints' (list).")
        if not isinstance(t_edge.get("qualifier_constraints", []), list):
            errors.append("Edge 't_edge' must include 'qualifier_constraints' (list).")

    state["errors"] = errors
    state["valid"] = not errors
    logger.info("xDTD validation %s", "passed" if not errors else f"failed: {errors}")
    return state
