#!/usr/bin/env python3
"""
validate_pathfinder_constrained.py

Validate the Pathfinder-constrained query_graph shape:

   exactly 2 nodes, each with non-empty 'ids'
   paths.p0 exists and connects those nodes
   p0.predicates includes 'biolink:related_to'
   p0.constraints[0].intermediate_categories is EXACTLY ONE canonical 'biolink:*' class
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List

from ..state_types import TRAPIState
from ..utils.biolink_utils import canonicalize_class

logger = logging.getLogger(__name__)
REQ_PRED = "biolink:related_to"


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def node(state: TRAPIState) -> TRAPIState:
    errors: List[str] = []

    qg: Dict[str, Any] = (
        state.get("output_json", {})
             .get("message", {})
             .get("query_graph", {})
        or {}
    )
    nodes = qg.get("nodes", {}) or {}
    paths = qg.get("paths", {}) or {}

    # --- Nodes: exactly two, each with ids list and at least one non-empty id
    if len(nodes) != 2:
        errors.append(f"Expected exactly 2 nodes; found {len(nodes)}")

    for nk, meta in nodes.items():
        ids = meta.get("ids")
        if not isinstance(ids, list) or not ids or not _is_nonempty_str(ids[0]):
            errors.append(f"Node {nk} missing non-empty 'ids' list with first element non-empty")

    # --- Path p0: existence, subject/object, and predicate
    p0 = paths.get("p0")
    if not isinstance(p0, dict):
        errors.append("Missing paths.p0")
    else:
        sub = p0.get("subject")
        obj = p0.get("object")
        if sub not in nodes or obj not in nodes:
            errors.append("paths.p0 subject/object must reference existing nodes")

        preds = p0.get("predicates", [])
        if not isinstance(preds, list) or REQ_PRED not in preds:
            errors.append(f"paths.p0.predicates must include '{REQ_PRED}'")

        # --- Constraint: exactly one canonical Biolink class in intermediate_categories
        cons = p0.get("constraints")
        if not isinstance(cons, list) or not cons:
            errors.append("paths.p0.constraints must be a non-empty list")
        else:
            c0 = cons[0] if isinstance(cons[0], dict) else None
            ic = c0.get("intermediate_categories") if c0 else None
            if not isinstance(ic, list) or not ic:
                errors.append("constraints[0].intermediate_categories must be a non-empty list")
            else:
                if len(ic) != 1:
                    errors.append(
                        f"constraints[0].intermediate_categories must contain exactly ONE class; got {len(ic)}"
                    )
                else:
                    cat = ic[0]
                    if not _is_nonempty_str(cat) or not cat.startswith("biolink:"):
                        errors.append(
                            f"Invalid intermediate category '{cat}' (must be a canonical 'biolink:*' class)"
                        )
                    else:
                        canonical = canonicalize_class(cat)  # returns 'biolink:Class' or None
                        if not canonical:
                            errors.append(f"Unknown intermediate category '{cat}'")
                        elif canonical != cat:
                            errors.append(
                                f"Intermediate category must be canonical; got '{cat}', expected '{canonical}'"
                            )

    state["errors"] = errors
    state["valid"] = not errors
    logger.info(
        "Pathfinder-constrained validation %s",
        "passed" if not errors else f"failed: {errors}",
    )
    return state


