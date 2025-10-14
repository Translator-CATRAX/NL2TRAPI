
from __future__ import annotations
import logging
from typing import Dict, Any, List
from ..state_types import TRAPIState

logger = logging.getLogger(__name__)

_CHEM_CATS = {"biolink:ChemicalEntity", "biolink:SmallMolecule", "biolink:Drug",
              "biolink:Metabolite", "biolink:ChemicalSubstance"}
_GENE_CATS = {"biolink:Gene", "biolink:GeneOrGeneProduct", "biolink:Protein"}

def _has_any(cat_list: list[str] | None, target: set[str]) -> bool:
    return bool(set(cat_list or []) & target)

def node(state: TRAPIState) -> TRAPIState:
    errors: List[str] = []

    qg: Dict[str, Any] = (state.get("output_json") or {}).get("message", {}).get("query_graph", {})
    nodes = qg.get("nodes", {})
    edges = qg.get("edges", {})

    sn = nodes.get("sn")
    on = nodes.get("on")
    e = edges.get("t_edge")

    if not sn or not on or not e:
        errors.append("Expected nodes 'sn','on' and edge 't_edge'.")
        state["errors"] = errors
        state["valid"] = False
        return state

    if not _has_any(sn.get("categories"), _CHEM_CATS):
        errors.append("Node 'sn' must be ChemicalEntity-like.")
    if not _has_any(on.get("categories"), _GENE_CATS):
        errors.append("Node 'on' must be Gene-like.")

    preds = e.get("predicates") or []
    if "biolink:affects" not in preds:
        errors.append("Edge 't_edge' must have predicate 'biolink:affects'.")

    qs = e.get("qualifier_constraints") or []
    try:
        qset = (qs[0] or {}).get("qualifier_set") or []
        qmap = {q["qualifier_type_id"]: q["qualifier_value"] for q in qset}
        if qmap.get("biolink:object_aspect_qualifier") != "activity_or_abundance":
            errors.append("object_aspect_qualifier must be 'activity_or_abundance'.")
        if qmap.get("biolink:object_direction_qualifier") not in {"increased", "decreased"}:
            errors.append("object_direction_qualifier must be 'increased' or 'decreased'.")
    except Exception:
        errors.append("Malformed qualifier_constraints on edge 't_edge'.")

    # Require at least one side to be pinned (CURIE present)
    sn_ids = sn.get("ids") or []
    on_ids = on.get("ids") or []
    if not sn_ids and not on_ids:
        errors.append("xCRG expects exactly one pinned side (either 'sn' or 'on' with 'ids').")

    state["errors"] = errors
    state["valid"] = not errors
    logger.info("xCRG validation %s", "passed" if not errors else f"failed: {errors}")
    return state
