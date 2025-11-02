# trapi_agent/nodes/construct_treats.py
from __future__ import annotations
import logging
from typing import Dict, Any

from ..state_types import TRAPIState
from ..utils.biolink_utils import category_from_curie

logger = logging.getLogger(__name__)

REQ_PRED = "biolink:treats"
_ACCEPT_DISEASE_CATS = {
    "biolink:Disease",
    "biolink:DiseaseOrPhenotypicFeature",
}
def _pick_pinned_disease(nodes: Dict[str, Dict[str, Any]]) -> str | None:
    best = None
    for _, meta in (nodes or {}).items():
        curie = meta.get("id")
        if not curie:
            continue
        cats = set(meta.get("category") or [])
        # Strict match first
        if "biolink:Disease" in cats:
            return curie
        # Broader match as a fallback
        if not best and "biolink:DiseaseOrPhenotypicFeature" in cats:
            best = curie
        # Fallback: infer from CURIE prefix when categories are missing
        if not cats and category_from_curie(curie) == "biolink:Disease":
            return curie
    return best

# def _pick_pinned_disease(nodes: Dict[str, Dict[str, Any]]) -> str | None:
#     """
#     Find any node resolved as a Disease:
#       • has 'id' and category includes Disease (or DiseaseOrPhenotypicFeature), OR
#       • has 'id' but no category, and prefix mapping says Disease.
#     Return its CURIE or None.
#     """
#     for _, meta in (nodes or {}).items():
#         curie = meta.get("id")
#         if not curie:
#             continue
#         cats = set(meta.get("category") or [])
#         if cats & _ACCEPT_DISEASE_CATS:
#             return curie
#         # Fallback: infer from CURIE prefix when categories are missing
#         if not cats and category_from_curie(curie) == "biolink:Disease":
#             return curie
#     return None

# inside construct_treats.py
# def _pick_pinned_disease(nodes: Dict[str, Dict[str, Any]]) -> str | None:
#     for _, meta in nodes.items():
#         curie = meta.get("id") or ""
#         cats = set(meta.get("category") or [])
#         if curie.startswith("MONDO:"):
#             return curie
#         if "biolink:Disease" in cats and curie:
#             return curie
#     return None


def node(state: TRAPIState) -> TRAPIState:
    """
    Build the fixed boilerplate xDTD query_graph:

    nodes:
      sn: ChemicalEntity (un-pinned; no ids)
      on: Disease (pinned; ids = [CURIE] if available)

    edge:
      t_edge: subject=sn, object=on, predicates=[biolink:treats], knowledge_type=inferred
              + empty attribute/qualifier constraints arrays
    """
    src_nodes: Dict[str, Dict[str, Any]] = state.get("nodes", {}) or {}
    disease_id = _pick_pinned_disease(src_nodes)
    disease_name = None
    if disease_id:
        for meta in src_nodes.values():
            if meta.get("id") == disease_id:
                disease_name = meta.get("name")
                break

    # Exact shape/keys requested
    qg_nodes: Dict[str, Any] = {
        "sn": {
            "categories": ["biolink:ChemicalEntity"],
            "constraints": [],
            "is_set": False,
        },
        "on": {
            "categories": ["biolink:Disease"],
            "constraints": [],
            "is_set": False,
        },
    }
    if disease_id:
        qg_nodes["on"]["ids"] = [disease_id]
        if disease_name or disease_id:
            qg_nodes["on"]["name"] = disease_name or disease_id

    qg_edges: Dict[str, Any] = {
        "t_edge": {
            "attribute_constraints": [],
            "knowledge_type": "inferred",
            "subject": "sn",
            "object": "on",
            "predicates": [REQ_PRED],
            "qualifier_constraints": [],
        }
    }

    state["output_json"] = {
        "message": {
            "query_graph": {
                "nodes": qg_nodes,
                "edges": qg_edges,
            }
        }
    }

    if disease_id:
        logger.info("xDTD constructed: disease=%s (pinned), subject ChemicalEntity unpinned.", disease_id)
    else:
        logger.warning("xDTD constructed without a pinned disease CURIE (validator will flag).")

    return state
