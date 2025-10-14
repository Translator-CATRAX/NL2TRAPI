from __future__ import annotations
import logging
from typing import Dict, Any, Optional

from ..state_types import TRAPIState
from ..utils.biolink_utils import category_from_curie

logger = logging.getLogger(__name__)

# Accept these as “chemical-ish” and “gene-ish” categories when deciding sides
_CHEM_CATS = {
    "biolink:ChemicalEntity", "biolink:SmallMolecule", "biolink:Drug",
    "biolink:Metabolite", "biolink:ChemicalSubstance"
}
_GENE_CATS = {"biolink:Gene", "biolink:GeneOrGeneProduct", "biolink:Protein"}

def _direction_from_query(q: str) -> str:
    ql = (q or "").lower()
    inc = ("upregulat", "increase", "activat", "induc", "stimulat", "agonist", "enhanc")
    dec = ("downregulat", "decreas", "inhibit", "repress", "reduc", "antagonist", "suppress", "block")
    if any(tok in ql for tok in inc):
        return "increased"
    if any(tok in ql for tok in dec):
        return "decreased"
    # default to “increased” if ambiguous
    return "increased"

def _find_pinned(nodes: Dict[str, Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Return (curie, category) for any pinned node if present."""
    for _, meta in (nodes or {}).items():
        curie = meta.get("id")
        if not curie:
            continue
        cats = set(meta.get("category") or [])
        if not cats:
            cats = {category_from_curie(curie)}
        # pick the first declared cat
        cat = next(iter(cats))
        return curie, cat
    return None, None

def node(state: TRAPIState) -> TRAPIState:
    """
    xCRG fixed shape:

      subject (sn): ChemicalEntity   ┐
         --[ biolink:affects ]-->    ├ with qualifier_set:
      object  (on): Gene             ┘   object_aspect_qualifier=activity_or_abundance
                                         object_direction_qualifier=increased|decreased

    Pinned→Unpinned:
      • If pinned is a Gene → put it on 'on' and make 'sn' ChemicalEntity (un-pinned)
      • If pinned is a Chemical → put it on 'sn' and make 'on' Gene (un-pinned)
    """
    src_nodes: Dict[str, Dict[str, Any]] = state.get("nodes", {}) or {}
    q = state.get("query", "") or ""

    pinned_id, pinned_cat = _find_pinned(src_nodes)
    direction = _direction_from_query(q)

    qg_nodes: Dict[str, Any] = {}

    if pinned_id and pinned_cat in _GENE_CATS:
        # Gene pinned → it must be the object
        qg_nodes["on"] = {"categories": ["biolink:Gene"], "ids": [pinned_id]}
        qg_nodes["sn"] = {"categories": ["biolink:ChemicalEntity"]}
    elif pinned_id and (pinned_cat in _CHEM_CATS or pinned_cat == "biolink:NamedThing"):
        # Chemical pinned → it must be the subject
        qg_nodes["sn"] = {"categories": ["biolink:ChemicalEntity"], "ids": [pinned_id]}
        qg_nodes["on"] = {"categories": ["biolink:Gene"]}
    else:
        # No pinned or ambiguous: prefer “chemical → gene” (most common for xCRG)
        qg_nodes["sn"] = {"categories": ["biolink:ChemicalEntity"]}
        qg_nodes["on"] = {"categories": ["biolink:Gene"]}
        if pinned_id:
            # If we had a pinned but couldn’t classify, put it on chemical side by default
            qg_nodes["sn"]["ids"] = [pinned_id]

    qg_edges: Dict[str, Any] = {
        "t_edge": {
            "subject": "sn",
            "object": "on",
            "predicates": ["biolink:affects"],
            "knowledge_type": "inferred",
            "qualifier_constraints": [
                {
                    "qualifier_set": [
                        {
                            "qualifier_type_id": "biolink:object_aspect_qualifier",
                            "qualifier_value": "activity_or_abundance",
                        },
                        {
                            "qualifier_type_id": "biolink:object_direction_qualifier",
                            "qualifier_value": direction,
                        },
                    ]
                }
            ],
        }
    }

    state["output_json"] = {"message": {"query_graph": {"nodes": qg_nodes, "edges": qg_edges}}}

    if pinned_id:
        logger.info(
            "xCRG constructed (pinned %s on %s, direction=%s).",
            pinned_id,
            "object" if "ids" in qg_nodes["on"] else "subject",
            direction,
        )
    else:
        logger.warning("xCRG constructed without a pinned CURIE; resolver didn’t find one. direction=%s", direction)

    return state
