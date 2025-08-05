
from __future__ import annotations

import logging
from typing import Any, Dict, Set

from ..state_types import TRAPIState
from ..utils.node_norm import lookup as nn_lookup
from ..utils.node_resolver import NodeResolver

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Static helpers
# --------------------------------------------------------------------- #

# Single NodeResolver instance (loads indices once)
resolver = NodeResolver()

# Quick heuristic: CURIE prefix → Biolink category
PREFIX2CAT: Dict[str, str] = {
    "CHEBI":    "biolink:ChemicalEntity",
    "DRUGBANK": "biolink:ChemicalEntity",
    "PUBCHEM":  "biolink:ChemicalEntity",
    "NCBIGene": "biolink:Gene",
    "ENSEMBL":  "biolink:Gene",
    "HGNC":     "biolink:Gene",
    "MONDO":    "biolink:Disease",
    "DOID":     "biolink:Disease",
    "EFO":      "biolink:Disease",
    "HP":       "biolink:PhenotypicFeature",
    "UBERON":   "biolink:AnatomicalEntity",
}


def _infer_category(curie: str) -> str:
    """Return a Biolink class given a CURIE prefix."""
    prefix = curie.split(":", 1)[0]
    return PREFIX2CAT.get(prefix, "biolink:NamedThing")


# --------------------------------------------------------------------- #
# LangGraph node
# --------------------------------------------------------------------- #
def node(state: TRAPIState) -> TRAPIState:  # noqa: C901
    """
    Resolve each entity string into a query-graph node and update ``state``.
    """
    # ── Ensure we start with a *dict*, even if upstream filled the key with None
    nodes: Dict[str, Dict[str, Any]] = state.get("nodes") or {}
    state["nodes"] = nodes  # write-back guarantees downstream nodes get a dict

    # Track CURIEs already used to avoid duplicates
    used_curies: Set[str] = {
        data.get("id") for data in nodes.values() if data.get("id")
    }

    for text in state.get("entities", []):
        # 1️⃣  Node-Normalization service
        curie, label, _ = nn_lookup(text, mode="lookup")
        if curie and curie not in used_curies:
            node_id = f"n{len(nodes)}"
            nodes[node_id] = {
                "id": curie,
                "name": label,
                "category": [_infer_category(curie)],
            }
            used_curies.add(curie)
            logger.debug("Resolved '%s' via NodeNorm → %s", text, curie)
            continue

        # 2️⃣  Local resolver (exact → fuzzy)
        hit = resolver.resolve(text)
        if hit and hit.get("id") not in used_curies:
            curie = hit["id"]
            category = hit.get("category", "biolink:NamedThing")
            if not category.startswith("biolink:"):
                category = f"biolink:{category.lstrip(':')}"
            node_id = f"n{len(nodes)}"
            nodes[node_id] = {
                "id": curie,
                "name": hit.get("name", text),
                "category": [category],
            }
            used_curies.add(curie)
            logger.debug("Resolved '%s' via local fallback → %s", text, curie)
            continue

        # 3️⃣  Fallback placeholder
        node_id = f"n{len(nodes)}"
        nodes[node_id] = {"name": text, "category": ["biolink:NamedThing"]}
        logger.debug("Placeholder node for '%s'", text)

    return state
