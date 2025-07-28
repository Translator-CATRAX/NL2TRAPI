#!/usr/bin/env python3
"""
resolve_entities.py

LangGraph node for entity resolution:

This node takes the free-text surface strings in state['entities'] and resolves
each to a Biolink CURIE and category via a cascading strategy:

  1. SRI Node-Normalization service (fast authoritative GET /lookup).
  2. Local NodeResolver (exact index lookup → fuzzy match).
  3. Fallback placeholder as biolink:NamedThing (with original text name).

Successful resolutions are added to state['nodes'] as TRAPI query_graph nodes:
    node_id: { id: CURIE, name: label, category: [BiolinkClass] }

Inputs (state):
  - state['entities']: List[str] of surface strings
  - state.get('nodes'): existing nodes dict (may be empty)

Outputs (state):
  - state['nodes']: updated with one entry per entity
"""

from __future__ import annotations
import logging
from typing import Dict, Set

from ..state_types import TRAPIState
from ..utils.node_norm import lookup as nn_lookup
from ..utils.node_resolver import NodeResolver

# Module-level logger
logger = logging.getLogger(__name__)

# Singleton resolver instance (loads in-memory indices once)
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
    """
    Derive a Biolink category from a CURIE prefix.
    """
    prefix = curie.split(":", 1)[0]
    return PREFIX2CAT.get(prefix, "biolink:NamedThing")


def node(state: TRAPIState) -> TRAPIState:
    """
    Resolve each text in state['entities'] into a query_graph node.

    Mutates state['nodes'] to include one node per entity text.
    """
    # Ensure nodes dict exists
    nodes: Dict[str, Dict[str, Any]] = state.setdefault("nodes", {})

    # Keep track of already-used CURIEs to avoid duplicates
    used_curies: Set[str] = {
        data.get("id") for data in nodes.values() if data.get("id")
    }

    for text in state.get("entities", []):
        # 1️⃣ SRI Node-Normalization lookup
        curie, label, _ = nn_lookup(text, mode="lookup")
        if curie and curie not in used_curies:
            node_id = f"n{len(nodes)}"
            category = _infer_category(curie)
            nodes[node_id] = {"id": curie, "name": label, "category": [category]}
            used_curies.add(curie)
            logger.debug("Resolved '%s' via NodeNorm → %s", text, curie)
            continue

        # 2️⃣ Local NodeResolver (exact → fuzzy)
        hit = resolver.resolve(text)
        if hit and hit.get("id") not in used_curies:
            curie = hit["id"]
            name = hit.get("name", text)
            category = hit.get("category", "biolink:NamedThing")
            if not category.startswith("biolink:"):
                # category = f"biolink:{category.lstrip(":")}"  # enforce prefix
                category = f"biolink:{category.lstrip(':')}"
            node_id = f"n{len(nodes)}"
            nodes[node_id] = {"id": curie, "name": name, "category": [category]}
            used_curies.add(curie)
            logger.debug("Resolved '%s' via fallback → %s", text, curie)
            continue

        # 3️⃣ Fallback placeholder
        node_id = f"n{len(nodes)}"
        nodes[node_id] = {"name": text, "category": ["biolink:NamedThing"]}
        logger.debug("Placeholder node for '%s'", text)

    return state
