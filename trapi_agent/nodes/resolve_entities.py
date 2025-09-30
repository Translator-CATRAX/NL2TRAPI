#!/usr/bin/env python3
"""
resolve_entities.py

LangGraph node: resolve free-text entities into TRAPI query_graph nodes.

Policy
------
1) Pinned nodes (we have a CURIE):
   • Source the Biolink category from the authoritative CURIE→category map
     (loaded once from data/curie2cat.pkl).
   • If the map has no entry, fall back to a prefix heuristic.
   • Mark node as pinned=True.

2) Unpinned nodes (no CURIE yet):
   • Leave as generic placeholders ("biolink:NamedThing").
   • They will be typed later by the ResolveSchema step (RAG on Biolink).

This keeps schema search/RAG from overriding categories of already-identified
entities, and makes the pipeline deterministic and repeatable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from ..state_types import TRAPIState
from ..utils.node_norm import lookup as nn_lookup
from ..utils.node_resolver import NodeResolver

# Prefer the curated CURIE→category map; on absence, we fall back locally.
try:
    # If you followed the earlier recommendation, this exists:
    # trapi_agent/utils/curie2cat.py -> category_for_curie()
    from ..utils.curie2cat import category_for_curie as _map_category_for_curie
except Exception:  # pragma: no cover - defensive import
    _map_category_for_curie = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Static helpers
# ──────────────────────────────────────────────────────────────────────────────

# Lazy resolver: exact index may be absent in some environments
_RESOLVER: Optional[NodeResolver] = None


def _get_resolver() -> Optional[NodeResolver]:
    global _RESOLVER
    if _RESOLVER is None:
        try:
            _RESOLVER = NodeResolver()
        except FileNotFoundError as e:
            logger.warning(
                "NodeResolver exact index missing; local fallback disabled: %s", e
            )
            _RESOLVER = None
        except Exception as e:  # pragma: no cover - extra safety
            logger.exception("Failed to initialize NodeResolver: %s", e)
            _RESOLVER = None
    return _RESOLVER


# Minimal, safe prefix→class heuristic as the ultimate fallback only.
_PREFIX2CAT: Dict[str, str] = {
    "CHEBI": "biolink:ChemicalEntity",
    "DRUGBANK": "biolink:ChemicalEntity",
    "PUBCHEM": "biolink:ChemicalEntity",
    "NCBIGene": "biolink:Gene",
    "ENSEMBL": "biolink:Gene",
    "HGNC": "biolink:Gene",
    "MONDO": "biolink:Disease",
    "DOID": "biolink:Disease",
    "EFO": "biolink:Disease",
    "HP": "biolink:PhenotypicFeature",
    "UBERON": "biolink:AnatomicalEntity",
}


def _infer_category_from_prefix(curie: str) -> str:
    prefix = curie.split(":", 1)[0]
    return _PREFIX2CAT.get(prefix, "biolink:NamedThing")


def _category_from_curie(curie: str) -> str:
    """
    Resolve Biolink category for CURIE via the curated map; if unavailable,
    fall back to a conservative prefix heuristic.
    """
    cat: Optional[str] = None
    if _map_category_for_curie is not None:
        try:
            cat = _map_category_for_curie(curie)  # returns str | None
        except Exception:  # pragma: no cover
            logger.exception("CURIE→category map lookup failed for %s", curie)
            cat = None

    cat = cat or _infer_category_from_prefix(curie)
    # Normalize to "biolink:*"
    if not str(cat).startswith("biolink:"):
        cat = f"biolink:{str(cat).lstrip(':')}"
    return cat


# ──────────────────────────────────────────────────────────────────────────────
# LangGraph node
# ──────────────────────────────────────────────────────────────────────────────
def node(state: TRAPIState) -> TRAPIState:  # noqa: C901
    """
    Resolve each text in state['entities'] into a TRAPI query_graph node.

    Mutates/returns `state` with:
      - state['nodes']: { node_id: {id?, name, category[], pinned?} }
    """
    # Ensure downstream nodes always see a dict, even if upstream put None
    nodes: Dict[str, Dict[str, Any]] = state.get("nodes") or {}
    state["nodes"] = nodes  # write-back to normalize the type

    # Track already-used CURIEs to avoid duplicates
    used_curies: Set[str] = {d.get("id") for d in nodes.values() if d.get("id")}

    for text in state.get("entities", []):
        # 1) SRI Node-Normalization (authoritative)
        curie, label, _ = nn_lookup(text, mode="lookup")
        if curie and curie not in used_curies:
            nid = f"n{len(nodes)}"
            nodes[nid] = {
                "id": curie,
                "name": label or text,
                "category": [_category_from_curie(curie)],
                "pinned": True,
            }
            used_curies.add(curie)
            logger.debug("Resolved via NodeNorm: '%s' → %s", text, curie)
            continue

        # 2) Local resolver (exact → fuzzy), if available
        resolver = _get_resolver()
        if resolver:
            hit = resolver.resolve(text)
            if hit:
                curie = hit.get("id")
                if curie and curie not in used_curies:
                    nid = f"n{len(nodes)}"
                    nodes[nid] = {
                        "id": curie,
                        "name": hit.get("name", text),
                        "category": [_category_from_curie(curie)],
                        "pinned": True,
                    }
                    used_curies.add(curie)
                    logger.debug("Resolved via local resolver: '%s' → %s", text, curie)
                    continue

        # 3) Fallback: generic placeholder (typed later by ResolveSchema)
        nid = f"n{len(nodes)}"
        nodes[nid] = {"name": text, "category": ["biolink:NamedThing"]}
        logger.debug("Placeholder (generic) node for '%s'", text)

    return state
