#!/usr/bin/env python3
"""
construct_pathfinder_constrained.py

Build a Pathfinder-style TRAPI query_graph with a single intermediate category constraint:

  nodes: exactly two (both pinned) → {"ids": ["CURIE"]}
  paths: single p0 with predicates=["biolink:related_to"] and
         constraints:[{"intermediate_categories":[<ONE biolink:Class>]}]

Priority for ONE intermediate class:
  1) state['intermediate_category'] (string)
  2) first usable from state['intermediate_hints'] (list[str])
  3) regex hint from the NL query (e.g., "via genes", "through diseases", "contain a drug")
  4) fallback to state['generic_types'] (but skip ChemicalEntity unless query mentions drug/chemical/compound)

All candidates are canonicalized via biolink_utils.canonicalize_class().
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from ..state_types import TRAPIState
from ..utils.biolink_utils import canonicalize_class

logger = logging.getLogger(__name__)
REQ_PRED = "biolink:related_to"


# ── helpers ───────────────────────────────────────────────────────────────────

def _unique(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq or []:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _pick_two_pinned(nodes: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str | None]]:
    """Return up to 2 (node_id, CURIE, label) triples for pinned nodes."""
    seen, out = set(), []
    for nid, meta in (nodes or {}).items():
        curie = meta.get("id")
        if curie and curie not in seen:
            label = meta.get("name")
            out.append((nid, curie, label))
            seen.add(curie)
        if len(out) == 2:
            break
    return out


# Single-class, deterministic regex hints. The FIRST match wins.
_HINT_ORDER: List[Tuple[str, str]] = [
    # genes / proteins
    (r"\b(?:via|through)\s+(?:the\s+)?genes?\b",              "Gene"),
    (r"\b(?:via|through)\s+(?:the\s+)?proteins?\b",           "Protein"),
    # diseases
    (r"\b(?:via|through)\s+(?:the\s+)?diseases?\b",           "Disease"),
    # drug / chemical
    (r"\bcontain(?:s|ing)?\s+(?:a\s+)?drug\b",                "Drug"),
    (r"\b(?:via|through)\s+(?:a\s+)?drug\b",                  "Drug"),
    (r"\b(?:via|through)\s+(?:a\s+)?chemicals?\b",            "ChemicalEntity"),
    (r"\b(?:via|through)\s+(?:a\s+)?compounds?\b",            "ChemicalEntity"),
    # pathway / phenotype / anatomy
    (r"\b(?:via|through)\s+(?:the\s+)?pathways?\b",           "Pathway"),
    (r"\b(?:via|through)\s+(?:the\s+)?phenotypes?\b",         "PhenotypicFeature"),
    (r"\b(?:via|through)\s+(?:the\s+)?tissues?\b",            "AnatomicalEntity"),
    (r"\b(?:via|through)\s+anatom(?:y|ical(?:\s+entity)?)\b", "AnatomicalEntity"),
]


def _query_hint_category(query: str) -> Optional[str]:
    q = (query or "").lower()
    for pat, raw in _HINT_ORDER:
        if re.search(pat, q, flags=re.I):
            cat = canonicalize_class(raw)
            if cat:
                return cat
    return None


def _pick_single_intermediate(state: TRAPIState) -> Optional[str]:
    """
    Choose exactly ONE intermediate Biolink category.
    """
    # 1) explicit single choice (e.g., from UI)
    ui_raw = (state.get("intermediate_category") or "").strip()
    if ui_raw:
        cat = canonicalize_class(ui_raw)
        if cat:
            return cat

    # 2) optional list of hints (first usable)
    for raw in _unique(state.get("intermediate_hints", []) or []):
        cat = canonicalize_class(raw)
        if cat:
            return cat

    # 3) regex hint from query
    cat = _query_hint_category(state.get("query", ""))
    if cat:
        return cat

    # 4) fallback from generic_types, but avoid ChemicalEntity unless query mentions drug/chemical/compound
    generics = _unique(state.get("generic_types", []) or [])
    ql = (state.get("query") or "").lower()
    allow_chem = bool(re.search(r"\b(drug|chemical|compound)s?\b", ql))

    # Preferred order for informative constraints
    preferred = [
        "biolink:Gene",
        "biolink:Protein",
        "biolink:Disease",
        "biolink:Pathway",
        "biolink:PhenotypicFeature",
        "biolink:AnatomicalEntity",
        "biolink:Drug",
        "biolink:ChemicalEntity",
    ]

    for want in preferred:
        if want in generics and (want != "biolink:ChemicalEntity" or allow_chem):
            return want

    # Last-ditch: first canonicalizable thing
    for raw in generics:
        cat = canonicalize_class(raw)
        if cat:
            return cat

    return None


# ── node ──────────────────────────────────────────────────────────────────────

def node(state: TRAPIState) -> TRAPIState:
    """
    nodes:
      n0: pinned CURIE
      n1: pinned CURIE
    paths:
      p0: subject=n0, object=n1, predicates=[biolink:related_to],
          constraints=[{'intermediate_categories':[<ONE class>]}] (only if chosen)
    """
    src_nodes: Dict[str, Dict[str, Any]] = state.get("nodes", {}) or {}
    pinned = _pick_two_pinned(src_nodes)
    if len(pinned) < 2:
        logger.warning("Pathfinder-constrained needs 2 pinned nodes; found %d", len(pinned))

    # Re-key as n0/n1
    qg_nodes: Dict[str, Dict[str, Any]] = {}
    for i, (_, curie, label) in enumerate(pinned[:2]):
        nid = f"n{i}"
        node_obj: Dict[str, Any] = {"ids": [curie]}
        if label or curie:
            node_obj["name"] = label or curie
        qg_nodes[nid] = node_obj

    p0: Dict[str, Any] = {
        "subject": "n0",
        "object": "n1",
        "predicates": [REQ_PRED],
    }

    picked = _pick_single_intermediate(state)
    if picked:
        p0["constraints"] = [{"intermediate_categories": [picked]}]
        logger.info("Added intermediate_categories constraint: %s", [picked])
    else:
        logger.info("No intermediate category found; emitting unconstrained path.")

    state["output_json"] = {
        "message": {
            "query_graph": {
                "nodes": qg_nodes,
                "paths": {"p0": p0},
            }
        }
    }
    return state
