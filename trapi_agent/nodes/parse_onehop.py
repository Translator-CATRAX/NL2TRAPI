#!/usr/bin/env python3
"""
parse_onehop.py

Specialised LangGraph node for one-hop parsing.

Responsibilities:
  • Re-run a constrained extraction tailored to single-hop questions.
  • Honour pinned nodes resolved earlier (via ResolveEntities).
  • Produce subject/object texts and a predicate phrase, along with
    a short candidate predicate list for downstream schema grounding.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from ..state_types import TRAPIState
from ..utils.biolink_utils import match_unpinned_category
from ..utils.predicate_grounder import ground_onehop_predicate
from .parse_query import ner_spans, _lexical_generic_hints
from .resolve_entities import pin_any_from_query

logger = logging.getLogger(__name__)

_UNPINNED_STRIP_PATTERNS = [
    r"\bprotein(s)?\b",
    r"\bgene(s)?\b",
    r"\bdrug(s)?\b",
    r"\bchemical(s)?\b",
    r"\bcompound(s)?\b",
    r"\bmetabolite(s)?\b",
    r"\bdisease(s)?\b",
    r"\bphenotype(s)?\b",
    r"\bsymptom(s)?\b",
    r"\bpathway(s)?\b",
    r"\bbiological process(es)?\b",
    r"\bprocess(es)?\b",
    r"\bmolecular activity(ies)?\b",
    r"\btissue(s)?\b",
    r"\banatom(y|ical)( entity)?\b",
    r"\borgan(s)?\b",
    r"\bcell line(s)?\b",
    r"\bcell(s)?\b",
]


def _best_category(cats: Optional[List[str]]) -> str:
    for cat in cats or []:
        if cat:
            return cat
    return "biolink:NamedThing"


def _fallback_predicate(query: str) -> str:
    ql = query.lower()
    if any(tok in ql for tok in ("interact", "bind", "target", "partner", "associate with")):
        return "interacts with"
    if any(tok in ql for tok in ("treat", "therapy", "treats", "drug for")):
        return "treats"
    if "express" in ql:
        return "expressed in"
    if any(tok in ql for tok in ("pathway", "process", "particip", "involved")):
        return "participates in"
    if any(tok in ql for tok in ("locat", "where", "found in", "present in")):
        return "located in"
    return "related to"


def _fallback_entities(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract simple 'between X and Y' / 'X and Y' patterns."""
    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", query, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"\b([A-Za-z0-9\- '()/]{3,})\s+and\s+([A-Za-z0-9\- '()/]{3,})\b", query)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _ensure_list_length(items: List[str], length: int, fill: str = "biolink:NamedThing") -> List[str]:
    out = list(items)
    while len(out) < length:
        out.append(fill)
    return out[:length]

def _query_minus_unpinned(query: str, unpinned_text: str) -> str:
    cleaned = query or ""
    term = (unpinned_text or "").strip()
    if term:
        if term.lower().startswith("biolink:"):
            term = term.split(":", 1)[1].replace("_", " ")
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.I)
    for pat in _UNPINNED_STRIP_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _query_minus_entities(query: str, pinned_text: str, unpinned_text: str) -> str:
    cleaned = _query_minus_unpinned(query, unpinned_text)
    term = (pinned_text or "").strip()
    if term:
        if term.lower().startswith("biolink:"):
            term = term.split(":", 1)[1].replace("_", " ")
        cleaned = re.sub(re.escape(term), " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ground_text_to_class(text: str, lex_hints: List[str]) -> str:
    """
    Map free-text descriptions to a Biolink class using lexical cues and query hints.
    """
    match = match_unpinned_category(text, allow_fuzzy=True)
    if match:
        return match

    priority = [
        "biolink:Protein",
        "biolink:Gene",
        "biolink:ChemicalEntity",
        "biolink:Disease",
        "biolink:BiologicalProcess",
        "biolink:Pathway",
        "biolink:AnatomicalEntity",
    ]
    for hint in priority:
        if hint in lex_hints:
            return hint

    return "biolink:NamedThing"


def node(state: TRAPIState) -> TRAPIState:
    """One-hop specific parsing (only runs when route == 'onehop')."""
    if (state.get("route") or "onehop") != "onehop":
        return state

    query = (state.get("query") or "").strip()
    if not query:
        return state

    # Gather pinned nodes (resolved earlier by ResolveEntities)
    nodes: Dict[str, Dict[str, Any]] = state.get("nodes") or {}
    pinned_nodes = [meta for meta in nodes.values() if meta.get("id")]
    pinned_names = [meta.get("name") or "" for meta in pinned_nodes if (meta.get("name") or "").strip()]

    spans = ner_spans(query)
    lex_hints = _lexical_generic_hints(query)
    pinned_lower = {name.lower() for name in pinned_names}
    span_unp_candidates = [
        span["text"] for span in spans
        if span.get("text") and span["text"].strip().lower() not in pinned_lower
    ]

    # For one-hop we don't require schema embedding retrieval; RENCI covers grounding.
    candidate_preds = state.get("candidate_preds") or []
    state["candidate_preds"] = candidate_preds
    first_line = ""
    pinned_text = ""
    partner_text = ""
    predicate_phrase = ""

    if not pinned_text and pinned_names:
        pinned_text = pinned_names[0]
    if pinned_text and pinned_names:
        for name in pinned_names:
            if pinned_text.lower() == name.lower():
                pinned_text = name
                break

    if not partner_text and span_unp_candidates:
        partner_text = span_unp_candidates[0]

    if not partner_text and lex_hints:
        priority = [
            "biolink:Protein",
            "biolink:Gene",
            "biolink:ChemicalEntity",
            "biolink:Disease",
            "biolink:BiologicalProcess",
            "biolink:Pathway",
            "biolink:AnatomicalEntity",
        ]
        chosen = next((h for h in priority if h in lex_hints), lex_hints[0])
        partner_text = chosen.split(":", 1)[-1].replace("_", " ")

    if not partner_text:
        c1, c2 = _fallback_entities(query)
        if c1 and c2:
            if pinned_text and pinned_text.lower() == c1.lower():
                partner_text = c2
            elif pinned_text and pinned_text.lower() == c2.lower():
                partner_text = c1
            else:
                partner_text = c2
                if not pinned_text:
                    pinned_text = c1

    if not partner_text and pinned_names:
        placeholder_tokens = {"proteins", "genes", "drugs", "tissues", "diseases"}
        partner_text = " ".join(set(query.lower().split()) & placeholder_tokens) or "entity"
    if not partner_text:
        partner_text = "entity"

    state["query_minus_unpinned"] = _query_minus_unpinned(query, partner_text)

    if not pinned_nodes:
        candidate = pin_any_from_query(state["query_minus_unpinned"])
        if candidate:
            nodes = state.get("nodes") or {}
            nid = f"n{len(nodes)}"
            nodes[nid] = candidate
            state["nodes"] = nodes
            pinned_nodes = [candidate]
            pinned_names = [candidate.get("name") or ""]
            pinned_lower = {name.lower() for name in pinned_names if name}
            if not pinned_text and pinned_names:
                pinned_text = pinned_names[0]

    state["query_minus_entities"] = _query_minus_entities(query, pinned_text, partner_text)
    relationship_text = state.get("query_minus_entities") or ""
    predicate_phrase = relationship_text or _fallback_predicate(query)
    raw_predicate_text = predicate_phrase

    subj_text = pinned_text or ""
    obj_text = partner_text or ""
    subj_role = "pinned" if subj_text.lower() in pinned_lower else "unpinned"
    obj_role = "pinned" if obj_text.lower() in pinned_lower else "unpinned"

    if subj_role != "pinned" and pinned_names:
        subj_text = pinned_names[0]
        subj_role = "pinned"

    if obj_role == "pinned" and subj_role == "pinned" and obj_text.lower() == subj_text.lower():
        obj_role = "unpinned"

    # Swap if the pinned entity is on the object slot
    subject_text = subj_text
    object_text = obj_text
    if subj_role == "unpinned" and obj_role == "pinned":
        subject_text, object_text = object_text, subject_text
        subj_role, obj_role = "pinned", "unpinned"

    def _pinned_categories(name: str) -> List[str]:
        norm = (name or "").strip().lower()
        for meta in pinned_nodes:
            if (meta.get("name") or "").strip().lower() == norm:
                return meta.get("category") or []
        return pinned_nodes[0].get("category") if pinned_nodes else []

    subject_cat = (
        _best_category(_pinned_categories(subject_text))
        if subj_role == "pinned"
        else _ground_text_to_class(subject_text, lex_hints)
    )
    object_cat = (
        _best_category(_pinned_categories(object_text))
        if obj_role == "pinned"
        else _ground_text_to_class(object_text, lex_hints)
    )

    relationship_text = relationship_text or predicate_phrase
    candidate_preds = candidate_preds or []
    state["candidate_preds"] = candidate_preds
    grounded = ground_onehop_predicate(
        relationship_text=relationship_text,
        subject_category=subject_cat,
        object_category=object_cat,
        subject_text=subject_text,
        object_text=object_text,
        context_text=query,
        candidate_preds=candidate_preds,
    )
    predicate_curie = grounded["predicate"]
    if grounded.get("swap"):
        subject_text, object_text = object_text, subject_text
        subject_cat, object_cat = object_cat, subject_cat
        subj_role, obj_role = obj_role, subj_role

    def _find_node_id_by_name(nodes_map: Dict[str, Dict[str, Any]], name: str) -> Optional[str]:
        norm = (name or "").strip().lower()
        if not norm:
            return None
        for nid, meta in nodes_map.items():
            if (meta.get("name") or "").strip().lower() == norm:
                return nid
        return None

    def _ensure_unpinned_node(text: str, category: str) -> None:
        if not text:
            return
        if _find_node_id_by_name(nodes, text):
            return
        nid = f"n{len(nodes)}"
        nodes[nid] = {
            "name": text,
            "category": [category or "biolink:NamedThing"],
        }

    nodes = state.get("nodes") or {}
    if subj_role == "unpinned":
        _ensure_unpinned_node(subject_text, subject_cat)
    if obj_role == "unpinned":
        _ensure_unpinned_node(object_text, object_cat)
    state["nodes"] = nodes

    state["entities"] = [subject_text, object_text]
    state["generic_types"] = _ensure_list_length(
        [subject_cat or "biolink:NamedThing", object_cat or "biolink:NamedThing"],
        2,
        "biolink:NamedThing",
    )
    state["predicate"] = predicate_curie
    state["onehop_parse"] = {
        "subject_text": subject_text,
        "object_text": object_text,
        "predicate_text": predicate_curie,
        "raw_predicate_text": raw_predicate_text,
        "query_minus_entities": relationship_text,
        "predicate_grounding": grounded,
        "subject_role": subj_role,
        "object_role": obj_role,
        "candidate_preds": candidate_preds,
        "raw_llm_line": first_line,
    }

    logger.info(
        "ParseOneHop extracted: subject='%s' (%s), object='%s' (%s), predicate='%s'",
        subject_text,
        subj_role,
        object_text,
        obj_role,
        predicate_curie,
    )
    return state
