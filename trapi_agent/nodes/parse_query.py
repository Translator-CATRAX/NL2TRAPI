#!/usr/bin/env python3
"""
parse_query.py

LangGraph node for initial natural-language parsing.

Does:
  • SciSpaCy NER to extract surface entities
  • Map NER labels → Biolink classes (coarse)
  • Add lexical class hints (e.g., “proteins”, “biological processes”)
  • Prompt LLM for a predicate; sanitize & fallback if needed
  • Fallback entity extraction for “between X and Y” / “… X and Y …” when NER misses one

Inputs:
  - state['query']: str
  - state['candidate_preds']: optional list of predicate suggestions (may be absent)

Outputs (added/updated on state):
  - state['entities']: List[str]
  - state['generic_types']: List[str]  (Biolink class CURIEs)
  - state['predicate']: str            (free-text phrase; normalized later)
  - state['ner_spans']: List[Dict[str, str]]
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Dict

import spacy

from ..prompts import PARSE_TEMPLATE
from ..state_types import TRAPIState
from ..utils.biolink_utils import match_unpinned_category
from ..utils.llm import run_llm

logger = logging.getLogger(__name__)


def _env_true(name: str) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}

# ── Load BioNER once ────────────────────────────────────────────────────────────
_nlp = spacy.load("en_ner_bionlp13cg_md")
# Safely disable non-NER components if present
for pipe in ("tagger", "parser", "attribute_ruler", "lemmatizer"):
    if pipe in _nlp.pipe_names:
        _nlp.disable_pipes(pipe)

# Labels we ignore as too generic/noisy for our use
_IGNORE_LABELS = {"GGP"}

# SciSpaCy label → Biolink class
_LABEL2CLASS = {
    "GENE_OR_GENE_PRODUCT": "biolink:Gene",
    "PROTEIN":              "biolink:Protein",
    "SIMPLE_CHEMICAL":      "biolink:ChemicalEntity",
    "CHEMICAL":             "biolink:ChemicalEntity",
    "DISEASE":              "biolink:Disease",
    "ORGAN":                "biolink:AnatomicalEntity",
}

# Placeholder-y model outputs we should ignore
_PLACEHOLDER_PREDICATES = {
    "", "string", "<string>", "<predicate>", "predicate", "<pick_one>", "<value>", "<answer>"
}


def _unique(seq: List[str]) -> List[str]:
    """Preserve order, drop dupes/falsy."""
    out, seen = [], set()
    for s in seq:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _lexical_generic_hints(text: str) -> List[str]:
    """
    Heuristics that add Biolink classes from plain phrasing.
    Covers cases where NER doesn't emit a class for the 'other side' of the hop.
    """
    t = (text or "").lower()
    hints = set()

    if re.search(r"\bprotein(s)?\b", t):
        hints.add("biolink:Protein")
    if re.search(r"\bgene(s)?\b", t):
        hints.add("biolink:Gene")
    if re.search(r"\bdrug(s)?\b|\bchemical(s)?\b|\bcompound(s)?\b", t):
        hints.add("biolink:ChemicalEntity")
    if re.search(r"\bbiological process(es)?\b|\bprocess(es)?\b", t):
        hints.add("biolink:BiologicalProcess")
    if re.search(r"\bphenotype(s)?\b", t):
        hints.add("biolink:PhenotypicFeature")
    if re.search(r"\bpathway(s)?\b", t):
        hints.add("biolink:Pathway")
    if re.search(r"\btissue(s)?\b|\banatom(y|ical)\b|\borgan(s)?\b|\bcell (type|types)\b", t):
        hints.add("biolink:AnatomicalEntity")
    if re.search(r"\bdisease(s)?\b", t):
        hints.add("biolink:Disease")

    return list(hints)


def ner_spans(text: str) -> List[Dict[str, str]]:
    """Return [{'text': ..., 'label': ...}, ...] from SciSpaCy."""
    doc = _nlp(text or "")
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ not in _IGNORE_LABELS
    ]


def node(state: TRAPIState) -> TRAPIState:
    query = (state.get("query") or "").strip()
    if not query:
        logger.error("Empty query string in state['query']")
        return state
    route = state.get("route") or ""
    skip_llm = route in {"pathfinder", "pathfinder_constrained", "treats", "xcrg"}

    # For pathfinder_constrained: extract intermediate category early and strip clause
    # so endpoint NER/pinning isn't polluted by "via/through/including ..." text.
    cleaned_query = query
    if route == "pathfinder_constrained":
        m = re.search(
            r"\b(via|through|including|contains|containing|going\s+through|that\s+includes?|that\s+include)\b\s+([^,;?.]+)",
            query,
            flags=re.I,
        )
        if m:
            raw_phrase = m.group(2).strip()
            # Trim at endpoint connectors only (do NOT split on "and")
            phrase = re.split(r"\b(from|between|to)\b", raw_phrase, flags=re.I)[0].strip()
            if phrase:
                cat = match_unpinned_category(phrase, allow_fuzzy=True)
                if cat:
                    state["intermediate_category"] = cat
                    # Also provide list form for forward-compatibility if needed
                    state["intermediate_categories"] = [cat]
                    # Remove the exact matched clause span (more reliable than reconstruction)
                    cleaned_query = (query[:m.start()] + " " + query[m.end():]).strip()
                    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

    # ── NER ──────────────────────────────────────────────────────────────────────
    doc = _nlp(cleaned_query)
    spans = [ent for ent in doc.ents if ent.label_ not in _IGNORE_LABELS]
    ner_list = [{"text": ent.text, "label": ent.label_} for ent in spans]
    state["ner_spans"] = ner_list

    entities: List[str] = [e["text"] for e in ner_list]

    # Start with classes from NER labels
    generic_types = [_LABEL2CLASS.get(e["label"], "biolink:NamedThing") for e in ner_list]

    # Merge lexical hints (e.g., “proteins”, “biological processes”)
    for cls in _lexical_generic_hints(query):
        if cls not in generic_types:
            generic_types.append(cls)

    # Drop NamedThing if we have more specific types; keep list compact
    specific = [t for t in generic_types if t != "biolink:NamedThing"]
    generic_types = _unique(specific or ["biolink:NamedThing"])[:2]

    state["entities"] = entities
    state["generic_types"] = generic_types

    predicate = ""
    if skip_llm:
        state["candidate_preds"] = []
    else:
        # ── Ask LLM for predicate (seed with schema candidates when available) ─
        choices = state.get("candidate_preds")
        if not choices:
            if _env_true("NO_LLM") or (state.get("route") == "onehop"):
                choices = []
            else:
                try:
                    from .resolve_schema import _top_predicates  # local import to avoid circular on module load
                    choices = _top_predicates(query, k=10)
                except Exception:
                    choices = []
            state["candidate_preds"] = choices
        choices = (choices or [])[:10]
        prompt = PARSE_TEMPLATE.format(
            query=query,
            spans=json.dumps(ner_list, indent=2),
            entities=json.dumps(entities, indent=2),
            types=json.dumps(generic_types, indent=2),
            predicate_choices=json.dumps(choices, indent=2),
        )

        logger.debug("ParseQuery prompt:\n%s", prompt)
        raw_output = run_llm(prompt, temperature=0.0, max_new_tokens=200, allow_empty=True)
        logger.debug("ParseQuery raw LLM output: %r", raw_output)

        try:
            # Extract the outermost JSON block
            start, end = raw_output.index("{"), raw_output.rindex("}") + 1
            parsed = json.loads(raw_output[start:end])
            predicate = (parsed.get("predicate") or "").strip()
        except Exception as e:
            logger.warning("Failed to parse predicate JSON from LLM: %s", e)

        # Treat placeholders as empty to trigger fallback
        if predicate.lower() in _PLACEHOLDER_PREDICATES:
            predicate = ""

        # Lightweight fallback if model didn't return something usable
        if not predicate:
            ql = query.lower()
            if "interact" in ql or "bind" in ql or "target" in ql:
                predicate = "interacts with"
            elif "treat" in ql or "therapy" in ql:
                predicate = "treats"
            elif "express" in ql:
                predicate = "expressed in"
            elif "locat" in ql or "where" in ql:
                predicate = "located in"
            elif "pathway" in ql or "process" in ql or "participat" in ql:
                predicate = "participates in"
            else:
                predicate = "related to"

    state["predicate"] = predicate

    # ── Fallback to guarantee two entities when phrased “between X and Y” / “… X and Y …” ─
    # Route-agnostic; runs only if we currently have < 2 entity texts.
    if len(state.get("entities") or []) < 2:
        c1 = c2 = None

        # Prefer the explicit "between X and Y" shape
        m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", query, flags=re.I)
        if m:
            c1, c2 = m.group(1), m.group(2)
            if route in {"pathfinder", "pathfinder_constrained"} and c2:
                c2 = re.split(r"\b(?:that\s+includes?|including|via|through|with)\b", c2, flags=re.I)[0].strip()
        else:
            # Generic “… X and Y …” fallback
            m2 = re.search(r"\b([A-Za-z0-9\- '()/]{3,})\s+and\s+([A-Za-z0-9\- '()/]{3,})\b", query)
            if m2:
                c1, c2 = m2.group(1), m2.group(2)
                if route in {"pathfinder", "pathfinder_constrained"} and c1:
                    if re.search(r"\b(find|show|give|list|path|paths|connect|connected|relation|related|how|are|is|was|were|what|which|by)\b", c1, re.I):
                        c1 = c2 = None
                if route in {"pathfinder", "pathfinder_constrained"} and c2:
                    c2 = re.split(r"\b(?:that\s+includes?|including|via|through|with)\b", c2, flags=re.I)[0].strip()
                    c2 = re.split(r"\b(?:related|connected|connections?|associated|association|connect|paths?)\b", c2, flags=re.I)[0].strip()

        def _clean(s: str) -> str:
            return re.sub(r"^[\"'(\s]+|[)\"'\s.,:;]+$", "", s or "").strip()

        c1, c2 = _clean(c1) if c1 else None, _clean(c2) if c2 else None

        if c1 and c2:
            # Prefer the explicit pair; replace to keep it deterministic for downstream pinning
            entities = [c1, c2]
            state["entities"] = entities

    if route in {"pathfinder", "pathfinder_constrained"} and len(state.get("entities") or []) > 2:
        state["entities"] = state["entities"][:2]

    logger.info(
        "Parsed %d entities and predicate: '%s' | generic types: %s",
        len(state["entities"]), state["predicate"], state["generic_types"]
    )
    return state
