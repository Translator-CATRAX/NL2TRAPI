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
import re
from typing import List, Dict

import spacy

from ..prompts import PARSE_TEMPLATE
from ..state_types import TRAPIState
from ..utils.llm import run_llm

logger = logging.getLogger(__name__)

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

    # ── NER ──────────────────────────────────────────────────────────────────────
    doc = _nlp(query)
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

    # ── Ask LLM for predicate (list of candidates may be supplied by schema step) ─
    choices = (state.get("candidate_preds") or [])[:10]
    prompt = PARSE_TEMPLATE.format(
        query=query,
        spans=json.dumps(ner_list, indent=2),
        entities=json.dumps(entities, indent=2),
        types=json.dumps(generic_types, indent=2),
        predicate_choices=json.dumps(choices, indent=2),
    )

    raw_output = run_llm(prompt, temperature=0.0, max_new_tokens=200)

    predicate = ""
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
        if "interact" in ql:
            predicate = "interacts with"
        elif "treat" in ql:
            predicate = "treats"
        else:
            predicate = "related to"

    state["predicate"] = predicate

    # ── Fallback to guarantee two entities when phrased “between X and Y” / “… X and Y …” ─
    # Route-agnostic; runs only if we currently have < 2 entity texts.
    if len(entities) < 2:
        c1 = c2 = None

        # Prefer the explicit "between X and Y" shape
        m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", query, flags=re.I)
        if m:
            c1, c2 = m.group(1), m.group(2)
        else:
            # Generic “… X and Y …” fallback
            m2 = re.search(r"\b([A-Za-z0-9\- '()/]{3,})\s+and\s+([A-Za-z0-9\- '()/]{3,})\b", query)
            if m2:
                c1, c2 = m2.group(1), m2.group(2)

        def _clean(s: str) -> str:
            return re.sub(r"^[\"'(\s]+|[)\"'\s.,:;]+$", "", s or "").strip()

        c1, c2 = _clean(c1) if c1 else None, _clean(c2) if c2 else None

        if c1 and c2:
            # Prefer the explicit pair; replace to keep it deterministic for downstream pinning
            entities = [c1, c2]
            state["entities"] = entities

    logger.info(
        "Parsed %d entities and predicate: '%s' | generic types: %s",
        len(state["entities"]), state["predicate"], state["generic_types"]
    )
    return state
