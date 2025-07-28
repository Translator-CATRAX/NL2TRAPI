#!/usr/bin/env python3
"""
parse_query.py

LangGraph node for initial natural-language parsing.

Responsibilities:
  - Run SciSpaCy to extract biomedical entities from the question
  - Map each entity to a generic Biolink class
  - Format an LLM prompt and extract a predicate from the model output
  - Apply simple fallback rules if LLM fails

Inputs:
  - state['query']: str
  - state['candidate_preds']: optional list of predicates

Outputs:
  - state['entities']: List[str]
  - state['generic_types']: List[str]
  - state['predicate']: str
  - state['ner_spans']: List[Dict[str, str]]
"""

import json
import logging
import spacy
from ..prompts import PARSE_TEMPLATE
from ..state_types import TRAPIState
from ..utils.llm import run_llm

logger = logging.getLogger(__name__)

# ─── Load BioNER model once ──────────────────────────────────────────
_nlp = spacy.load("en_ner_bionlp13cg_md")
_nlp.disable_pipes("tagger", "parser", "attribute_ruler", "lemmatizer")

# Labels we ignore as too generic
_IGNORE_LABELS = {"GGP"}

# Mapping from SciSpaCy labels → Biolink classes
_LABEL2CLASS = {
    "GENE_OR_GENE_PRODUCT": "biolink:Gene",
    "SIMPLE_CHEMICAL":      "biolink:ChemicalEntity",
    "CHEMICAL":             "biolink:ChemicalEntity",
    "DISEASE":              "biolink:Disease",
    "ORGAN":                "biolink:AnatomicalEntity",
    "PROTEIN":              "biolink:Protein",
}


def ner_spans(text: str) -> list[dict[str, str]]:
    """Return {'text': ..., 'label': ...} spans from SciSpaCy."""
    doc = _nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ not in _IGNORE_LABELS
    ]


def node(state: TRAPIState) -> TRAPIState:
    query = state.get("query", "").strip()
    if not query:
        logger.error("Empty query string in state['query']")
        return state

    # ─── NER Step ─────────────────────────────────────────────────────
    doc = _nlp(query)
    spans = [ent for ent in doc.ents if ent.label_ not in _IGNORE_LABELS]
    ner_list = [{"text": ent.text, "label": ent.label_} for ent in spans]
    state["ner_spans"] = ner_list

    entities = [e["text"] for e in ner_list]
    generic_types = [
        _LABEL2CLASS.get(e["label"], "biolink:NamedThing")
        for e in ner_list
    ]
    if "protein" in query.lower() and "biolink:Protein" not in generic_types:
        generic_types.append("biolink:Protein")

    state["entities"] = entities
    state["generic_types"] = generic_types

    # ─── Prompt LLM for Predicate ─────────────────────────────────────
    choices = state.get("candidate_preds", [])
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
        json_str = raw_output[raw_output.index("{") : raw_output.rindex("}") + 1]
        parsed = json.loads(json_str)
        predicate = parsed.get("predicate", "").strip()
    except Exception as e:
        logger.warning("Failed to parse predicate JSON from LLM: %s", e)

    if not predicate:
        if "interact" in query.lower():
            predicate = "interacts with"
        elif "treat" in query.lower():
            predicate = "treats"
        else:
            predicate = "related to"

    state["predicate"] = predicate
    logger.info("Parsed %d entities and predicate: '%s'", len(entities), predicate)
    return state
