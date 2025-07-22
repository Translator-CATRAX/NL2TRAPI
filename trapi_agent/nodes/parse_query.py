#!/usr/bin/env python3
"""
parse_query.py

LangGraph node for initial natural-language parsing:

This node:
  • Uses SciSpaCy to detect biomedical entity spans in the input text.
  • Maps each span label to a generic Biolink type.
  • Emits state['ner_spans'], state['entities'], and state['generic_types'].
  • Constructs an LLM prompt (PARSE_TEMPLATE) with candidate predicate choices.
  • Invokes a deterministic LLM call to extract the free-text predicate.
  • Applies simple keyword-based fallbacks for common verbs.

Inputs (state):
  - state['query']: str, the user’s natural-language question
  - state['candidate_preds']: List[str], optional predicate suggestions from schema

Outputs (state):
  - state['ner_spans']: List[Dict[str, str]] of detected spans
  - state['entities']: List[str] of span texts
  - state['generic_types']: List[str] of Biolink generic types
  - state['predicate']: str, free-text predicate for resolution
"""

from __future__ import annotations
import json
import logging

import spacy

from ..prompts import PARSE_TEMPLATE
from ..state_types import TRAPIState
from ..utils.llm import run_llm

# Initialize module-level logger
logger = logging.getLogger(__name__)

# ─── 1. Load SciSpaCy model once ──────────────────────────────────────
_nlp = spacy.load("en_ner_bionlp13cg_md")
_nlp.disable_pipes("tagger", "parser", "attribute_ruler", "lemmatizer")

# Drop overly generic spans
_IGNORE_LABELS = {"GGP"}

# Map SpaCy labels to generic Biolink types
_LABEL2CLASS: dict[str, str] = {
    "GENE_OR_GENE_PRODUCT": "biolink:Gene",
    "SIMPLE_CHEMICAL":      "biolink:ChemicalEntity",
    "CHEMICAL":             "biolink:ChemicalEntity",
    "DISEASE":              "biolink:Disease",
    "ORGAN":                "biolink:AnatomicalEntity",
    "PROTEIN":              "biolink:Protein",
}


def ner_spans(text: str) -> list[dict[str, str]]:
    """
    Return raw NER spans for debugging.

    Args:
        text: The input text to annotate.

    Returns:
        A list of {'text': span, 'label': label} dicts.
    """
    doc = _nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ not in _IGNORE_LABELS
    ]


def node(state: TRAPIState) -> TRAPIState:
    """
    Parse the input query to extract entities and a free-text predicate.

    Steps:
      1. Run SciSpaCy NER on state['query'] → spans
      2. Map spans to generic_types via _LABEL2CLASS
      3. Append 'biolink:Protein' if keyword 'protein' in text
      4. Build and send PARSE_TEMPLATE to the LLM for predicate extraction
      5. Parse LLM JSON output and apply simple fallbacks

    Mutates and returns the state with new fields.
    """
    text = state.get("query", "").strip()
    if not text:
        logger.error("Empty query string in state['query']")
        return state

    # 1️⃣ NER annotation
    doc = _nlp(text)
    spans = [ent for ent in doc.ents if ent.label_ not in _IGNORE_LABELS]
    ner_list = [{"text": e.text, "label": e.label_} for e in spans]
    state["ner_spans"] = ner_list

    # 2️⃣ Extract entity texts and assign generic types
    entities = [e.text for e in spans]
    generic_types = [
        _LABEL2CLASS.get(e.label_, "biolink:NamedThing")
        for e in spans
    ]
    # Heuristic: if user mentions 'protein', ensure a protein type
    if "protein" in text.lower() and "biolink:Protein" not in generic_types:
        generic_types.append("biolink:Protein")

    state["entities"]      = entities
    state["generic_types"] = generic_types

    # 3️⃣ Build LLM prompt with predicate choices
    choices = state.get("candidate_preds", [])
    prompt = PARSE_TEMPLATE.format(
        query             = text,
        spans             = json.dumps(ner_list, indent=2),
        entities          = json.dumps(entities, indent=2),
        types             = json.dumps(generic_types, indent=2),
        predicate_choices = json.dumps(choices, indent=2),
    )

    # 4️⃣ Invoke LLM deterministically
    raw = run_llm(prompt, temperature=0.0, max_new_tokens=200)

    # 5️⃣ Extract predicate from the first JSON object
    predicate = ""
    try:
        start = raw.index("{")
        end   = raw.index("}", start) + 1
        result = json.loads(raw[start:end])
        predicate = result.get("predicate", "")
    except Exception as err:
        logger.warning("Failed to parse predicate JSON: %s", err)

    # 6️⃣ Simple keyword fallbacks
    if not predicate:
        low = text.lower()
        if "interact" in low:
            predicate = "interacts with"
        elif "treat" in low:
            predicate = "treats"

    state["predicate"] = predicate
    logger.info("Parsed %d entities and predicate '%s'", len(entities), predicate)
    return state
