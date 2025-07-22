#!/usr/bin/env python3
"""
retrieve_examples.py

LangGraph node for few-shot example retrieval:

Fetches up to N NL→TRAPI examples from a Chroma collection and
stores them in state["retrieved_examples"].

Inputs:
  - state['query']: the user's free-text question

Outputs:
  - state['retrieved_examples']: List of example dicts
      { nl_text: str, trapi_json: dict }
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Any

from ..state_types import TRAPIState
from ..config import settings
from ..utils.chroma_client import get_collection

# Module-level logger
logger = logging.getLogger(__name__)

# Lazy-load the Chroma collection for examples
_examples_col = get_collection(settings.EXAMPLE_COLLECTION)


def _parse_examples(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert raw Chroma query output into a list of examples.

    Each example contains:
      - 'nl_text': the original NL query (str)
      - 'trapi_json': the parsed TRAPI JSON (dict)

    Silently skips entries with invalid JSON.
    """
    examples: List[Dict[str, Any]] = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas",  [[]])[0]

    for text, meta in zip(docs, metas):
        raw_json = meta.get("trapi_query")
        if not raw_json:
            continue
        try:
            parsed = json.loads(raw_json)
            examples.append({"nl_text": text, "trapi_json": parsed})
        except json.JSONDecodeError as e:
            logger.warning("Failed to decode example TRAPI JSON: %s", e)
    return examples


def node(state: TRAPIState) -> TRAPIState:
    """
    Retrieve up to 3 few-shot NL→TRAPI examples for the user's query.

    Populates:
      state['retrieved_examples'] -> List[{'nl_text': str, 'trapi_json': dict}]
    """
    query = state.get("query", "")
    if not query:
        state["retrieved_examples"] = []
        return state

    try:
        response = _examples_col.query(query_texts=[query], n_results=3)
        state["retrieved_examples"] = _parse_examples(response)
        logger.info("Retrieved %d example(s) for query", len(state["retrieved_examples"]))
    except Exception as e:
        logger.warning("Error querying examples collection: %s", e)
        state["retrieved_examples"] = []

    return state
