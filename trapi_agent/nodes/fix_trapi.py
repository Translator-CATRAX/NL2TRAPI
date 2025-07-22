#!/usr/bin/env python3
"""
fix_trapi.py

LangGraph node to repair malformed TRAPI query_graph JSON using an LLM.

This node:
  • Tracks the number of fix attempts in state['fix_attempts'].
  • Builds a prompt with existing errors and broken JSON.
  • Invokes a greedy LLM call to produce corrected query_graph.
  • Normalizes edge predicates, ensuring a 'predicates' list per TRAPI 1.4.

Inputs (state):
  - state['errors']: List[str] of validation errors
  - state['output_json']: Dict containing the broken 'query_graph'
  - state['predicate']: fallback Biolink predicate CURIE

Outputs (state):
  - Updated state['output_json']['message']['query_graph'] if fix succeeds
  - state['fix_attempts'] incremented
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict

from ..state_types import TRAPIState
from ..prompts import FIX_TEMPLATE
from ..utils.llm import run_llm

logger = logging.getLogger(__name__)


def node(state: TRAPIState) -> TRAPIState:
    """
    Attempt to correct a TRAPI query_graph via an LLM 'fix' prompt.

    - Increments state['fix_attempts'].
    - Uses FIX_TEMPLATE to ask the LLM to output corrected JSON.
    - Parses the first JSON object in the LLM response.
    - Ensures each edge has a 'predicates' list (fallback if missing).

    Args:
        state: Mutable TRAPIState with 'output_json', 'errors', and 'predicate'.

    Returns:
        The updated state with repaired 'query_graph' or original if parsing fails.
    """
    # Increment fix attempt counter
    attempts = state.get("fix_attempts", 0) + 1
    state["fix_attempts"] = attempts
    logger.debug("Fix attempt #%d", attempts)

    # Build and send the LLM prompt
    prompt = FIX_TEMPLATE.format(
        errors="\n".join(state.get("errors", [])),
        json=json.dumps(state.get("output_json", {}), indent=2),
    )
    raw_output = run_llm(prompt, max_new_tokens=400, temperature=0.0)

    try:
        # Extract the first JSON object from the model output
        json_start = raw_output.index("{")
        json_end = raw_output.rindex("}") + 1
        fixed_qg = json.loads(raw_output[json_start:json_end])

        # Normalize predicates in each edge
        fallback_pred = state.get("predicate", "biolink:related_to")
        for edge_id, edge in fixed_qg.get("edges", {}).items():
            # Convert singular 'predicate' to list if present
            if "predicate" in edge and "predicates" not in edge:
                edge["predicates"] = [edge.pop("predicate")]
            # Ensure 'predicates' list exists and is non-empty
            if not edge.get("predicates"):
                edge["predicates"] = [fallback_pred]

        # Write back the corrected query_graph
        state["output_json"]["message"]["query_graph"] = fixed_qg
        logger.info("Successfully applied fix_trapi on attempt %d", attempts)
    except Exception as err:
        logger.warning("Failed to parse/fix query_graph on attempt %d: %s", attempts, err)

    return state
