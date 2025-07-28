from __future__ import annotations  # Required for generic type annotations in Python < 3.11

"""
Defines and compiles the LangGraph StateGraph for converting natural language queries
into TRAPI Query Graphs.

Workflow:
  1. ParseQuery       – Extracts entities & predicate
  2. ExampleRetrieval – Retrieves few-shot NL→TRAPI examples
  3. ResolveEntities  – Resolves free-text entities to CURIEs + categories
  4. ResolveSchema    – Resolves generic types and predicates to Biolink classes/predicates
  5. ConstructTRAPI   – Constructs the query_graph portion of the TRAPI message
  6. ValidateTRAPI    – Checks for valid structure (nodes, edges, predicates)
  7. FixTRAPI         – Attempts repair if the TRAPI is invalid

Entry point:    "ParseQuery"
Success path:   END (if TRAPI is valid)
Failure path:   Retry via FixTRAPI up to MAX_FIX_ATTEMPTS → END
"""

import logging
from langgraph.graph import StateGraph, END

from .state_types import TRAPIState
from .config import settings
from .nodes import (
    parse_query,
    retrieve_examples,
    resolve_entities,
    resolve_schema,
    construct_trapi,
    validate_trapi,
    fix_trapi,
)

__all__ = ["graph"]

logger = logging.getLogger(__name__)


def build_agent_graph() -> StateGraph:
    """
    Assemble the LangGraph NL→TRAPI state machine pipeline.

    Returns:
        StateGraph instance ready for inference.
    """
    builder = StateGraph(TRAPIState)

    # ── Stepwise pipeline ─────────────────────────────────────────────
    builder.add_node("ParseQuery",        parse_query.node)
    builder.add_node("ExampleRetrieval",  retrieve_examples.node)
    builder.add_node("ResolveEntities",   resolve_entities.node)
    builder.add_node("ResolveSchema",     resolve_schema.node)
    builder.add_node("ConstructTRAPI",    construct_trapi.node)
    builder.add_node("ValidateTRAPI",     validate_trapi.node)
    builder.add_node("FixTRAPI",          fix_trapi.node)

    # ── Linear edges ─────────────────────────────────────────────────
    builder.add_edge("ParseQuery",        "ExampleRetrieval")
    builder.add_edge("ExampleRetrieval",  "ResolveEntities")
    builder.add_edge("ResolveEntities",   "ResolveSchema")
    builder.add_edge("ResolveSchema",     "ConstructTRAPI")
    builder.add_edge("ConstructTRAPI",    "ValidateTRAPI")

    # ── Conditional edges: Validation success vs Fix loop ────────────
    builder.add_conditional_edges(
        "ValidateTRAPI",
        {
            END:        lambda s: s.get("valid", False),
            "FixTRAPI": lambda s: not s.get("valid", False),
        },
    )

    builder.add_conditional_edges(
        "FixTRAPI",
        {
            "ValidateTRAPI": lambda s: s.get("fix_attempts", 0) < settings.MAX_FIX_ATTEMPTS,
            END:             lambda s: s.get("fix_attempts", 0) >= settings.MAX_FIX_ATTEMPTS,
        },
    )

    builder.set_entry_point("ParseQuery")
    graph = builder.compile()

    logger.info("✅ LangGraph NL→TRAPI pipeline compiled.")
    return graph


# Instantiate global graph for CLI use
graph: StateGraph = build_agent_graph()
