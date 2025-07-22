"""
Defines and compiles the LangGraph StateGraph for converting natural language queries
into TRAPI Query Graphs.

Workflow:
  1. ParseQuery       – extract entities & predicate
  2. ExampleRetrieval – fetch few-shot examples
  3. ResolveEntities  – map entity strings to CURIEs
  4. ResolveSchema    – map generic types & predicates to Biolink classes/predicates
  5. ConstructTRAPI   – assemble minimal TRAPI query_graph
  6. ValidateTRAPI    – check structure (nodes, edges, predicates)
  7. FixTRAPI         – attempt automated fixes on invalid graphs

Entry point: "ParseQuery". Terminal states: END (success) or END after max fix attempts.
"""
import logging
from typing import Type

from langgraph.graph import StateGraph, END

from .state_types import TRAPIState
from .nodes import (
    parse_query,
    retrieve_examples,
    resolve_entities,
    resolve_schema,
    construct_trapi,
    validate_trapi,
    fix_trapi,
)
from .config import settings

__all__ = ["graph"]

logger = logging.getLogger(__name__)


def build_agent_graph() -> StateGraph[TRAPIState]:
    """
    Build and return the LangGraph StateGraph for the NL→TRAPI pipeline.
    """
    builder: StateGraph[TRAPIState] = StateGraph(TRAPIState)

    # Register each processing node
    builder.add_node("ParseQuery", parse_query.node)
    builder.add_node("ExampleRetrieval", retrieve_examples.node)
    builder.add_node("ResolveEntities", resolve_entities.node)
    builder.add_node("ResolveSchema", resolve_schema.node)
    builder.add_node("ConstructTRAPI", construct_trapi.node)
    builder.add_node("ValidateTRAPI", validate_trapi.node)
    builder.add_node("FixTRAPI", fix_trapi.node)

    # Linear sequence
    builder.add_edge("ParseQuery", "ExampleRetrieval")
    builder.add_edge("ExampleRetrieval", "ResolveEntities")
    builder.add_edge("ResolveEntities", "ResolveSchema")
    builder.add_edge("ResolveSchema", "ConstructTRAPI")
    builder.add_edge("ConstructTRAPI", "ValidateTRAPI")

    # On validation: success or fix required
    builder.add_conditional_edges(
        "ValidateTRAPI",
        {
            END: lambda state: bool(state.get("valid", False)),
            "FixTRAPI": lambda state: not bool(state.get("valid", False)),
        },
    )

    # Retry loop for fixes up to MAX_FIX_ATTEMPTS
    builder.add_conditional_edges(
        "FixTRAPI",
        {
            "ValidateTRAPI": lambda state: state.get("fix_attempts", 0) < settings.MAX_FIX_ATTEMPTS,
            END: lambda state: state.get("fix_attempts", 0) >= settings.MAX_FIX_ATTEMPTS,
        },
    )

    # Define the graph entry point
    builder.set_entry_point("ParseQuery")

    graph = builder.compile()
    logger.info("Compiled agent graph with entry 'ParseQuery'")
    return graph

# Build graph on import for CLI use
graph: StateGraph[TRAPIState] = build_agent_graph()
