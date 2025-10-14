

#!/usr/bin/env python3
"""
Command-line interface for invoking the TRAPI agent graph on natural-language queries.

Usage:
    python -m scripts.run_agent_cli "What drugs treat asthma?"
    python -m scripts.run_agent_cli "Find paths between ibuprofen and COX1" --route pathfinder
    python -m scripts.run_agent_cli "what genes are upregulated by filgrastim?" --route xcrg
    echo "What proteins interact with aspirin?" | python -m scripts.run_agent_cli
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict, Any

from trapi_agent.agent_graph import graph

# Add xcrg to the allowed routes
ROUTE_CHOICES = ["onehop", "pathfinder", "pathfinder_constrained", "treats", "chem_gene", "multihop", "xcrg"]



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the agent CLI."""
    parser = argparse.ArgumentParser(description="Run the TRAPI agent on a natural-language query")
    parser.add_argument("query", nargs="*", help="The natural-language query. If omitted, read from stdin.")
    parser.add_argument(
        "--route",
        choices=ROUTE_CHOICES,
        help="Force a specific route (overrides auto-router)."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: invoke the graph and print TRAPI query_graph or errors."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    args = parse_args()
    query_text = " ".join(args.query) if args.query else sys.stdin.read().strip()
    if not query_text:
        logger.error("No query provided. Use positional args or pipe input text.")
        sys.exit(2)

    logger.info("Invoking agent for query: %s", query_text)

    # Initial state (allow forced route for demos)
    state: Dict[str, Any] = {"query": query_text}
    if args.route:
        state["route"] = args.route

    # Run pipeline
    final_state: Dict[str, Any] = graph.invoke(state)

    # Output
    if final_state.get("valid"):
        print(json.dumps(final_state.get("output_json", {}), indent=2))
        sys.exit(0)
    else:
        errors = final_state.get("errors", ["Unknown failure"])
        logger.error(" Query invalid after fixes: %s", errors)
        if "output_json" in final_state:
            print(json.dumps(final_state["output_json"], indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
