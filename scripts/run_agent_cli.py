#!/usr/bin/env python3
"""
Command-line interface for invoking the TRAPI agent graph on natural-language queries.

This script accepts a question (either as positional arguments or via stdin),
passes it through the agent pipeline (`graph.invoke`), and prints the
resulting TRAPI query_graph JSON or error messages.

Usage:
    python -m scripts.run_agent_cli "What drugs treat asthma?"
    echo "What proteins interact with aspirin?" | python -m scripts.run_agent_cli
"""

import argparse
import json
import logging
import sys
from typing import Dict, Any

from trapi_agent.agent_graph import graph


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the agent CLI.

    Returns:
        argparse.Namespace with attribute `query` (list of tokens).
    """
    parser = argparse.ArgumentParser(
        description="Run the TRAPI agent on a natural-language query"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="The natural-language query. If omitted, read from stdin."
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: constructs the query string, invokes the graph,
    and prints the JSON output or errors.
    """
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Determine query text
    if args.query:
        query_text = " ".join(args.query)
    else:
        query_text = sys.stdin.read().strip()

    if not query_text:
        logger.error("No query provided. Use positional args or pipe input text.")
        sys.exit(2)

    logger.info("Invoking agent for query: %s", query_text)

    # Run the agent pipeline
    final_state: Dict[str, Any] = graph.invoke({"query": query_text})

    # Output
    if final_state.get("valid"):
        print(json.dumps(final_state["output_json"], indent=2))
        sys.exit(0)
    else:
        errors = final_state.get("errors", ["Unknown failure"])
        logger.error(" Query invalid after fixes: %s", errors)
        sys.exit(1)


if __name__ == "__main__":
    main()
