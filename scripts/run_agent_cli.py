#!/usr/bin/env python3
"""
scripts/run_agent_cli.py
────────────────────────
Command-line helper for NL→TRAPI.

Examples
--------
$ PYTHONPATH=. python scripts/run_agent_cli.py \
    "What proteins does acetaminophen interact with?"

# or read from stdin
$ echo "What drugs treat asthma?" | PYTHONPATH=. python scripts/run_agent_cli.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from trapi_agent.service import run_agent   # single entry-point → keeps models warm (loaded once per process)



# Argument parsing                                                            

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invoke the NL→TRAPI agent on a natural-language query"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Natural-language question (omit to read from stdin)",
    )
    return parser.parse_args()



# Main                                                                        

def main() -> None:
    args = _parse_args()

    # Resolve input text
    query_text: str = (
        " ".join(args.query) if args.query else sys.stdin.read().strip()
    )
    if not query_text:
        sys.stderr.write("  No query provided (args or stdin required).\n")
        sys.exit(2)

    # Minimal logging (inherits project-wide config if present)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info("Invoking agent for query: %s", query_text)

    # Call the shared helper (loads graph + caches models only once per process)
    result = run_agent(query_text)

    # Pretty-print JSON to stdout
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
