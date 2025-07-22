#!/usr/bin/env python3
"""
Embed natural-language to TRAPI few-shot examples into a Chroma collection.

Reads an examples JSON file containing pairs of {
  "nl_query": str,
  "trapi_query": dict
}
and upserts them (idempotently) into the Chroma collection defined by
`settings.EXAMPLE_COLLECTION`.

Usage:
    python -m scripts.embed_nl2trapi_examples \
        --examples-file path/to/examples.json \
        [--chroma-path data/chroma_indexes] \
        [--collection-name nl_to_trapi]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from trapi_agent.utils.chroma_client import get_embedding_function
from trapi_agent.config import settings

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace with:
          examples_file: Path to the examples.json file
          chroma_path: Path for the Chroma persistent store
          collection_name: Name of the Chroma collection
    """
    parser = argparse.ArgumentParser(
        description="Embed NL→TRAPI few-shot examples into Chroma."
    )
    parser.add_argument(
        "--examples-file",
        type=Path,
        default=Path("examples.json"),
        help="Path to the JSON file with few-shot examples."
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=settings.CHROMA_PERSIST_PATH,
        help="Filesystem path for Chroma persistence."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=settings.EXAMPLE_COLLECTION,
        help="Name of the Chroma collection for examples."
    )
    return parser.parse_args()


def load_examples(examples_file: Path) -> List[Dict[str, Any]]:
    """
    Load and return the list of examples from the JSON file.

    Args:
        examples_file: Path to the JSON file.

    Returns:
        A list of dicts with keys "nl_query" and "trapi_query".

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not examples_file.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_file}")
    return json.loads(examples_file.read_text())


def main() -> None:
    """
    Main entry point:
      - Parses arguments
      - Loads examples
      - Upserts them into Chroma
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info(
        "Loading %s for embedding into '%s' (path: %s)",
        args.examples_file,
        args.collection_name,
        args.chroma_path,
    )

    examples = load_examples(args.examples_file)

    # Initialize Chroma client and collection
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=get_embedding_function(),
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        ids.append(f"ex-{idx}")
        documents.append(ex["nl_query"])
        metadatas.append({"trapi_query": json.dumps(ex["trapi_query"])})

    # Upsert is idempotent: updates existing IDs, adds new ones
    logger.info("Upserting %d examples into collection '%s'", len(ids), args.collection_name)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        "Embedded %d NL→TRAPI examples into Chroma collection '%s'",
        len(ids), args.collection_name,
    )


if __name__ == "__main__":
    main()
