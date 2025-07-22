#!/usr/bin/env python3
"""
Embed Biolink-Model YAML into a local Chroma collection for fast semantic
lookup of *classes* (node categories) and *slots* (predicates).

This script reads a Biolink YAML specification and upserts each class and
predicate into a Chroma collection for use in schema resolution.

Usage:
    python -m scripts.embed_biolink_yaml \
        --yaml-file path/to/biolink-model.yaml \
        [--chroma-path data/chroma_indexes] \
        [--collection-name yaml_schema] \
        [--no-increment]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
import chromadb

from trapi_agent.utils.chroma_client import get_embedding_function
from trapi_agent.config import settings

# Configure module-level logger
logger = logging.getLogger(__name__)


def flatten_metadata(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert metadata values into string form for Chroma ingestion.

    Args:
        meta: A dict of raw metadata values (could be None, list, dict, etc.).

    Returns:
        A dict of stringified metadata values.
    """
    flat: Dict[str, str] = {}
    for key, val in meta.items():
        if val is None:
            flat[key] = ""
        elif isinstance(val, (list, tuple, set)):
            flat[key] = ", ".join(map(str, val))
        elif isinstance(val, dict):
            flat[key] = json.dumps(val)
        else:
            flat[key] = str(val)
    return flat


def iter_yaml_section(
    entries: Dict[str, Dict[str, Any]],
    section: str,
    category: str,
) -> Iterable[Tuple[str, Dict[str, str], str]]:
    """
    Iterate over a Biolink YAML section ('classes' or 'slots') and yield
    tuples of (key, metadata, document_text) for embedding.

    Args:
        entries: The sub-dictionary from the YAML (classes or slots).
        section: 'classes' or 'slots'.
        category: Biolink category label ('class' or 'predicate').

    Yields:
        Tuples of (id, flattened_metadata, document_text).
    """
    for key, spec in entries.items():
        description = (spec.get("description") or "").strip()
        if section == "classes":
            raw_meta = {
                "category": category,
                "key": key,
                "description": description,
                "aliases": spec.get("aliases", []),
                "is_a": spec.get("is_a", ""),
                "slots": spec.get("slots", []),
            }
        else:  # predicates
            raw_meta = {
                "category": category,
                "key": key,
                "description": description,
                "domain": spec.get("domain", ""),
            }
        meta = flatten_metadata(raw_meta)
        document = f"{key} | {category} — {description}"
        yield key, meta, document


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        An argparse Namespace with yaml_file, chroma_path, collection_name,
        and no_increment flag.
    """
    parser = argparse.ArgumentParser(
        description="Embed Biolink YAML into Chroma for semantic lookup."
    )
    parser.add_argument(
        "--yaml-file",
        type=Path,
        default=Path("biolink-model.yaml"),
        help="Path to the Biolink YAML file."
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=settings.CHROMA_PERSIST_PATH,
        help="Path to the Chroma persistent store."
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=settings.YAML_SCHEMA_COLLECTION,
        help="Name of the Chroma collection to upsert into."
    )
    parser.add_argument(
        "--no-increment",
        action="store_true",
        help="Disable index increment optimization for faster re-runs."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point: loads the YAML, processes entries, and upserts into Chroma.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    yaml_path: Path = args.yaml_file
    if not yaml_path.exists():
        logger.error("Biolink YAML not found at: %s", yaml_path)
        raise FileNotFoundError(f"Biolink YAML not found: {yaml_path}")

    logger.info("Loading Biolink YAML from %s", yaml_path)
    data = yaml.safe_load(yaml_path.read_text())

    # Collect classes and predicates
    rows: List[Tuple[str, Dict[str, str], str]] = []
    rows.extend(iter_yaml_section(data.get("classes", {}), "classes", "class"))
    rows.extend(iter_yaml_section(data.get("slots", {}), "slots", "predicate"))

    if not rows:
        logger.error("No classes or predicates found in YAML.")
        raise ValueError("Empty Biolink YAML sections.")

    ids, metas, docs = zip(*rows)

    # Initialize Chroma client and collection
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=get_embedding_function(),
    )

    # Upsert embeddings
    logger.info(
        "Upserting %d items into collection '%s'", len(rows), args.collection_name
    )
    collection.upsert(
        ids=list(ids),
        documents=list(docs),
        metadatas=list(metas),
        incremental_index=not args.no_increment,
    )

    logger.info(
        " Embedded %d Biolink schema items into '%s' (store: %s)",
        len(rows), args.collection_name, args.chroma_path,
    )


if __name__ == "__main__":
    main()
