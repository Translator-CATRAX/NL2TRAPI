#!/usr/bin/env python3
"""
Embed nodes into a Chroma collection for semantic lookup.

Reads a large JSON dump of node objects (CURIE, name, category) and
upserts them in batches into the Chroma collection defined by
`settings.NODES_COLLECTION`.

Usage:
    python -m scripts.load_nodes_info \
        --nodes-file data/nodes.json \
        --chroma-path data/chroma_indexes \
        --collection-name nodes_info \
        [--batch-size 5000]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
import chromadb

from trapi_agent.utils.chroma_client import get_client, get_embedding_function
from trapi_agent.config import settings

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for node embedding.

    Returns:
        argparse.Namespace with:
          - nodes_file: Path to nodes.json
          - chroma_path: Path for Chroma persistence
          - collection_name: Chroma collection name
          - batch_size: Number of nodes per upsert batch
    """
    parser = argparse.ArgumentParser(description="Embed nodes into Chroma.")
    parser.add_argument(
        "--nodes-file",
        type=Path,
        default=Path(settings.DATA_DIR) / "nodes.json",
        help="Path to the canonical nodes JSON dump."
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
        default=settings.NODES_COLLECTION,
        help="Name of the Chroma collection for node embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of nodes to upsert per batch."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point: loads nodes, batches, and upserts into Chroma.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    nodes_file: Path = args.nodes_file
    if not nodes_file.exists():
        logger.error("Nodes file not found: %s", nodes_file)
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")

    logger.info("Loading nodes from %s", nodes_file)
    with nodes_file.open("r") as fh:
        nodes = json.load(fh)

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_or_create_collection(
        name=args.collection_name,
        embedding_function=get_embedding_function(),
    )

    logger.info(
        "Embedding %d nodes into collection '%s'", len(nodes), args.collection_name
    )

    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    total = 0

    for node in tqdm(nodes, desc="Embedding nodes", unit="node"):
        curie = node.get("id")
        name = (node.get("name") or "").strip()
        category = (node.get("category") or "").strip()
        if not curie or not name:
            continue

        batch_ids.append(curie)
        batch_docs.append(name)
        batch_metas.append({"id": curie, "category": category})
        total += 1

        # upsert in batches
        if total % args.batch_size == 0:
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            logger.info("Upserted %d nodes", total)
            batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    # flush remaining
    if batch_ids:
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )
    logger.info(" Finished embedding %d nodes into '%s'", total, args.collection_name)


if __name__ == "__main__":
    main()
