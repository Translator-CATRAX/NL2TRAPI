#!/usr/bin/env python3
"""
Build a lightning-fast exact-name index for NodeResolver.

Reads the canonical nodes dump (data/nodes.json) and writes a compact
pickle index mapping normalized names to CURIE, label, and category.

Usage:
    python scripts/build_exact_index.py \
        --input data/nodes.json \
        --output data/exact_index.pkl
"""

import argparse
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, Any, Optional

from tqdm import tqdm

from trapi_agent.config import settings
from trapi_agent.utils.biolink_utils import category_from_curie

# Configure module-level logger
d_logger = logging.getLogger(__name__)

def normalize(text: str) -> str:
    """
    Normalize a string for indexing:
    - lowercase
    - replace non-word characters with spaces
    - trim leading/trailing whitespace
    """
    return re.sub(r"\W+", " ", text.lower()).strip()


def build_index(source_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build an exact-name index from the nodes JSON dump.

    Returns:
        A dictionary mapping normalized names to entries:
          {
            "normalized_name": {
                "id": CURIE,
                "name": original_label,
                "category": biolink_category,
            },
            ...
          }
    """
    index: Dict[str, Dict[str, Any]] = {}
    skipped = 0

    # Load the entire JSON array (ensure sufficient memory)
    with source_path.open("r") as fh:
        nodes = json.load(fh)

    for node in tqdm(nodes, desc="Indexing nodes", unit="node"):
        curie: Optional[str] = node.get("id")
        name: str = node.get("name") or ""
        if not curie or not name.strip():
            skipped += 1
            continue

        entry = {
            "id": curie,
            "name": name,
            "category": category_from_curie(curie),
        }

        # Index the primary label
        key = normalize(name)
        index[key] = entry

        # Index all synonyms / aliases
        for alias in node.get("all_names", []):
            alias_key = normalize(str(alias))
            index.setdefault(alias_key, entry)

    if skipped:
        d_logger.warning("Skipped %d records without id or name", skipped)
    d_logger.info("Built index with %d total keys", len(index))
    return index


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build a pickle index mapping normalized names to CURIE entries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(settings.DATA_DIR) / "nodes.json",
        help="Path to the nodes JSON dump (canonical nodes.json)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.EXACT_INDEX_PKL,
        help="Path to write the generated pickle index."
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: builds the index and writes it to disk.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    d_logger.info("Starting exact-index build: %s → %s", args.input, args.output)

    if not args.input.exists():
        d_logger.error("Input file not found: %s", args.input)
        raise FileNotFoundError(f"Nodes dump not found: {args.input}")

    index = build_index(args.input)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        pickle.dump(index, fh)

    d_logger.info(
        "Wrote exact-index (%d keys) to %s",
        len(index),
        args.output,
    )


if __name__ == "__main__":
    main()
