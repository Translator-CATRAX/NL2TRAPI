#!/usr/bin/env python3
"""
Build a CURIE → Biolink category mapping (pickle) from nodes.json.

Usage:
  python -m scripts.build_curie2cat \
    --nodes-file data/nodes.json \
    --output data/curie2cat.pkl
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import Any, Dict, Iterable

def coerce_category(raw: Any) -> str:
    """
    Normalize category to a single 'biolink:*' string.
    Accepts str or list[str]; if list, pick the first.
    """
    if raw is None:
        return "biolink:NamedThing"
    if isinstance(raw, str):
        return raw if raw.startswith("biolink:") else f"biolink:{raw.lstrip(':')}"
    if isinstance(raw, (list, tuple)) and raw:
        c = raw[0]
        if isinstance(c, str) and c.startswith("biolink:"):
            return c
        return f"biolink:{str(c).lstrip(':')}"
    return "biolink:NamedThing"

def iter_nodes(nodes_path: Path) -> Iterable[Dict[str, Any]]:
    with nodes_path.open("r") as fh:
        data = json.load(fh)
    for n in data:
        yield n

def main() -> None:
    ap = argparse.ArgumentParser(description="Build CURIE→category pickle.")
    ap.add_argument("--nodes-file", type=Path, default=Path("data/nodes.json"))
    ap.add_argument("--output",     type=Path, default=Path("data/curie2cat.pkl"))
    args = ap.parse_args()

    curie2cat: Dict[str, str] = {}
    count = 0
    for n in iter_nodes(args.nodes_file):
        curie = n.get("id")
        if not curie:
            continue
        curie2cat[curie] = coerce_category(n.get("category"))
        count += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        pickle.dump(curie2cat, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Wrote {count} CURIE→category entries to {args.output}")

if __name__ == "__main__":
    main()
