#!/usr/bin/env python3
"""
biolink_utils.py

Utility functions for normalizing Biolink categories and predicates.

Module Responsibilities:
  - Map CURIE prefixes to Biolink classes when upstream inference is unavailable
  - Normalize free-text predicates to canonical Biolink predicate identifiers

Loading the full Biolink spec is supported but optional for extensions.
"""

import re
import yaml
from pathlib import Path
from typing import Any, Dict

from ..config import settings

# ─── Biolink YAML Load (optional) ───────────────────────────────
# Path assumed at project root; adjust if needed
_BIOLINK_YAML_PATH: Path = Path(__file__).parent.parent / "biolink-model.yaml"
try:
    _YAML: Dict[str, Any] = yaml.safe_load(_BIOLINK_YAML_PATH.read_text())
except Exception:
    _YAML = {}

# ─── Prefix → Biolink Category Fallback Mapping ────────────────
PREFIX2CAT: Dict[str, str] = {
    "CHEBI":    "biolink:ChemicalEntity",
    "DRUGBANK": "biolink:ChemicalEntity",
    "NCBIGene": "biolink:Gene",
    "MONDO":    "biolink:Disease",
    "HGNC":     "biolink:Gene",
    # Extend with new prefixes as encountered
}

# ─── Predicate Synonym Map ─────────────────────────────────────
# Maps common verb phrases to canonical Biolink predicates
PRED_SYNONYM: Dict[str, str] = {
    "interacts with":           "biolink:physically_interacts_with",
    "interacts":                "biolink:physically_interacts_with",
    "targets":                  "biolink:physically_interacts_with",
    "binds":                    "biolink:physically_interacts_with",
    "related":                  "biolink:related_to",
    "treats":                   "biolink:treats",
    "contraindicated":          "biolink:contraindicated_for",
}


def category_from_curie(curie: str) -> str:
    """
    Infer a Biolink class CURIE from a namespace prefix.

    Parameters:
        curie: A string of form 'PREFIX:Identifier'.

    Returns:
        The corresponding Biolink class, or 'biolink:NamedThing' if unknown.
    """
    prefix = curie.split(":", 1)[0]
    return PREFIX2CAT.get(prefix, "biolink:NamedThing")


def normalize_pred(text: str) -> str:
    """
    Normalize free-text predicate to a canonical, underscored form.

    Steps:
      1. Lowercase, remove non-word characters
      2. Map via PRED_SYNONYM if available
      3. Replace spaces with underscores

    Parameters:
        text: Raw predicate phrase (e.g., 'interacts with')

    Returns:
        A normalized key, e.g. 'biolink:physically_interacts_with' or
        'related_to' for unmapped inputs.
    """
    # clean text
    cleaned = re.sub(r"\W+", " ", text.lower()).strip()
    # apply synonym map
    mapped = PRED_SYNONYM.get(cleaned, cleaned)
    # ensure underscores and strip any leading 'biolink:'
    underscored = mapped.replace(" ", "_")
    if underscored.startswith("biolink:"):
        return underscored
    return underscored
