#!/usr/bin/env python3
"""
node_norm.py

Thin, cached wrapper around the SRI Node-Normalization REST API.

Exports:
  lookup(name: str, mode: str = "lookup", timeout: float = 5.0)
    ➔ Tuple[Optional[str], str, str]

Behavior:
  • mode="lookup":   GET /lookup → best CURIE, label, semanticType
  • mode="normalized": GET /lookup ➔ POST /get_normalized_nodes for canonical CURIE
  • Results are LRU-cached (maxsize=4096) to avoid redundant HTTP calls.

Handles network errors gracefully, falling back to best-available results.
"""

from __future__ import annotations
import functools
import logging
from typing import Tuple, Optional

import requests

# API endpoints
_LOOKUP_URL: str = "https://name-lookup.transltr.io/lookup"
_NORMALIZE_URL: str = "https://nodenormalization-sri.renci.org/get_normalized_nodes"

# Configure module-level logger
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4096)
def lookup(
    name: str,
    mode: str = "lookup",
    timeout: float = 5.0,
) -> Tuple[Optional[str], str, str]:
    """
    Query SRI Node-Normalization service for a best-matching CURIE.

    Args:
        name: Free-text entity (e.g. "aspirin", "TP53").
        mode: "lookup" for fast single-stage lookup,
              "normalized" for two-stage canonicalization.
        timeout: HTTP timeout in seconds.

    Returns:
        (curie or None, label, semanticType)
    """
    try:
        # Stage 1: lookup
        resp = requests.get(
            _LOOKUP_URL,
            params={"string": name, "autocomplete": "false"},
            timeout=timeout,
        )
        resp.raise_for_status()
        hits = resp.json()

        if not hits:
            return None, "—", "—"

        hit = hits[0]
        curie = hit.get("curie")
        label = hit.get("label", "—")
        sem_type = hit.get("semanticType", "—")

    except Exception as e:
        logger.warning("NodeNorm lookup failed for '%s': %s", name, e)
        return None, "—", "—"

    # Stage 2: optional normalization
    if mode == "normalized" and curie:
        try:
            resp2 = requests.post(
                _NORMALIZE_URL,
                json={"curies": [curie]},
                timeout=timeout,
            )
            resp2.raise_for_status()
            data = resp2.json().get(curie, {})
            if data:
                id_val = data.get("id")
                # id can be dict or list
                if isinstance(id_val, dict):
                    curie = id_val.get("identifier", curie)
                elif isinstance(id_val, list) and id_val:
                    curie = id_val[0]
                label = data.get("label", label)
                sem_type = data.get("type", sem_type)
        except Exception as e:
            logger.debug("Normalization step failed for '%s': %s", curie, e)

    return curie, label, sem_type
