#!/usr/bin/env python3
"""
node_resolver.py
---------------

Fallback resolver for free-text entity names:
1. SRI Node-Normalization  (fast, authoritative)
2. Exact match via local pickle index
3. Fuzzy match via rapidfuzz (score ≥ cutoff)
4. (Optional) Semantic match via Chroma embeddings

Provides:
  NodeResolver.resolve(text) → Optional[Dict[str, Any]]
  NodeResolver.node_norm(text) → Optional[Dict[str, Any]]

Usage:
    resolver = NodeResolver()
    result = resolver.resolve("aspirin")
"""
import re
import pickle
import logging
import requests
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional
from rapidfuzz import process
from .chroma_client import get_collection
from ..config import settings

logger = logging.getLogger(__name__)


class NodeResolver:
    """
    Attempts to resolve free-text entity names to CURIEs and metadata.

    Resolution order:
      1. SRI Node-Normalization (fast, authoritative)
      2. Exact lookup in a local pickle index
      3. Fuzzy match via rapidfuzz
      4. Semantic lookup via Chroma embeddings (optional)

    Attributes:
        fuzzy_cutoff (int): Minimum score for fuzzy matching (0-100).
        top_k        (int): Number of neighbours for semantic matching.
    """

    _NODE_NORM_URL = "https://name-lookup.transltr.io/lookup"

    def __init__(
        self,
        fuzzy_cutoff: int = 85,
        top_k: int = 3
    ) -> None:
        # Validate index file
        if not settings.EXACT_INDEX_PKL.exists():
            raise FileNotFoundError(
                f"Exact index not found: {settings.EXACT_INDEX_PKL}"
            )
        self.fuzzy_cutoff = fuzzy_cutoff
        self.top_k = top_k
        self._load_exact_index()
        # Semantic matching can be enabled by uncommenting:
        # self._chroma = get_collection(settings.NODES_COLLECTION)

    def _load_exact_index(self) -> None:
        """
        Load the normalized-name index from the pickle file.
        Format: {norm_name: {id, name, category}}
        """
        with open(settings.EXACT_INDEX_PKL, "rb") as fh:
            self.exact_index: Dict[str, Dict[str, Any]] = pickle.load(fh)
        self.name_keys = list(self.exact_index.keys())

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text: lowercase, remove non-word chars, collapse spaces."""
        return re.sub(r"\W+", " ", text or "").lower().strip()

    def exact_match(self, text: str) -> Optional[Dict[str, Any]]:
        """Return entry if normalized text exactly matches an index key."""
        key = self._normalize_text(text)
        return self.exact_index.get(key)

    def fuzzy_match(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Return best fuzzy match if its score ≥ fuzzy_cutoff.
        Uses rapidfuzz.process.extractOne under the hood.
        """
        key = self._normalize_text(text)
        match = process.extractOne(
            key,
            self.name_keys,
            score_cutoff=self.fuzzy_cutoff
        )
        if match:
            return self.exact_index.get(match[0])
        return None

    def semantic_match(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Semantic lookup via Chroma embeddings (requires embedding index).
        Currently disabled by default.
        """
        # res = self._chroma.query(text, n_results=self.top_k)
        # if res and res["metadatas"][0]:
        #     return res["metadatas"][0][0]
        return None

    @lru_cache(maxsize=4096)
    def node_norm(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Call SRI Node-Normalization service to canonicalize a name.
        Returns the first hit containing: curie, label, semanticType.
        """
        try:
            resp = requests.get(
                self._NODE_NORM_URL,
                params={"string": text, "autocomplete": "false"},
                timeout=5,
            )
            resp.raise_for_status()
            hits = resp.json()
            if not hits:
                return None
            hit = hits[0]
            return {
                "id":       hit["curie"],
                "name":     hit.get("label", text),
                "category": f"biolink:{hit.get('semanticType','NamedThing')}",
                "method":   "node_norm",
            }
        except Exception as e:
            logger.debug(
                "NodeNorm lookup failed for '%s': %s", text, e
            )
            return None

    def resolve(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Resolve `text` to a node dict via:
          node_norm → exact_match → fuzzy_match → semantic_match
        """
        return (
            self.node_norm(text)
            or self.exact_match(text)
            or self.fuzzy_match(text)
            or self.semantic_match(text)
        )
