#!/usr/bin/env python3
"""
chroma_client.py

Helper module for managing Chromadb client, collections, and embedding functions.

Responsibilities:
  - Lazy-load a single Chromadb client instance
  - Provide a shared embedding function for all collections
  - Fetch or create named collections with the correct embedding function
"""

import chromadb
from functools import lru_cache
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.api import ClientAPI
from pathlib import Path
from typing import Any

from ..config import settings

@lru_cache(maxsize=1)
def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """
    Return a singleton SentenceTransformerEmbeddingFunction,
    configured with the model in settings.EMB_MODEL on GPU.

    Uses LRU cache to ensure only one instance is created.
    """
    return SentenceTransformerEmbeddingFunction(
        model_name=settings.EMB_MODEL,
        device="cuda"
    )

@lru_cache(maxsize=1)
def get_client() -> ClientAPI:
    """
    Return a singleton Chromadb PersistentClient.

    The client persists to disk at settings.CHROMA_PERSIST_PATH.
    """
    path = Path(settings.CHROMA_PERSIST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_PATH))


def get_collection(name: str) -> Any:
    """
    Get or create a Chromadb collection with the shared embedding function.

    Parameters:
      name: The collection name (e.g., settings.YAML_SCHEMA_COLLECTION)

    Returns:
      A Chromadb collection instance ready for .query, .add, .upsert, etc.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function()
    )