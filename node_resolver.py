# node_resolver.py — Final Optimized Version with Logging
import re
import os
import pickle
import logging
from typing import List, Dict, Optional
from rapidfuzz import process

logger = logging.getLogger("NodeResolver")
logger.setLevel(logging.INFO)

class NodeResolver:
    def __init__(self, chroma_client, index_path="exact_index.pkl", fuzzy_cutoff=85, semantic_top_k=3):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Exact index file not found: {index_path}")

        self.collection = chroma_client.get_collection(name="nodes_info")
        with open(index_path, "rb") as f:
            self.exact_index = pickle.load(f)

        self.name_keys = list(self.exact_index.keys())
        self.fuzzy_cutoff = fuzzy_cutoff
        self.semantic_top_k = semantic_top_k

    def normalize(self, text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    def exact_match(self, text: str) -> Optional[Dict]:
        key = self.normalize(text)
        result = self.exact_index.get(key)
        if result:
            logger.info(f"Exact match → {result['name']} | {result['id']} | {result['category']}")
        return result

    def semantic_fallback(self, text: str) -> List[Dict]:
        results = self.collection.query(query_texts=[text], n_results=self.semantic_top_k)
        matches = [meta for meta in results["metadatas"][0] if meta.get("category")] if results else []
        if matches:
            logger.info(f"Semantic match → {[m['name'] for m in matches[:1]]}")
        return matches

    def fuzzy_match(self, text: str) -> Optional[Dict]:
        matches = process.extract(
            self.normalize(text), self.name_keys, limit=1, score_cutoff=self.fuzzy_cutoff
        )
        if matches:
            match_name, score, _ = matches[0]
            logger.info(f"Fuzzy match → {match_name} (score={score})")
            return self.exact_index.get(match_name)
        return None

    def resolve(self, text: str) -> List[Dict]:
        if (exact := self.exact_match(text)):
            return [exact]
        if (semantic := self.semantic_fallback(text)):
            return semantic
        if (fuzzy := self.fuzzy_match(text)):
            return [fuzzy]
        logger.warning(f"No match found for: {text}")
        return []
