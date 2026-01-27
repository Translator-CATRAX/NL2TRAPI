#!/usr/bin/env python3
"""
renci_pred_mapping.py

RENCI pred-mapping embedding fallback for Biolink predicate grounding.
Requires:
  - precomputed vectors JSON (all_biolink_mapped_vectors.json)
  - biolink-model.yaml for domain/range/inverse metadata
  - sentence-transformers + numpy
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .biolink_utils import allowed_object_categories, allowed_subject_categories, is_a

DEFAULT_VECTORS_PATH = Path(
    os.getenv("RENCI_PRED_VECTORS", "data/all_biolink_mapped_vectors.json")
)
DEFAULT_BIOLINK_YAML = Path(os.getenv("BIOLINK_YAML", "data/biolink-model.yaml"))
DEFAULT_EMB_MODEL = os.getenv(
    "RENCI_PRED_EMB_MODEL", "nomic-ai/nomic-embed-text-v1.5"
)

CAT_HINTS = {
    "biolink:Protein": {"protein", "gene product", "gene or gene product"},
    "biolink:Gene": {"gene", "gene or gene product"},
    "biolink:SmallMolecule": {"chemical entity", "molecular entity", "drug", "small molecule"},
    "biolink:ChemicalEntity": {"chemical entity", "molecular entity", "drug"},
    "biolink:Drug": {"drug", "chemical entity", "molecular entity"},
    "biolink:AnatomicalEntity": {"anatomical entity", "tissue", "organ"},
    "biolink:BiologicalProcess": {"biological process", "process"},
}


def _clean(text: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z0-9 :/_-]+", " ", (text or "").lower())
    ).strip()


def ensure_biolink(value: str) -> str:
    if not value:
        return value
    return value if value.startswith("biolink:") else f"biolink:{value}"


def _fits(domain_or_range: str, cat: Optional[str]) -> bool:
    if not domain_or_range or not cat:
        return True
    dom = _clean(domain_or_range)
    hints = CAT_HINTS.get(cat, set())
    if not hints:
        return True
    return any(h in dom for h in hints)


@lru_cache(maxsize=1)
def _load_biolink_slots() -> Dict[str, Dict[str, Any]]:
    path = DEFAULT_BIOLINK_YAML
    if not path.exists():
        raise FileNotFoundError(f"Biolink YAML not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    slots = (data.get("slots") or {}) if isinstance(data, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in slots.items():
        if isinstance(spec, dict):
            out[str(key)] = spec
    return out


def _slot_meta(slot_key: str) -> Dict[str, Any]:
    if not slot_key:
        return {}
    slots = _load_biolink_slots()
    base = slot_key.replace("biolink:", "").strip()
    variants = [
        base,
        base.replace("_", " "),
        base.replace(" ", "_"),
        slot_key,
        slot_key.replace("_", " "),
        slot_key.replace(" ", "_"),
    ]
    for key in variants:
        meta = slots.get(key)
        if isinstance(meta, dict):
            return meta
    return {}


@lru_cache(maxsize=1)
def _load_vectors() -> Tuple[List[str], "Any"]:
    if not DEFAULT_VECTORS_PATH.exists():
        raise FileNotFoundError(f"RENCI vectors not found: {DEFAULT_VECTORS_PATH}")
    import numpy as np

    raw = json.loads(DEFAULT_VECTORS_PATH.read_text())
    pred_names: List[str] = []
    pred_vecs: List[List[float]] = []

    if isinstance(raw, dict):
        for key, vec in raw.items():
            pred_names.append(str(key))
            pred_vecs.append(vec)
    elif isinstance(raw, list):
        for item in raw:
            name = item.get("mapped_predicate") or item.get("predicate") or item.get("pred")
            vec = item.get("vector") or item.get("embedding") or item.get("vec")
            if name is not None and vec is not None:
                pred_names.append(str(name))
                pred_vecs.append(vec)
    else:
        raise ValueError("Unknown predicate vector JSON format")

    mat = np.asarray(pred_vecs, dtype=np.float32)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return pred_names, mat


@lru_cache(maxsize=1)
def _model() -> "Any":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_EMB_MODEL, trust_remote_code=True)


def embed_query(text: str) -> "Any":
    import numpy as np

    q = "search_query: " + text
    vec = _model().encode([q], normalize_embeddings=True)[0].astype(np.float32)
    return vec


def renci_ground_predicate(
    *,
    subject_text: str,
    object_text: str,
    relationship_text: str,
    context_text: str = "",
    subject_category: Optional[str] = None,
    object_category: Optional[str] = None,
    top_k: int = 10,
    overfetch: int = 50,
    try_inverse: bool = True,
) -> Dict[str, Any]:
    pred_names, pred_mat = _load_vectors()

    query = (
        f"relationship: {_clean(relationship_text)} | "
        f"subject: {_clean(subject_text)} | "
        f"object: {_clean(object_text)} | "
        f"context: {_clean(context_text)}"
    )

    qv = embed_query(query)
    if pred_mat.shape[1] != qv.shape[0]:
        return {
            "predicate": "biolink:related_to",
            "score": None,
            "swap": False,
            "used_inverse": False,
            "chosen_based_on_candidate": None,
            "query_used": query,
            "candidates": [],
            "overfetched": 0,
            "method": "renci_embedding",
            "debug": {
                "error": "embedding_dim_mismatch",
                "vector_dim": int(pred_mat.shape[1]),
                "query_dim": int(qv.shape[0]),
            },
        }

    import numpy as np

    sims = pred_mat @ qv
    idxs = np.argsort(-sims)[: max(overfetch, top_k)]

    all_candidates: List[Dict[str, Any]] = []
    for i in idxs:
        raw_name = pred_names[int(i)]
        score = float(sims[int(i)])

        pred_curie = ensure_biolink(raw_name)
        pred_key = pred_curie.replace("biolink:", "")

        meta = _slot_meta(pred_key)
        domain = meta.get("domain")
        range_ = meta.get("range")
        inv = meta.get("inverse")
        inv_curie = ensure_biolink(inv) if inv else None

        all_candidates.append(
            {
                "predicate": pred_curie,
                "score": score,
                "domain": domain,
                "range": range_,
                "inverse": inv_curie,
            }
        )

    chosen = "biolink:related_to"
    chosen_score = None
    used_inverse = False
    swapped = False
    chosen_based_on_candidate = None

    def _allowed_ok(pred_curie: str, subj_cat: Optional[str], obj_cat: Optional[str]) -> Optional[bool]:
        allowed_sub = allowed_subject_categories(pred_curie)
        allowed_obj = allowed_object_categories(pred_curie)
        if not allowed_sub and not allowed_obj:
            return None
        if subj_cat and subj_cat != "biolink:NamedThing" and allowed_sub:
            if subj_cat not in allowed_sub and not any(is_a(subj_cat, a) for a in allowed_sub):
                return False
        if obj_cat and obj_cat != "biolink:NamedThing" and allowed_obj:
            if obj_cat not in allowed_obj and not any(is_a(obj_cat, a) for a in allowed_obj):
                return False
        return True

    for cand in all_candidates:
        pred_curie = cand["predicate"]
        pred_key = pred_curie.replace("biolink:", "")

        meta = _slot_meta(pred_key)
        domain = meta.get("domain")
        range_ = meta.get("range")
        inv = meta.get("inverse")
        inv_curie = ensure_biolink(inv) if inv else None

        allowed = _allowed_ok(pred_curie, subject_category, object_category)
        ok_forward = allowed if allowed is not None else (
            _fits(domain or "", subject_category) and _fits(range_ or "", object_category)
        )

        if ok_forward:
            chosen = pred_curie
            chosen_score = cand["score"]
            chosen_based_on_candidate = pred_curie
            break

        if try_inverse and inv_curie:
            inv_allowed = _allowed_ok(inv_curie, subject_category, object_category)
            ok_inverse = inv_allowed if inv_allowed is not None else (
                _fits(domain or "", object_category) and _fits(range_ or "", subject_category)
            )
            if ok_inverse:
                chosen = inv_curie
                chosen_score = cand["score"]
                used_inverse = True
                swapped = True
                chosen_based_on_candidate = pred_curie
                break

    return {
        "predicate": chosen,
        "score": chosen_score,
        "swap": swapped,
        "used_inverse": used_inverse,
        "chosen_based_on_candidate": chosen_based_on_candidate,
        "query_used": query,
        "candidates": all_candidates[:top_k],
        "overfetched": len(all_candidates),
        "method": "renci_embedding",
    }
