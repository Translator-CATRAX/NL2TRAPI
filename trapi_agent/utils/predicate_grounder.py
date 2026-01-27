#!/usr/bin/env python3
"""
predicate_grounder.py

Deterministic one-hop predicate grounding (no LLMs).

Strategy:
  1) Exact/alias mapping via biolink_utils.normalize_pred(text)
  2) Enforce Biolink domain/range compatibility (+ inverse handling)
  3) Fallback to candidate predicates (e.g., embedding retrieval output)
  4) Default to biolink:related_to
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml

from .biolink_utils import allowed_object_categories, allowed_subject_categories, is_a, normalize_pred

_STOPWORDS = {
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "a", "an", "the",
    "of", "for", "to", "in", "on", "at", "by", "with", "from", "into", "about",
    "and", "or",
}

_ONEHOP_PRED_ALLOWLIST: Set[str] = {
    "biolink:physically_interacts_with",
    "biolink:interacts_with",
    "biolink:expressed_in",
    "biolink:expresses",
    "biolink:located_in",
    "biolink:participates_in",
    "biolink:in_pathway_with",
    "biolink:affects",
    "biolink:regulates",
    "biolink:associated_with",
    "biolink:related_to",
    "biolink:treats",
    "biolink:contraindicated_for",
    "biolink:causes",
}

_NEGATION_PAT = re.compile(r"\b(no|not|without|lacking|lack|lacks|absence|absent|neither|nor)\b", re.I)
_CAUSAL_PAT = re.compile(r"\b(caus\w*|lead(?:s)? to|result(?:s)? in|induc\w*|trigger\w*|drive\w*|contribut\w*)\b", re.I)


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 _:-]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_stopwords(text: str) -> str:
    toks = [t for t in _clean_text(text).split() if t and t not in _STOPWORDS]
    return " ".join(toks).strip()


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_PAT.search(text or ""))


def _has_causal_cue(text: str) -> bool:
    return bool(_CAUSAL_PAT.search(text or ""))


def _is_neg_predicate(pred_curie: str) -> bool:
    key = _slot_key(pred_curie).lower()
    return key.endswith("_neg") or key.endswith("_negative") or "_neg_" in key or key.startswith("neg_")


def _allow_predicate(pred_curie: str, text: str) -> bool:
    if pred_curie not in _ONEHOP_PRED_ALLOWLIST:
        return False
    if _is_neg_predicate(pred_curie) and not _has_negation(text):
        return False
    if pred_curie == "biolink:causes" and not _has_causal_cue(text):
        return False
    return True

def _lexical_predicate(
    relationship_text: str,
    context_text: str,
    subject_category: Optional[str],
    object_category: Optional[str],
) -> Optional[str]:
    text = " ".join(t for t in (relationship_text, context_text) if t).lower()
    if not text:
        return None

    if re.search(r"\b(interact|bind|target|partner)\w*\b", text):
        return "biolink:physically_interacts_with"
    if re.search(r"\bexpress\w*\b", text):
        return "biolink:expressed_in"
    if re.search(r"\b(locat|localiz|found|present)\w*\b", text):
        return "biolink:located_in"
    if re.search(r"\b(treat\w*|therap\w*|used\s+for)\b", text):
        return "biolink:treats"
    if re.search(r"\bcontraindicat\w*\b", text):
        return "biolink:contraindicated_for"
    if re.search(r"\b(regulat|regulation)\w*\b", text):
        return "biolink:regulates"
    if re.search(r"\b(affect|influenc|impact|modulat)\w*\b", text):
        return "biolink:affects"
    if re.search(r"\b(caus\w*|lead(?:s)? to|result(?:s)? in|induc\w*|trigger\w*|drive\w*)\b", text):
        return "biolink:causes"

    if object_category in {"biolink:Pathway", "biolink:BiologicalProcess"}:
        if re.search(r"\b(participat|involv|pathway|process)\w*\b", text):
            return "biolink:participates_in"

    if re.search(r"\b(associat|link|related|correlat)\w*\b", text):
        return "biolink:associated_with"

    return None


def _ensure_biolink(curie_or_key: str) -> str:
    if not curie_or_key:
        return "biolink:related_to"
    return curie_or_key if curie_or_key.startswith("biolink:") else f"biolink:{curie_or_key}"


def _slot_key(pred_curie: str) -> str:
    return (pred_curie or "").split(":", 1)[-1].strip()

def _slot_key_variants(key: str) -> List[str]:
    k = (key or "").strip()
    if not k:
        return []
    return list(dict.fromkeys([k, k.replace("_", " "), k.replace(" ", "_")]))


@lru_cache(maxsize=1)
def _load_biolink_slots() -> Dict[str, Dict[str, Any]]:
    candidates: List[Path] = []
    env_path = os.getenv("BIOLINK_YAML")
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "biolink-model.yaml")
    candidates.append(here.parents[2] / "data" / "biolink-model.yaml")

    data: Dict[str, Any] = {}
    for candidate in candidates:
        try:
            if candidate.exists():
                data = yaml.safe_load(candidate.read_text()) or {}
                if isinstance(data, dict) and data.get("slots"):
                    break
        except Exception:
            continue

    slots = (data.get("slots") or {}) if isinstance(data, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in slots.items():
        if isinstance(spec, dict):
            out[str(key)] = spec
    return out


def _slot_meta_by_pred(pred_curie: str) -> Dict[str, Any]:
    slots = _load_biolink_slots()
    base = _slot_key(pred_curie)
    for key in _slot_key_variants(base):
        meta = slots.get(key)
        if isinstance(meta, dict):
            return meta
    return {}


def _slot_exists(pred_curie: str) -> bool:
    return bool(_slot_meta_by_pred(pred_curie))


def _inverse_of(pred_curie: str) -> Optional[str]:
    meta = _slot_meta_by_pred(pred_curie)
    inv = meta.get("inverse")
    if not inv:
        return None
    if isinstance(inv, (list, tuple, set)):
        inv = next(iter(inv), None)
    return _ensure_biolink(str(inv)) if inv else None


def _explicit_related(text: str) -> bool:
    return bool(re.search(r"\b(relat|associat)\w*\b", _clean_text(text)))

def _normalize_class(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.lower().startswith("biolink:"):
        return f"biolink:{text.split(':', 1)[1].strip()}"
    return f"biolink:{text}"


def _compatible(pred_curie: str, subj_cat: Optional[str], obj_cat: Optional[str]) -> bool:
    allowed_sub = allowed_subject_categories(pred_curie)
    allowed_obj = allowed_object_categories(pred_curie)

    if allowed_sub or allowed_obj:
        if subj_cat and subj_cat != "biolink:NamedThing" and allowed_sub:
            if subj_cat not in allowed_sub and not any(is_a(subj_cat, a) for a in allowed_sub):
                return False
        if obj_cat and obj_cat != "biolink:NamedThing" and allowed_obj:
            if obj_cat not in allowed_obj and not any(is_a(obj_cat, a) for a in allowed_obj):
                return False
        return True

    meta = _slot_meta_by_pred(pred_curie)
    domain = _normalize_class(meta.get("domain"))
    range_ = _normalize_class(meta.get("range"))
    if not domain and not range_:
        return True
    if subj_cat and subj_cat != "biolink:NamedThing" and domain:
        if not is_a(subj_cat, domain):
            return False
    if obj_cat and obj_cat != "biolink:NamedThing" and range_:
        if not is_a(obj_cat, range_):
            return False
    return True


def _canonical_candidate(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "biolink:related_to"
    if text.lower().startswith("biolink:"):
        return normalize_pred(text)
    text = text.replace(" ", "_")
    return f"biolink:{text}"


def _iter_candidates(candidates: Optional[Iterable[str]]) -> Iterable[str]:
    for raw in candidates or []:
        canon = _canonical_candidate(raw)
        if canon:
            yield canon


def _try_predicate(
    pred_curie: str,
    subject_category: Optional[str],
    object_category: Optional[str],
    *,
    method: str,
    relationship_text: str,
) -> Optional[Dict[str, Any]]:
    if not pred_curie or not _slot_exists(pred_curie):
        return None

    inverse = _inverse_of(pred_curie)

    if _compatible(pred_curie, subject_category, object_category):
        return {
            "predicate": pred_curie,
            "swap": False,
            "method": method,
            "debug": {
                "relationship_text": relationship_text,
                "predicate": pred_curie,
                "inverse": inverse,
                "decision": "forward",
            },
        }

    if inverse and _slot_exists(inverse) and _compatible(inverse, subject_category, object_category):
        return {
            "predicate": inverse,
            "swap": False,
            "method": method,
            "debug": {
                "relationship_text": relationship_text,
                "predicate": inverse,
                "inverse": inverse,
                "decision": "use_inverse_same_direction",
            },
        }

    if inverse and _slot_exists(inverse) and _compatible(pred_curie, object_category, subject_category):
        return {
            "predicate": inverse,
            "swap": True,
            "method": method,
            "debug": {
                "relationship_text": relationship_text,
                "predicate": inverse,
                "inverse": inverse,
                "decision": "swap_and_use_inverse",
            },
        }

    return None


def ground_onehop_predicate(
    relationship_text: str,
    subject_category: Optional[str],
    object_category: Optional[str],
    *,
    subject_text: str = "",
    object_text: str = "",
    context_text: str = "",
    candidate_preds: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Ground a one-hop predicate with schema-aware directionality.

    Returns:
      {
        "predicate": "biolink:...",
        "swap": bool,
        "method": "exact" | "candidate" | "fallback",
        "debug": {...}
      }
    """
    rel = _strip_stopwords(relationship_text or "")
    if not rel:
        rel = "related to"

    full_text = " ".join(t for t in (relationship_text, context_text) if t)

    lex_pred = _lexical_predicate(
        relationship_text,
        context_text,
        subject_category,
        object_category,
    )
    if lex_pred and _allow_predicate(lex_pred, full_text):
        lex = _try_predicate(
            lex_pred,
            subject_category,
            object_category,
            method="lexical",
            relationship_text=relationship_text,
        )
        if lex:
            return lex

    pred = normalize_pred(rel)
    pred = _ensure_biolink(pred)

    if (pred != "biolink:related_to" or _explicit_related(rel)) and _allow_predicate(pred, full_text):
        exact = _try_predicate(
            pred,
            subject_category,
            object_category,
            method="exact",
            relationship_text=relationship_text,
        )
        if exact:
            return exact

    for cand in _iter_candidates(candidate_preds):
        if not _allow_predicate(cand, full_text):
            continue
        hit = _try_predicate(
            cand,
            subject_category,
            object_category,
            method="candidate",
            relationship_text=relationship_text,
        )
        if hit:
            return hit

    try:
        from .renci_pred_mapping import renci_ground_predicate
        renci = renci_ground_predicate(
            subject_text=subject_text,
            object_text=object_text,
            relationship_text=relationship_text,
            context_text=context_text,
            subject_category=subject_category,
            object_category=object_category,
            top_k=10,
            overfetch=50,
            try_inverse=True,
        )
        if renci and renci.get("predicate") and _allow_predicate(renci["predicate"], full_text):
            hit = _try_predicate(
                renci["predicate"],
                subject_category,
                object_category,
                method="renci_embedding",
                relationship_text=relationship_text,
            )
            if hit:
                hit["debug"]["renci"] = renci
                return hit
    except Exception:
        pass

    return {
        "predicate": "biolink:related_to",
        "swap": False,
        "method": "fallback",
        "debug": {
            "relationship_text": relationship_text,
            "reason": "no exact or candidate match passed schema checks",
        },
    }
