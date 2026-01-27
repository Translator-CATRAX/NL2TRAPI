# below code is after updating the pathfinder logic and  1 hop with quantity 2 and 1st pinned and 2nd unppinned (direction/ topplogy ?)

#!/usr/bin/env python3
"""
construct_trapi.py

LangGraph node that assembles a minimal TRAPI 1.4+ query_graph from
resolved nodes and a predicate.

Heuristics
----------
• Prefer SUBJECT = the first *unpinned* node (no 'id' and not 'pinned').
• Prefer OBJECT  = the first *pinned* node (has 'id').
• If those don't exist, fall back gracefully to the first two nodes.

Other niceties
--------------
• Ensure each node has a 'name' (empty string if missing).
• Ensure predicate is a 'biolink:*' CURIE (fallback 'biolink:related_to').
• (Default) keep only the two nodes used by e0 to avoid stray nodes.
• Emit TRAPI 1.4 node fields: 'ids' and 'categories' (not internal 'id'/'category').
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..state_types import TRAPIState

logger = logging.getLogger(__name__)

# Set this to False if you want to keep any extra nodes that may exist in state["nodes"]
PRUNE_TO_TWO_NODES = True


def _choose_subject_object(nodes: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    """Pick subject and object node ids following the heuristic described above."""
    node_ids: List[str] = list(nodes.keys())
    if len(node_ids) >= 2:
        # Partition nodes
        unpinned = [nid for nid, data in nodes.items() if not data.get("id") and not data.get("pinned")]
        pinned   = [nid for nid, data in nodes.items() if data.get("id")]

        # SUBJECT: prefer unpinned
        subj = unpinned[0] if unpinned else node_ids[0]

        # OBJECT: prefer pinned (not equal to subj)
        obj = None
        for nid in pinned:
            if nid != subj:
                obj = nid
                break

        # If we didn't find a pinned object, try another unpinned (not subj)
        if obj is None:
            for nid in unpinned:
                if nid != subj:
                    obj = nid
                    break

        # Final fallback: first node that isn't subj; else the first two nodes
        if obj is None:
            for nid in node_ids:
                if nid != subj:
                    obj = nid
                    break

        if obj is None and len(node_ids) >= 2:
            subj, obj = node_ids[0], node_ids[1]

        return subj, obj

    # If fewer than 2 nodes, just mirror (caller will handle insufficiency)
    only = node_ids[0] if node_ids else "n0"
    return only, only


def _choose_subject_object_by_role(
    nodes: Dict[str, Dict[str, Any]],
    subject_role: str | None,
    object_role: str | None,
) -> Tuple[str, str]:
    node_ids: List[str] = list(nodes.keys())
    pinned = [nid for nid, data in nodes.items() if data.get("id")]
    unpinned = [nid for nid, data in nodes.items() if not data.get("id") and not data.get("pinned")]

    def _pick(role: str | None, exclude: str | None = None) -> Optional[str]:
        pool = node_ids
        if role == "pinned":
            pool = pinned or node_ids
        elif role == "unpinned":
            pool = unpinned or node_ids
        for nid in pool:
            if nid != exclude:
                return nid
        return None

    subj = _pick(subject_role)
    obj = _pick(object_role, exclude=subj)

    if subj is None or obj is None:
        return _choose_subject_object(nodes)

    if subj == obj and len(node_ids) >= 2:
        subj, obj = node_ids[0], node_ids[1]

    return subj, obj


def _find_node_id_by_text(nodes: Dict[str, Dict[str, Any]], text: str) -> Optional[str]:
    norm = (text or "").strip().lower()
    if not norm:
        return None
    for nid, meta in nodes.items():
        if (meta.get("name") or "").strip().lower() == norm:
            return nid
    return None


def _to_trapi_node(meta: Dict[str, Any], label_map: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Map internal node metadata → TRAPI 1.4 node object."""
    out: Dict[str, Any] = {}
    curie = meta.get("id")
    if curie:
        out["ids"] = [curie]
    if meta.get("category"):
        # internal key 'category' (list[str]) → TRAPI 'categories'
        out["categories"] = list(meta["category"])
    # 'name' isn't required by TRAPI but is handy for UI/debug
    name = meta.get("name")
    if curie and label_map:
        name = label_map.get(curie) or name
    if not name and curie:
        name = curie
    if name:
        out["name"] = name
    return out


def node(state: TRAPIState) -> TRAPIState:
    """
    Build a single-edge TRAPI query_graph from 'nodes' and 'predicate'.
    Writes:
      state['edges']
      state['output_json'] = { 'message': { 'query_graph': {...} } }
    """
    nodes_dict: Dict[str, Dict[str, Any]] = state.get("nodes", {}) or {}
    predicate_curie: str = (state.get("predicate") or "").strip() or "biolink:related_to"
    if not predicate_curie.startswith("biolink:"):
        predicate_curie = "biolink:related_to"

    # Need at least two nodes
    if len(nodes_dict) < 2:
        logger.warning("Insufficient nodes to construct edge: %d found", len(nodes_dict))
        state["output_json"] = {}
        return state

    # Pick subject/object (onehop can provide explicit roles)
    onehop = state.get("onehop_parse") or {}
    if state.get("route") == "onehop" and onehop:
        subj_id = _find_node_id_by_text(nodes_dict, onehop.get("subject_text", ""))
        obj_id = _find_node_id_by_text(nodes_dict, onehop.get("object_text", ""))
        if not subj_id or not obj_id:
            role_subj, role_obj = _choose_subject_object_by_role(
                nodes_dict,
                onehop.get("subject_role"),
                onehop.get("object_role"),
            )
            subj_id = subj_id or role_subj
            obj_id = obj_id or role_obj
        if subj_id == obj_id:
            subj_id, obj_id = _choose_subject_object(nodes_dict)
    else:
        subj_id, obj_id = _choose_subject_object(nodes_dict)
    logger.debug("Selected subject '%s', object '%s'", subj_id, obj_id)

    # Ensure each node has a 'name' key (UI convenience)
    for meta in nodes_dict.values():
        meta.setdefault("name", "")

    label_map: Dict[str, str] = state.get("curie_labels", {})

    # Optionally prune to just the two nodes we use (keeps one-hop QG clean)
    if PRUNE_TO_TWO_NODES:
        kept = {subj_id: nodes_dict[subj_id], obj_id: nodes_dict[obj_id]}
        nodes_dict = kept
        # Re-assign to state so downstream validation sees the pruned set
        state["nodes"] = nodes_dict

    # Build TRAPI-compliant nodes
    qg_nodes: Dict[str, Dict[str, Any]] = {nid: _to_trapi_node(meta, label_map) for nid, meta in nodes_dict.items()}

    # Build the single edge
    edge_id = "e0"
    edges: Dict[str, Dict[str, Any]] = {
        edge_id: {
            "subject": subj_id,
            "object": obj_id,
            "predicates": [predicate_curie],
        }
    }

    # Persist to state
    state["edges"] = edges
    state["output_json"] = {
        "message": {
            "query_graph": {
                "nodes": qg_nodes,
                "edges": edges,
            }
        }
    }

    logger.info("Constructed query_graph with %d node(s) and 1 edge", len(qg_nodes))
    return state
