from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Set
from ..state_types import TRAPIState
import logging

logger = logging.getLogger(__name__)

PRUNE_TO_TWO_NODES = True  # avoid stray/unreferenced nodes

def _to_trapi_node(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Map internal node metadata → TRAPI 1.4 node object."""
    out: Dict[str, Any] = {}
    if meta.get("id"):
        out["ids"] = [meta["id"]]
    if meta.get("category"):
        out["categories"] = list(meta["category"])
    if meta.get("name") is not None:
        out["name"] = meta["name"]
    return out

def _pick_by_cat(
    nodes: Dict[str, Dict[str, Any]],
    subj_cats: Set[str], obj_cats: Set[str]
) -> Tuple[str | None, str | None]:
    subj = obj = None
    # prefer pinned subject in requested categories
    for nid, meta in nodes.items():
        cats = set(meta.get("category", []))
        if meta.get("id") and not subj and (cats & subj_cats):
            subj = nid
    # prefer pinned object in requested categories (not equal to subject)
    for nid, meta in nodes.items():
        cats = set(meta.get("category", []))
        if meta.get("id") and nid != subj and not obj and (cats & obj_cats):
            obj = nid
    return subj, obj

def make_edge_constructor(
    subj_cats: Iterable[str],
    obj_cats: Iterable[str],
    default_predicate: str,
):
    subj_cats, obj_cats = set(subj_cats), set(obj_cats)

    def node(state: TRAPIState) -> TRAPIState:
        nodes = state.get("nodes", {}) or {}
        # make sure every node has a name for UI convenience
        for m in nodes.values():
            m.setdefault("name", "")

        subj, obj = _pick_by_cat(nodes, subj_cats, obj_cats)

        # fallbacks: allow class placeholders if pinned not found
        if not subj:
            for nid, m in nodes.items():
                if not m.get("id") and (subj_cats & set(m.get("category", []))):
                    subj = nid; break
        if not obj:
            for nid, m in nodes.items():
                if nid != subj and (not m.get("id")) and (obj_cats & set(m.get("category", []))):
                    obj = nid; break

        # last resort: first two nodes if available
        if (not subj or not obj) and len(nodes) >= 2:
            nids = list(nodes.keys())
            subj, obj = nids[0], nids[1]

        # If we still don't have a valid pair, emit nodes-only; validator will flag it.
        if not subj or not obj or subj == obj:
            qg_nodes = {nid: _to_trapi_node(m) for nid, m in nodes.items()}
            if PRUNE_TO_TWO_NODES and len(qg_nodes) > 2:
                keep = list(qg_nodes.keys())[:2]
                qg_nodes = {k: qg_nodes[k] for k in keep}
            state["output_json"] = {"message": {"query_graph": {"nodes": qg_nodes, "edges": {}}}}
            return state

        # prune to only the nodes referenced by the edge
        qg_nodes = nodes
        if PRUNE_TO_TWO_NODES:
            qg_nodes = {subj: nodes[subj], obj: nodes[obj]}
            state["nodes"] = qg_nodes  # keep state consistent with output

        # normalize predicate defensively
        pred = default_predicate if str(default_predicate).startswith("biolink:") else "biolink:related_to"

        edge = {"subject": subj, "object": obj, "predicates": [pred]}
        state["output_json"] = {
            "message": {
                "query_graph": {
                    "nodes": {nid: _to_trapi_node(m) for nid, m in qg_nodes.items()},
                    "edges": {"e0": edge}
                }
            }
        }
        return state

    return node
