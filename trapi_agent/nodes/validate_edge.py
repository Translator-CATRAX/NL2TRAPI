from __future__ import annotations
from ..state_types import TRAPIState

def make_edge_validator(required_pred: str | None = None):
    def node(state: TRAPIState) -> TRAPIState:
        qg = (state.get("output_json", {}).get("message", {}).get("query_graph", {}))
        nodes, edges = qg.get("nodes", {}), qg.get("edges", {})
        errors = []
        if len(nodes) < 2:
            errors.append("Need ≥2 nodes")
        if not edges:
            errors.append("Missing edges")
        else:
            e = next(iter(edges.values()))
            subj = e.get("subject"); obj = e.get("object")
            if subj not in nodes or obj not in nodes:
                errors.append("Edge subject/object must reference existing nodes")
            preds = e.get("predicates", [])
            if not preds:
                errors.append("Edge missing predicates")
            if required_pred and required_pred not in preds:
                errors.append(f"Predicate must include '{required_pred}'")
        state["errors"] = errors
        state["valid"] = not errors
        return state
    return node
