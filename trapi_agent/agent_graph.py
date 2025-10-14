
from __future__ import annotations

import logging
import importlib
from typing import Any
from langgraph.graph import StateGraph, END

from .state_types import TRAPIState
from .config import settings
from .nodes import parse_query, resolve_entities, resolve_schema
from .nodes import router as router_node
from .nodes import fix_trapi as fix_node
from .routes import registry as R  # central route registry

logger = logging.getLogger(__name__)


def _import_route_modules() -> None:
    # Register all available routes (explicit routing uses these)
    for mod in ("onehop", "pathfinder", "treats", "chem_gene", "xcrg","pathfinder_constrained"):
        try:
            importlib.import_module(f"{__package__}.routes.{mod}")
            logger.debug("Imported route module: %s", mod)
        except Exception as e:
            logger.warning(" Failed importing route '%s': %s", mod, e)


# Import at module import time (before graph is built)
_import_route_modules()

# Optional direct fallbacks if pathfinder not registered but nodes exist
try:
    from .nodes import construct_pathfinder as _cpf
    from .nodes import validate_pathfinder as _vpf
except Exception:
    _cpf = _vpf = None  # type: ignore


def build_agent_graph() -> Any:
    g = StateGraph(TRAPIState)

    # ── Spine ─────────────────────────────────────────────────────────────────
    g.add_node("ParseQuery",        parse_query.node)
    g.add_node("Router",            router_node.node)
    g.add_node("ResolveEntities",   resolve_entities.node)

    def _set_route_flags(state: TRAPIState) -> TRAPIState:
        inbound = state.get("route")
        route = inbound or "onehop"
        handler = R.ROUTES.get(route)

        if handler is None:
            # Keep the requested route string; only guess skip_schema for pathfinder
            state["route"] = route
            state["skip_schema"] = (route == "pathfinder")
            logger.info(
                "SetRouteFlags: inbound='%s' → kept route='%s' (registered=%s, skip_schema=%s)",
                inbound, state["route"], False, state["skip_schema"],
            )
            return state

        # Canonicalize to the registered handler
        state["route"] = handler.name
        state["skip_schema"] = handler.skip_schema
        logger.info(
            "SetRouteFlags: inbound='%s' → handler='%s' (skip_schema=%s)",
            inbound, handler.name, handler.skip_schema,
        )
        return state

    g.add_node("SetRouteFlags", _set_route_flags)

    # Schema resolver: no-ops if state['skip_schema'] is True
    g.add_node("ResolveSchema", resolve_schema.node)

    # Route-dispatched nodes (safe lookup + direct fallbacks)
    def _construct(state: TRAPIState) -> TRAPIState:
        route = state.get("route") or "onehop"
        handler = R.ROUTES.get(route)
        if handler is not None:
            state["route"] = handler.name
            return handler.construct(state)

        if route == "pathfinder" and _cpf is not None:
            state["route"] = "pathfinder"
            return _cpf.node(state)

        onehop = R.ROUTES.get("onehop")
        if onehop is None:
            raise RuntimeError("Required route 'onehop' failed to register.")
        state["route"] = onehop.name
        return onehop.construct(state)

    def _validate(state: TRAPIState) -> TRAPIState:
        route = state.get("route") or "onehop"
        handler = R.ROUTES.get(route)
        if handler is not None:
            state["route"] = handler.name
            return handler.validate(state)

        if route == "pathfinder" and _vpf is not None:
            state["route"] = "pathfinder"
            return _vpf.node(state)

        onehop = R.ROUTES.get("onehop")
        if onehop is None:
            raise RuntimeError("Required route 'onehop' failed to register.")
        state["route"] = onehop.name
        return onehop.validate(state)

    g.add_node("Construct", _construct)
    g.add_node("Validate",  _validate)

    # LLM fixer (route-agnostic)
    g.add_node("Fix", fix_node.node)

    # ── Edges (linear flow + repair loop) ─────────────────────────────────────
    g.set_entry_point("ParseQuery")
    g.add_edge("ParseQuery",      "Router")
    g.add_edge("Router",          "ResolveEntities")
    g.add_edge("ResolveEntities", "SetRouteFlags")
    g.add_edge("SetRouteFlags",   "ResolveSchema")
    g.add_edge("ResolveSchema",   "Construct")
    g.add_edge("Construct",       "Validate")

    g.add_conditional_edges(
        "Validate",
        {
            END:  lambda s: s.get("valid", False),
            "Fix": lambda s: not s.get("valid", False),
        }
    )
    g.add_conditional_edges(
        "Fix",
        {
            "Validate": lambda s: s.get("fix_attempts", 0) < settings.MAX_FIX_ATTEMPTS,
            END:        lambda s: s.get("fix_attempts", 0) >= settings.MAX_FIX_ATTEMPTS,
        }
    )

    compiled = g.compile()
    logger.info("✅ LangGraph compiled (linear spine + route flags + LLM repair loop).")
    try:
        logger.info("Registered routes: %s", sorted(R.ROUTES.keys()))
    except Exception:
        pass
    return compiled


# Global
graph = build_agent_graph()
