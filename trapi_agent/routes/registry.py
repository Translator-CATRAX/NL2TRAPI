from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict
from ..state_types import TRAPIState

NodeFn = Callable[[TRAPIState], TRAPIState]

@dataclass(frozen=True)
class Route:
    name: str
    skip_schema: bool
    construct: NodeFn
    validate: NodeFn
    fix_trapi: bool = False  

ROUTES: Dict[str, Route] = {}

def register(route: Route) -> None:
    ROUTES[route.name] = route
