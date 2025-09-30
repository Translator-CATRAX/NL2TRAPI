from __future__ import annotations
from .registry import Route, register
from ..nodes import construct_pathfinder, validate_pathfinder

register(Route(
    name="pathfinder",
    skip_schema=True,
    construct=construct_pathfinder.node,
    validate=validate_pathfinder.node,
))
