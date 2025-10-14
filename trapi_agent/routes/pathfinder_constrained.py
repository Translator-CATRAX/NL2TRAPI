from __future__ import annotations
from .registry import Route, register
from ..nodes import construct_pathfinder_constrained, validate_pathfinder_constrained

# Pathfinder-constrained: 2 pinned nodes + related_to + intermediate_categories constraint
register(Route(
    name="pathfinder_constrained",
    skip_schema=True,  # schema not needed; predicate fixed; class constraint supplied directly
    construct=construct_pathfinder_constrained.node,
    validate=validate_pathfinder_constrained.node,
))
