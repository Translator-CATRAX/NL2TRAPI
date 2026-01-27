from __future__ import annotations
from .registry import Route, register
from ..nodes import construct_trapi, validate_trapi

register(Route(
    name="onehop",
    skip_schema=True,                  # one-hop now self-grounds predicate/categories
    construct=construct_trapi.node,    # your current 1-hop constructor
    validate=validate_trapi.node,
))
