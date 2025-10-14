from __future__ import annotations
from .registry import Route, register
from ..nodes import construct_xcrg, validate_xcrg

register(Route(
    name="xcrg",
    skip_schema=True,               # fixed shape; no schema/RAG
    construct=construct_xcrg.node,  # build the boilerplate graph
    validate=validate_xcrg.node,    # enforce shape + qualifiers
    fix_trapi=True,                 # optional; keeps behavior consistent with other routes
))
