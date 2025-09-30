from __future__ import annotations
from .registry import Route, register
from ..nodes.construct_edge_route import make_edge_constructor
from ..nodes.validate_edge import make_edge_validator

construct = make_edge_constructor(
    subj_cats=["biolink:ChemicalEntity"],  # drug
    obj_cats=["biolink:Disease"],
    default_predicate="biolink:treats",
)
validate = make_edge_validator(required_pred="biolink:treats")

register(Route(
    name="treats",
    skip_schema=False,  # we still want ResolveSchema to add class nodes if missing
    construct=construct,
    validate=validate,
))
