from __future__ import annotations
from .registry import Route, register
from ..nodes.construct_edge_route import make_edge_constructor
from ..nodes.validate_edge import make_edge_validator

construct = make_edge_constructor(
    subj_cats=["biolink:ChemicalEntity"],
    obj_cats=["biolink:Gene", "biolink:Protein"],
    default_predicate="biolink:physically_interacts_with",  # or related_to
)
validate = make_edge_validator()  # no fixed predicate required

register(Route(
    name="chem_gene",
    skip_schema=False,
    construct=construct,
    validate=validate,
))
