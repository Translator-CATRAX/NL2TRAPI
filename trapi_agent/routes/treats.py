# from __future__ import annotations
# from .registry import Route, register
# from ..nodes.construct_edge_route import make_edge_constructor
# from ..nodes.validate_edge import make_edge_validator

# construct = make_edge_constructor(
#     subj_cats=["biolink:ChemicalEntity"],  # drug
#     obj_cats=["biolink:Disease"],
#     default_predicate="biolink:treats",
# )
# validate = make_edge_validator(required_pred="biolink:treats")

# register(Route(
#     name="treats",
#     skip_schema=False,  # we still want ResolveSchema to add class nodes if missing
#     construct=construct,
#     validate=validate,
# ))

# #!/usr/bin/env python3
# """
# Route: treats  (xDTD — ChemicalEntity → treats → Disease)

# This route builds a fixed TRAPI query_graph of the form:

#   ChemicalEntity ──biolink:treats──▶ Disease

# • Subject (sn):  unpinned  biolink:ChemicalEntity
# • Object  (on):  pinned    biolink:Disease  (resolved via NodeNorm)
# • Edge:          biolink:treats, knowledge_type=inferred

# We skip schema lookup since this shape is fully defined.
# """

# from __future__ import annotations
# from .registry import Route, register
# from ..nodes import construct_treats, validate_treats

# # ────────────────────────────────────────────────────────────────────────────────
# # Register route
# # ────────────────────────────────────────────────────────────────────────────────
# register(Route(
#     name="treats",
#     skip_schema=True,                # ✅ no need for Biolink schema RAG step
#     construct=construct_treats.node, # builds boilerplate query_graph
#     validate=validate_treats.node,
#     fix_trapi=True,   # structural + semantic validation
# ))
# trapi_agent/routes/treats.py
from __future__ import annotations
from .registry import Route, register
from ..nodes import construct_treats, validate_treats

# xDTD: Drug (ChemicalEntity) --[biolink:treats]--> Disease
# We rely on ResolveEntities to pin the Disease CURIE. Schema step is not needed.
register(Route(
    name="treats",
    skip_schema=True,                 # ← IMPORTANT: schema is skipped for this fixed-shape route
    construct=construct_treats.node,  # builds the exact boilerplate with 'sn', 'on', 't_edge'
    validate=validate_treats.node,
    fix_trapi=True,   # checks biolink:treats + pinned disease id
))
