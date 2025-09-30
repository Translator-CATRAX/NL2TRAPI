# #!/usr/bin/env python3
# """
# trapi_agent/service.py
# ──────────────────────
# • Helper  `run_agent(query:str) -> {"message": <TRAPI>}`
# • FastAPI endpoints:
#     • POST /nl2trapi   { "query": "…" }
#     • GET  /healthz    → { "status": "ok" }
# """

# from __future__ import annotations

# import logging
# from functools import lru_cache

# from fastapi import FastAPI
# from pydantic import BaseModel, Field

# from .agent_graph import graph                # compiled LangGraph
# from .state_types import TRAPIState

# logger = logging.getLogger(__name__)

# # ──────────────────────────  internal helpers  ──────────────────────────────
# @lru_cache(maxsize=1)
# def _cached_graph():
#     """Return (and cache) the compiled LangGraph instance."""
#     return graph


# # ──────────────────────────  public helper  ─────────────────────────────────
# def run_agent(query: str) -> dict:
#     """
#     Execute the NL→TRAPI pipeline and return a JSON-serialisable TRAPI message.

#     Raises
#     ------
#     KeyError
#         If ConstructTRAPI failed to populate ``state["trapi"]`` (shouldn’t happen
#         in the happy path—ValidateTRAPI will have caught structural issues).
#     """
#     state: TRAPIState = {"query": query}
#     result = _cached_graph().invoke(state)

#     # ConstructTRAPI now guarantees this key exists
#     return {"message": result["trapi"]}


# # ──────────────────────────  FastAPI surface  ───────────────────────────────
# app = FastAPI(
#     title="NL→TRAPI Agent",
#     description="Converts natural-language biomedical questions into a TRAPI query_graph.",
#     version="0.1.0",
# )

# # ----- request / response models -------------------------------------------
# class NLQuery(BaseModel):
#     query: str = Field(..., example="What proteins does acetaminophen interact with?")


# class TRAPIResponse(BaseModel):
#     message: dict


# # ----- routes ---------------------------------------------------------------
# @app.post("/nl2trapi", response_model=TRAPIResponse, tags=["convert"])
# async def nl2trapi(body: NLQuery):
#     """Main conversion endpoint."""
#     return run_agent(body.query)


# @app.get("/healthz", tags=["meta"])
# async def healthz():
#     """Simple liveness probe (for Docker/K8s)."""
#     return {"status": "ok"}
