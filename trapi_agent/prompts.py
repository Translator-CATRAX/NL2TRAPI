# trapi_agent/prompts.py

"""
Prompt templates for the NL→TRAPI agent pipeline.

This module defines the three core prompt formats used in the LangGraph:
  1. PARSE_TEMPLATE      – Extract entities, generic types, and predicate hint.
  2. FIX_TEMPLATE        – Repair malformed TRAPI query_graph JSON.
  3. CONSTRUCT_TEMPLATE  – Assemble final TRAPI query_graph from context.
"""

# ─── PARSE TEMPLATE ────────────────────────────────────────────────────────
PARSE_TEMPLATE: str = """
You are a biomedical NLP assistant.
The NER tool already spotted these spans in the question:
{spans}

These are the extracted entity strings:
{entities}

These are the generic Biolink types for those entities:
{types}

Return EXACTLY this JSON (no extra keys) and nothing else:
{{"entities": {entities}, "generic_types": {types}, "predicate": "<string>"}}

Possible Biolink predicates you can pick from (choose one):
{predicate_choices}

<|end|>

Question: "{query}"
JSON:
"""

# ─── FIX TEMPLATE ─────────────────────────────────────────────────────────
FIX_TEMPLATE: str = """
You are given a TRAPI query_graph JSON that failed validation.
Errors:
{errors}

Please output ONLY a corrected `query_graph` JSON object,
with all required fields (no markdown, no commentary).

Broken JSON:
{json}
"""

# ─── CONSTRUCT TEMPLATE ───────────────────────────────────────────────────
CONSTRUCT_TEMPLATE: str = """
You are a TRAPI specialist. Given the following information:
  • Resolved nodes and their categories:
{node_context}

  • Schema matches and few-shot examples (if any):
{schema_hits}
{few_shot}

Question: "{query}"

Return ONLY the final `query_graph` JSON object that adheres to TRAPI specs.
"""
