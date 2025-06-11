# query_generation_refactored.py — FINAL Version with Hybrid Resolution and Structured Prompting

import chromadb
import json
import logging
import time
import torch
import re
from transformers import pipeline
from datetime import datetime
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from functools import wraps
from node_resolver import NodeResolver

# ========== Logging Setup ==========
logger = logging.getLogger("TRAPIQueryGen")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] [%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(f"trapi_querygen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

# ========== Constants ==========
EXCLUDED_CATEGORIES = {
    "biolink:ClinicalAttribute", "biolink:InformationContentEntity", "biolink:Event",
    "biolink:ExposureEvent", "biolink:GeographicLocation", "biolink:PopulationOfIndividualOrganisms",
    "biolink:Publication", "biolink:RetrievalSource", "biolink:StudyPopulation", "biolink:SubjectOfInvestigation"
}

STOPWORDS = {"a", "an", "the", "is", "are", "what", "and", "to", "of"}

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device="cuda"
)

# ========== Utility ==========
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time.time() - t0:.2f}s")
        return result
    return wrapper

@timed
def initialize_chroma_db(path):
    return chromadb.PersistentClient(path=path)

@timed
def search_chroma(chroma_client, collection_name, query, top_k=1):
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        return collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        logger.error(f"Search error in '{collection_name}': {e}")
        return None

# ========== Entity & Schema Resolver ==========
@timed
def resolve_entities_and_schema_terms(query, resolver, chroma_client, schema_top_k=1):
    resolved_nodes = []
    schema_matches = []
    matched_indices = set()
    tokens = query.lower().split()

    for n in range(3, 0, -1):
        for i in range(len(tokens) - n + 1):
            if any(j in matched_indices for j in range(i, i + n)):
                continue
            ngram = " ".join(tokens[i:i+n])
            node_result = resolver.resolve(ngram)
            if node_result:
                if node_result[0].get("category") not in EXCLUDED_CATEGORIES:
                    resolved_nodes.extend(node_result)
                    matched_indices.update(range(i, i+n))
                continue
            if n == 1 and ngram in STOPWORDS:
                continue
            schema_result = search_chroma(chroma_client, "yaml_schema", ngram, top_k=schema_top_k)
            if schema_result and schema_result.get("documents") and schema_result["documents"][0]:
                schema_result["query_term"] = ngram
                schema_matches.append(schema_result)
                matched_indices.update(range(i, i+n))

    unique_resolved_nodes = [dict(t) for t in {tuple(d.items()) for d in resolved_nodes}]
    return unique_resolved_nodes, schema_matches

# ========== Prompt Construction ==========
def construct_prompt(nl_query, examples, schema_matches, resolved_nodes):
    base_prompt = (
        "You are an expert at converting natural language queries into TRAPI JSON format.\n"
        "- Use the 'Identified Specific Entities' for nodes with provided CURIEs.\n"
        "- Use 'Potential Biolink Schema Matches' to assign categories and predicates.\n"
        "- Do not hallucinate CURIEs.\n"
        "- Only output the TRAPI query_graph JSON object.\n\n"
    )
    prompt = base_prompt + "--- CONTEXT FOR YOUR TASK ---\n"

    if not resolved_nodes and not schema_matches:
        prompt += "No specific context was found. Rely on the provided examples and your general knowledge of TRAPI.\n"

    if resolved_nodes:
        prompt += "Identified Specific Entities (Nodes with CURIEs):\n"
        for node in resolved_nodes:
            name = node.get('name', 'N/A')
            node_id = node.get('id', '?')
            category = node.get('category', 'biolink:NamedThing')
            description = node.get('description', '')[:150].strip()
            prompt += f"- {name}: ID={node_id}, Category={category}, Desc={description}\n"

    if schema_matches:
        prompt += "\nPotential Biolink Schema Matches:\n"
        for match in schema_matches:
            term = match['query_term']
            for i in range(len(match["documents"][0])):
                key = match["metadatas"][0][i].get("key")
                desc = match["metadatas"][0][i].get("description", "")[:150].strip()
                prompt += f"- '{term}' → {key}: {desc}\n"

    if examples:
        prompt += "\n--- EXAMPLES ---\n"
        for i in range(len(examples["documents"][0])):
            nl = examples["documents"][0][i]
            trapi = examples["metadatas"][0][i].get("trapi_query", "{}")
            try:
                trapi_json = json.loads(trapi)
                prompt += f"NL Query: {nl}\nTRAPI query_graph: {json.dumps(trapi_json, indent=2)}\n\n"
            except json.JSONDecodeError:
                continue

    prompt += f"--- YOUR TASK ---\nQuery: \"{nl_query}\"\n\nTRAPI query_graph:"
    logger.info("Prompt Preview:\n" + prompt[:2000])
    return prompt

# ========== LLM Interaction ==========
@timed
def initialize_mistral():
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="auto"
    )

def extract_trapi_query(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logger.error(f"JSON parsing failed: {e}")
    return None

@timed
def generate_trapi(nl_query, examples, schema_matches, resolved_nodes, mistral):
    prompt = construct_prompt(nl_query, examples, schema_matches, resolved_nodes)
    result = mistral(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )[0]["generated_text"]

    logger.info("Raw LLM Response:\n" + result)
    return extract_trapi_query(result)

# ========== Main ==========
@timed
def main():
    nl_query = "What biological processes are related to GFAP"
    logger.info(f"Received query: '{nl_query}'")

    db_examples = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db")
    db_nodeschema = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/chroma_db")
    examples = search_chroma(db_examples, "nl_to_trapi", nl_query, top_k=3)

    resolver = NodeResolver(db_nodeschema, index_path="/scratch/vmm5481/NL2TRAPI/exact_index.pkl")
    resolved_nodes, schema_matches = resolve_entities_and_schema_terms(nl_query, resolver, db_nodeschema)

    mistral = initialize_mistral()
    trapi_query = generate_trapi(nl_query, examples, schema_matches, resolved_nodes, mistral)

    if trapi_query:
        logger.info("\n\u2705 Final TRAPI Query:")
        print(json.dumps(trapi_query, indent=2))
    else:
        logger.error("\u274C Failed to generate a valid TRAPI query.")

if __name__ == "__main__":
    main()
