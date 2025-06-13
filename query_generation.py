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
STOPWORDS = {"a", "an", "the", "is", "are", "what", "and", "to", "of"}

EXCLUDED_CATEGORIES = {
    "biolink:ClinicalAttribute", "biolink:InformationContentEntity", "biolink:Event",
    "biolink:ExposureEvent", "biolink:GeographicLocation", "biolink:PopulationOfIndividualOrganisms",
    "biolink:Publication", "biolink:RetrievalSource", "biolink:StudyPopulation", "biolink:SubjectOfInvestigation"
}

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

@timed
def resolve_entities_and_schema_terms(query, resolver, chroma_client, schema_top_k=1):
    tokens = query.lower().split()
    resolved_nodes = []
    schema_matches = []
    matched_tokens = set()

    for token in tokens:
        if token in STOPWORDS or token in matched_tokens:
            continue
        node = resolver.exact_match(token)
        if node:
            category = node.get("category", "")
            if category not in EXCLUDED_CATEGORIES:
                resolved_nodes.append(node)
                matched_tokens.add(token)
        else:
            schema = search_chroma(chroma_client, "yaml_schema", token, top_k=schema_top_k)
            if schema and schema.get("documents") and schema["documents"][0]:
                schema["query_term"] = token
                schema_matches.append(schema)
                matched_tokens.add(token)
    return resolved_nodes, schema_matches

def construct_prompt(nl_query, examples, schema_matches, resolved_nodes):
    prompt = """You are an expert at converting natural language queries into TRAPI JSON format.
- Use the 'Identified Specific Entities' for specific(pinned) nodes with provided CURIEs.
- Use 'Potential Biolink Schema Matches' for generic node and predicates.
- Only output the TRAPI query_graph JSON object.

--- CONTEXT FOR YOUR TASK ---
"""

    if resolved_nodes:
        prompt += "Identified Specific Entities (Nodes with CURIEs):\n"
        for node in resolved_nodes:
            name = node.get("name", "?")
            node_id = node.get("id", "?")
            category = node.get("category", "?")
            desc = node.get("description", "").strip()[:150]
            prompt += f"- {name}: ID={node_id}, Category={category}, Desc={desc}\n"

    if schema_matches:
        prompt += "\nPotential Biolink Schema Matches:\n"
        for schema in schema_matches:
            query_term = schema["query_term"]
            for i in range(len(schema["documents"][0])):
                key = schema["metadatas"][0][i].get("key", "?")
                desc = schema["metadatas"][0][i].get("description", "")[:150]
                prompt += f"- {query_term} → {key}: {desc}\n"

    if examples:
        prompt += "\n--- EXAMPLES ---\n"
        for i in range(len(examples["documents"][0])):
            nl = examples["documents"][0][i]
            trapi = examples["metadatas"][0][i].get("trapi_query", "{}")
            try:
                trapi_json = json.loads(trapi)
                prompt += f"NL Query: {nl}\nTRAPI query_graph: {json.dumps(trapi_json, indent=2)}\n\n"
            except Exception:
                continue

    prompt += f"--- YOUR TASK ---\nQuery: \"{nl_query}\"\n\nTRAPI query_graph:"
    logger.info("Prompt Preview:\n" + prompt[:1500])
    return prompt

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
        logger.error(f"TRAPI parsing failed: {e}")
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
        logger.info("\n✅ Final TRAPI Query:")
        print(json.dumps(trapi_query, indent=2))
    else:
        logger.error("❌ Failed to generate a valid TRAPI query.")

if __name__ == "__main__":
    main()
