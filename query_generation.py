# query_generation_refactored.py — with NodeResolver, clean timing, and optimized resolution

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
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] [%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(f"trapi_querygen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
fh.setFormatter(formatter)
logger.addHandler(fh)

# ========== Utility ==========
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time.time() - t0:.2f}s")
        return result
    return wrapper

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device="cuda"
)

# ========== Chroma Init ==========
@timed
def initialize_chroma_db(path):
    return chromadb.PersistentClient(path=path)

@timed
def search_chroma(chroma_client, collection_name, query, top_k=5):
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        return collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        logger.error(f"Search error in '{collection_name}': {e}")
        return None

# ========== Prompt ==========
def construct_prompt(nl_query, examples, schema, resolved_nodes):
    prompt = "You are an expert at converting natural language queries into TRAPI JSON format.\n\n"
    if examples:
        for i in range(len(examples["documents"][0])):
            nl = examples["documents"][0][i]
            trapi = examples["metadatas"][0][i].get("trapi_query", "N/A")
            prompt += f"Example {i+1}:\nNL Query: {nl}\nTRAPI Query: {trapi}\n\n"
    if schema:
        prompt += "Schema Knowledge:\n"
        for i in range(len(schema["documents"][0])):
            k = schema["metadatas"][0][i].get("key", "Unknown")
            d = schema["documents"][0][i]
            prompt += f"- {k}: {d}\n"
    if resolved_nodes:
        prompt += "\nNode Context:\n"
        for node in resolved_nodes:
            name = node["name"]
            node_id = node["id"]
            cat = node["category"]
            desc = node.get("description", "")
            prompt += f"{name} | {node_id} | {cat}: {desc}\n"
    prompt += f"\nNow, generate a TRAPI JSON for this natural language query:\n{nl_query}\n\nTRAPI JSON:"
    logger.info("Prompt Preview:\n" + prompt[:1000] + ("... [truncated]" if len(prompt) > 1000 else ""))
    return prompt

# ========== LLM + TRAPI ==========
@timed
def initialize_mistral():
    logger.info("Loading Mistral...")
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

def extract_trapi_query(llm_response):
    try:
        lines = llm_response.splitlines()
        start = next((i for i, l in enumerate(lines) if "TRAPI JSON" in l), None)
        if start is not None:
            json_block = "\n".join(lines[start+1:]).strip("` \n")
            return json.loads(json_block)
    except Exception as e:
        logger.error(f"Failed to parse TRAPI JSON: {e}")
    return None

@timed
def generate_trapi(nl_query, examples, schema, resolved_nodes, mistral):
    prompt = construct_prompt(nl_query, examples, schema, resolved_nodes)
    response = mistral(
        prompt,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=0.7,
        max_new_tokens=256,
        num_return_sequences=1,
        eos_token_id=mistral.tokenizer.eos_token_id
    )[0]["generated_text"]
    logger.info("Raw LLM Response:\n" + response[:1000] + ("... [truncated]" if len(response) > 1000 else ""))
    return extract_trapi_query(response)

# ========== Main ==========
@timed
def main():
    nl_query = "What biological processes are related to GFAP"
    logger.info(f"Received query: '{nl_query}'")

    db_examples = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db")
    db_nodeschema = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/chroma_db")
    examples = search_chroma(db_examples, "nl_to_trapi", nl_query, top_k=5)

    resolver = NodeResolver(db_nodeschema, index_path="/scratch/vmm5481/NL2TRAPI/exact_index.pkl")
    resolved_nodes = resolver.resolve(nl_query)
    logger.info(f"Resolved {len(resolved_nodes)} nodes")

    schema = None
    if not resolved_nodes:
        schema = search_chroma(db_nodeschema, "yaml_schema", nl_query, top_k=6)

    mistral = initialize_mistral()
    trapi_query = generate_trapi(nl_query, examples, schema, resolved_nodes, mistral)

    if trapi_query:
        logger.info("\n✅ Final TRAPI Query:")
        print(json.dumps(trapi_query, indent=2))
    else:
        logger.error("❌ Failed to generate a valid TRAPI query.")

if __name__ == "__main__":
    main()
