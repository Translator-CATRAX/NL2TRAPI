import re
import json
import logging
from datetime import datetime
from pathlib import Path

import chromadb
import torch
from transformers import pipeline
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# -------------------- Configuration --------------------
class Config:
    EXAMPLES_DB       = "/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db"
    SCHEMA_DB         = "/scratch/vmm5481/NL2TRAPI/chroma_db"
    MODEL_NAME        = "mistralai/Mistral-7B-Instruct-v0.1"
    EMBEDDING_MODEL   = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    DEVICE            = "cuda"
    TOP_K_EXAMPLES    = 5
    TOP_K_SCHEMA      = 2
    MAX_NEW_TOKENS    = 512
    LOG_DIR           = "experiment_logs"

# -------------------- Logging Setup --------------------

def setup_logging():
    Path(Config.LOG_DIR).mkdir(exist_ok=True)
    logfile = Path(Config.LOG_DIR) / f"nl2trapi_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        filename=logfile,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)
    logging.info(f"Experiment log: {logfile}")

setup_logging()

# -------------------- Constants --------------------
STOPWORDS = {"what","which","is","are","a","an","the","in","of","to","for","and","on","with","by","how","at","that"}
EXCLUDED_CATEGORIES = {
    "biolink:ClinicalAttribute","biolink:InformationContentEntity","biolink:Event",
    "biolink:ExposureEvent","biolink:GeographicLocation",
    "biolink:PopulationOfIndividualOrganisms","biolink:Publication",
    "biolink:RetrievalSource","biolink:StudyPopulation","biolink:SubjectOfInvestigation"
}

# -------------------- Embedding --------------------
logging.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=Config.EMBEDDING_MODEL,
    device=Config.DEVICE
)

# -------------------- ChromaDB Helpers --------------------

def init_client(path: str) -> chromadb.PersistentClient:
    logging.info(f"Initializing ChromaDB at {path}")
    client = chromadb.PersistentClient(path=path)
    logging.info("ChromaDB initialized")
    return client


def search_chroma(client, collection_name: str, query: str, top_k: int):
    logging.info(f"Searching '{collection_name}' for: {query}")
    col = client.get_collection(name=collection_name, embedding_function=embedding_fn)
    return col.query(query_texts=[query], n_results=top_k)

# -------------------- Keyword Extraction --------------------

def extract_keywords(nl_query: str):
    tokens = re.findall(r"\b\w+\b", nl_query.lower())
    kws = [t for t in tokens if t not in STOPWORDS]
    logging.info(f"Extracted keywords: {kws}")
    return kws

# -------------------- Node Resolution (Exact Match) --------------------

def lookup_nodes_by_keyword(client, terms):
    logging.info(f"Exact node lookup for keywords: {terms}")
    col = client.get_collection(name="nodes_info", embedding_function=embedding_fn)
    nodes = []
    resolved = set()
    for term in terms:
        try:
            res = col.get(where={"name": term})
            if res and res.get("metadatas") and res["metadatas"][0]:
                m = res["metadatas"][0]
                if m.get("category", "") not in EXCLUDED_CATEGORIES:
                    nodes.append({
                        "documents": [[res["documents"][0]]],
                        "metadatas": [[m]]
                    })
                    resolved.add(term)
                    logging.info(f"Resolved keyword to node: '{term}' → {m.get('id')}")
        except Exception as e:
            logging.warning(f"Keyword lookup failed for '{term}': {e}")
    return nodes, resolved

# -------------------- Schema Fallback --------------------

def schema_lookup(client, terms):
    logging.info(f"Schema lookup for: {terms}")
    col = client.get_collection(name="yaml_schema", embedding_function=embedding_fn)
    schema = {"documents": [[]], "metadatas": [[]]}
    for term in terms:
        try:
            res = col.query(query_texts=[term], n_results=Config.TOP_K_SCHEMA)
            if res["documents"][0]:
                schema["documents"][0].extend(res["documents"][0])
                schema["metadatas"][0].extend(res["metadatas"][0])
                logging.info(f"Resolved '{term}' → schema hint")
        except Exception as e:
            logging.warning(f"Schema lookup failed for '{term}': {e}")
    return schema

# -------------------- Prompt & LLM --------------------

def construct_prompt(nl_query, examples, schema, nodes):
    logging.info("Constructing prompt")
    prompt = (
        "You are an expert at converting natural language queries into TRAPI JSON format.\n"
        "Output ONLY the JSON for query_graph. No extra text.\n\n"
    )
    # examples
    if examples and examples.get("documents"):
        for i, nl in enumerate(examples["documents"][0]):
            tr = examples["metadatas"][0][i].get("trapi_query", "N/A")
            prompt += f"Example {i+1}: NL: {nl} → TRAPI: {tr}\n"
    # schema hints
    if schema["documents"][0]:
        prompt += "\nSchema Knowledge:\n"
        for m, d in zip(schema["metadatas"][0], schema["documents"][0]):
            prompt += f"- {m.get('key','?')}: {d}\n"
        prompt += "\n"
    # node context
    if nodes:
        prompt += "Node Context:\n"
        for n in nodes:
            m = n["metadatas"][0][0]
            prompt += f"- {m.get('name')} ({m.get('id')}, {m.get('category')}): {n['documents'][0][0]}\n"
        prompt += "\n"
    prompt += f"Convert query to TRAPI JSON:\n{nl_query}\n\n```json\n"
    return prompt


def finalize_prompt(prompt: str) -> str:
    return prompt + "```"

# -------------------- JSON Extraction --------------------

def extract_trapi_query(resp: str):
    logging.info("Extracting JSON from LLM output")
    m = re.search(r'```json\s*(\{.*?\})\s*```', resp, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    logging.error("No JSON fence found; trying naked JSON")
    m2 = re.search(r'(\{\s*"nodes".*?\})', resp, re.DOTALL)
    if m2:
        return json.loads(m2.group(1))
    return None

# -------------------- Main Pipeline --------------------

def main():
    nl_query = "What biological processes are related to GFAP?"
    logging.info(f"Processing query: {nl_query}")

    # Initialize clients
    db_ex     = init_client(Config.EXAMPLES_DB)
    db_schema = init_client(Config.SCHEMA_DB)

    # 1) Retrieve examples
    examples = search_chroma(db_ex, "nl_to_trapi", nl_query, Config.TOP_K_EXAMPLES)

    # 2) Extract keywords
    kws = extract_keywords(nl_query)

    # 3) Exact-match node lookup on keywords
    nodes, resolved = lookup_nodes_by_keyword(db_schema, kws)

    # 4) Schema fallback for unresolved
    unresolved = [k for k in kws if k not in resolved]
    logging.info(f"Schema fallback terms: {unresolved}")
    schema = schema_lookup(db_schema, unresolved)

    # 5) Build & send prompt
    prompt = construct_prompt(nl_query, examples, schema, nodes)
    prompt = finalize_prompt(prompt)
    logging.debug(f"Prompt:\n{prompt}")

    # 6) Greedy LLM call
    llm = pipeline(
        "text-generation",
        model=Config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        return_full_text=False
    )
    resp = llm(prompt, do_sample=False, temperature=0.0, max_new_tokens=Config.MAX_NEW_TOKENS)[0]["generated_text"]
    logging.info("LLM response received")

    # 7) Extract & validate TRAPI
    trapi = extract_trapi_query(resp)
    if trapi and isinstance(trapi, dict) and "nodes" in trapi and "edges" in trapi:
        print(json.dumps(trapi, indent=2))
        logging.info("✅ TRAPI generated successfully")
    else:
        logging.error("❌ Failed to generate valid TRAPI JSON")
        print("Check logs for details.")

if __name__ == "__main__":
    main()
