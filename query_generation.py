import re
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
import chromadb
from transformers import pipeline
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configuration
class Config:
    examples_db_path = "/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db"
    schema_db_path   = "/scratch/vmm5481/NL2TRAPI/chroma_db"
    model_name       = "mistralai/Mistral-7B-Instruct-v0.1"
    embedding_model  = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    device           = "cuda"
    top_k_examples   = 5
    max_new_tokens   = 512
    temperature      = 0.7
    top_k_nodes      = 5
    log_dir          = "experiment_logs"

# Enhanced logging setup
def setup_logging(log_dir: str) -> str:
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"nl2trapi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)

    return str(log_file)

# Initialize logger and embedding function
data_log = setup_logging(Config.log_dir)
logging.info(f"Experiment log: {data_log}")
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=Config.embedding_model,
    device=Config.device
)

# Core utilities
STOPWORDS = {"what","which","is","are","a","an","the","in","of","to","for","and","on","with","by","how","at","that"}
EXCLUDED_CATEGORIES = {
    "biolink:ClinicalAttribute", "biolink:InformationContentEntity",
    "biolink:Event", "biolink:ExposureEvent", "biolink:GeographicLocation",
    "biolink:PopulationOfIndividualOrganisms", "biolink:Publication",
    "biolink:RetrievalSource", "biolink:StudyPopulation", "biolink:SubjectOfInvestigation"
}

def initialize_chroma_db(path: str) -> chromadb.PersistentClient:
    logging.info(f"Initializing ChromaDB at {path}")
    client = chromadb.PersistentClient(path=path)
    logging.info("ChromaDB initialized")
    return client


def search_chroma(client, collection_name: str, query: str, top_k: int=Config.top_k_examples):
    logging.info(f"Searching '{collection_name}' for: {query}")
    col = client.get_collection(name=collection_name, embedding_function=embedding_fn)
    result = col.query(query_texts=[query], n_results=top_k)
    return result


def exact_node_match(client, query: str):
    logging.info(f"Exact node match for: {query}")
    col = client.get_collection(name="nodes_info", embedding_function=embedding_fn)
    res = col.get(where={"name": query})
    if res and res.get("metadatas") and res["metadatas"][0]:
        return [{
            "documents": [[res["documents"][0]]],
            "metadatas": [[res["metadatas"][0]]]
        }]
    return []


def extract_keywords(query: str):
    tokens = re.findall(r"\b\w+\b", query.lower())
    kws = [t for t in tokens if t not in STOPWORDS]
    logging.info(f"Keywords extracted: {kws}")
    return kws


def construct_prompt(nl_query, examples, schema, nodes):
    # (use original detailed prompt text here...)
    base = (
        "You are an expert converting NL queries to TRAPI JSON.\n"
        "Output ONLY the JSON for query_graph, no additional text.\n"
    )
    # append examples, schema, nodes, and the final conversion instruction
    return base + f"Convert: {nl_query}\nOutput JSON:"


def initialize_mistral():
    logging.info("Loading model...")
    return pipeline(
        "text-generation",
        model=Config.model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        return_full_text=False
    )


def extract_trapi_query(resp: str):
    # Try fenced JSON
    m = re.search(r'```json\s*(\{.*?\})\s*```', resp, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Fallback: naked JSON
    m2 = re.search(r'(\{\s*"nodes".*?\})', resp, re.DOTALL)
    if m2:
        return json.loads(m2.group(1))
    return None


def generate_trapi(nl_query, examples, schema, nodes, model):
    prompt = construct_prompt(nl_query, examples, schema, nodes)
    logging.debug(f"Prompt:\n{prompt}")
    out = model(prompt,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=Config.max_new_tokens)[0]["generated_text"]
    logging.debug(f"Raw LLM output:\n{out}")
    return extract_trapi_query(out)


def validate_trapi(trapi: dict) -> bool:
    if not isinstance(trapi, dict): return False
    return "nodes" in trapi and "edges" in trapi


def main():
    nl_query = "What biological processes are related to GFAP"
    logging.info(f"Processing: {nl_query}")

    db_ex = initialize_chroma_db(Config.examples_db_path)
    db_sc = initialize_chroma_db(Config.schema_db_path)

    examples = search_chroma(db_ex, "nl_to_trapi", nl_query)
    nodes = exact_node_match(db_sc, nl_query)

    kws = extract_keywords(nl_query)
    col = db_sc.get_collection(name="nodes_info", embedding_function=embedding_fn)
    for kw in kws:
        if any(n["metadatas"][0][0]["name"].lower() == kw for n in nodes):
            continue
        try:
            r = col.get(where={"name": kw})
            if r and r.get("metadatas") and r["metadatas"][0]:
                cat = r["metadatas"][0].get("category", "")
                if cat not in EXCLUDED_CATEGORIES:
                    nodes.append({
                        "documents": [[r["documents"][0]]],
                        "metadatas": [[r["metadatas"][0]]]
                    })
        except Exception:
            continue

    unresolved = [kw for kw in kws if not any(n["metadatas"][0][0]["name"].lower()==kw for n in nodes)]
    schema = {"documents": [[]], "metadatas": [[]]}
    if unresolved:
        sc_col = db_sc.get_collection(name="yaml_schema", embedding_function=embedding_fn)
        for term in unresolved:
            try:
                res = sc_col.query(query_texts=[term], n_results=2)
                if res and res["documents"][0]:
                    schema["documents"][0].extend(res["documents"][0])
                    schema["metadatas"][0].extend(res["metadatas"][0])
            except Exception:
                continue

    model = initialize_mistral()
    trapi = generate_trapi(nl_query, examples, schema, nodes, model)

    if trapi and validate_trapi(trapi):
        print(json.dumps(trapi, indent=2))
        logging.info("✅ TRAPI generated successfully")
    else:
        logging.error("❌ Failed to generate valid TRAPI")
        print("Check logs for debugging details.")

if __name__ == "__main__":
    main()
