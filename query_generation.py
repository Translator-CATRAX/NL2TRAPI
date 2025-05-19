# below code is for updated node along with 2-3 words not resolved + not considered goes for the schema mapping
import re
import json
import torch
import logging
from datetime import datetime
import chromadb
from transformers import pipeline
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Setup Logging
log_filename = f"nl2trapi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Script started.")

# Constants
STOPWORDS = {
    "what", "which", "is", "are", "a", "an", "the", "in", "of", "to", "for", "and", "on", "with", "by", "how", "at", "that"
}
EXCLUDED_CATEGORIES = {
    "biolink:ClinicalAttribute", "biolink:InformationContentEntity", "biolink:Event",
    "biolink:ExposureEvent", "biolink:GeographicLocation", "biolink:PopulationOfIndividualOrganisms",
    "biolink:Publication", "biolink:RetrievalSource", "biolink:StudyPopulation", "biolink:SubjectOfInvestigation"
}

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device="cuda"
)

def initialize_chroma_db(path):
    logging.info(f"Initializing ChromaDB from {path}")
    return chromadb.PersistentClient(path=path)

def search_chroma(chroma_client, collection_name, query, top_k=5):
    try:
        logging.info(f"Searching collection '{collection_name}' for: {query}")
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        return collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        logging.error(f"Search error in '{collection_name}': {e}")
        return None

def exact_node_match(chroma_client, nl_query):
    try:
        collection = chroma_client.get_collection(name="nodes_info", embedding_function=embedding_function)
        results = collection.get(where={"name": nl_query})
        if results and results.get("metadatas") and results["metadatas"][0]:
            logging.info(f"Exact match found for node: {nl_query}")
            return [{
                "documents": [[results["documents"][0]]],
                "metadatas": [[results["metadatas"][0]]]
            }]
    except Exception as e:
        logging.warning(f"Exact match failed for node '{nl_query}': {e}")
    return []

def extract_keywords(nl_query):
    tokens = re.findall(r"\b\w+\b", nl_query.lower())
    keywords = [t for t in tokens if t not in STOPWORDS]
    logging.info(f"Extracted keywords: {keywords}")
    return keywords

def construct_prompt(nl_query, examples, schema, resolved_nodes):
    prompt = "You are an expert at converting natural language queries into TRAPI JSON format.\n\n"
    if examples:
        for i in range(len(examples["documents"][0])):
            nl = examples["documents"][0][i]
            trapi = examples["metadatas"][0][i].get("trapi_query", "N/A")
            prompt += f"Example {i+1}:\nNL Query: {nl}\nTRAPI Query: {trapi}\n\n"
    if schema and schema["documents"][0]:
        prompt += "Schema Knowledge:\n"
        for i in range(len(schema["documents"][0])):
            k = schema["metadatas"][0][i].get("key", "Unknown")
            d = schema["documents"][0][i]
            prompt += f"- {k}: {d}\n"
    if resolved_nodes:
        prompt += "\nNode Context:\n"
        for result in resolved_nodes:
            meta = result["metadatas"][0][0]
            name = meta.get("name", "")
            node_id = meta.get("id", "")
            cat = meta.get("category", "")
            desc = result["documents"][0][0]
            prompt += f"{name} | {node_id} | {cat}: {desc}\n"
    prompt += f"\nNow, generate a TRAPI JSON for this natural language query:\n{nl_query}\n\nTRAPI JSON:"
    logging.info("Prompt constructed successfully.")
    return prompt

def initialize_mistral():
    logging.info("Loading Mistral model...")
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
        logging.error(f"Failed to parse TRAPI JSON: {e}")
    return None

def generate_trapi(nl_query, examples, schema, nodes, mistral):
    prompt = construct_prompt(nl_query, examples, schema, nodes)
    logging.info("Sending prompt to Mistral LLM...")
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
    logging.info("Received response from LLM.")
    print("=== LLM Output ===\n", response)
    return extract_trapi_query(response)

def main():
    nl_query = "What proteins does acetaminophen interact with?"
    logging.info(f"Processing query: {nl_query}")

    db_examples = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db")
    db_nodeschema = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/chroma_db")

    examples = search_chroma(db_examples, "nl_to_trapi", nl_query, top_k=5)
    resolved_nodes = exact_node_match(db_nodeschema, nl_query)
    keywords = extract_keywords(nl_query)

    node_collection = db_nodeschema.get_collection(name="nodes_info", embedding_function=embedding_function)
    resolved_node_terms = set()
    if resolved_nodes:
        for result in resolved_nodes:
            meta = result["metadatas"][0][0]
            resolved_node_terms.add(meta.get("name", "").lower())

    for kw in keywords:
        if kw in resolved_node_terms:
            continue
        try:
            results = node_collection.get(where={"name": kw})
            if results and results.get("metadatas") and results["metadatas"][0]:
                category = results["metadatas"][0].get("category", "")
                if category not in EXCLUDED_CATEGORIES:
                    resolved_nodes.append({
                        "documents": [[results["documents"][0]]],
                        "metadatas": [[results["metadatas"][0]]]
                    })
                    resolved_node_terms.add(kw)
                    logging.info(f"Resolved node term: {kw}")
        except Exception:
            logging.warning(f"Failed to resolve node term: {kw}")
            continue

    # Fallback to schema
    unresolved_terms = [kw for kw in keywords if kw not in resolved_node_terms]
    schema = {"documents": [[]], "metadatas": [[]]}
    if unresolved_terms:
        logging.info(f"Falling back to schema resolution for: {unresolved_terms}")
        schema_collection = db_nodeschema.get_collection(name="yaml_schema", embedding_function=embedding_function)
        for term in unresolved_terms:
            try:
                result = schema_collection.query(query_texts=[term], n_results=2)
                if result and result["documents"][0]:
                    schema["documents"][0].extend(result["documents"][0])
                    schema["metadatas"][0].extend(result["metadatas"][0])
                    logging.info(f"Schema term resolved: {term}")
            except Exception:
                logging.warning(f"Schema resolution failed for: {term}")
                continue

    mistral = initialize_mistral()
    trapi_query = generate_trapi(nl_query, examples, schema, resolved_nodes, mistral)

    if trapi_query:
        print("\n✅ Final TRAPI Query:")
        print(json.dumps(trapi_query, indent=4))
        logging.info("TRAPI query successfully generated.")
    else:
        print("\n❌ Failed to generate a valid TRAPI query.")
        logging.error("TRAPI generation failed.")

if __name__ == "__main__":
    main()
