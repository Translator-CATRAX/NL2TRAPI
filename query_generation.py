import re
import json
import torch
import logging
from datetime import datetime
import chromadb
from transformers import pipeline
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Enhanced logging setup
log_filename = f"nl2trapi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logging.info("Script started with enhanced logging")

# Constants remain unchanged
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
    start_time = datetime.now()
    logging.info(f"Initializing ChromaDB from {path}")
    client = chromadb.PersistentClient(path=path)
    logging.info(f"ChromaDB initialized in {(datetime.now() - start_time).total_seconds():.2f}s")
    return client

def search_chroma(chroma_client, collection_name, query, top_k=5):
    try:
        start_time = datetime.now()
        logging.info(f"Searching collection '{collection_name}' for: {query}")
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        result = collection.query(query_texts=[query], n_results=top_k)
        logging.info(f"Search completed in {(datetime.now() - start_time).total_seconds():.2f}s")
        
        # Debug logging of results
        if result and result["documents"]:
            logging.debug(f"Top result: {result['documents'][0][0][:100]}...")
        return result
    except Exception as e:
        logging.error(f"Search error in '{collection_name}': {e}")
        return None

def exact_node_match(chroma_client, nl_query):
    try:
        logging.info(f"Attempting exact node match for: {nl_query}")
        collection = chroma_client.get_collection(name="nodes_info", embedding_function=embedding_function)
        results = collection.get(where={"name": nl_query})
        if results and results.get("metadatas") and results["metadatas"][0]:
            logging.info(f"Exact match found for node: {nl_query}")
            logging.debug(f"Node details: {json.dumps(results['metadatas'][0][0], indent=2)}")
            return [{
                "documents": [[results["documents"][0]]],
                "metadatas": [[results["metadatas"][0]]]
            }]
        logging.info(f"No exact match found for: {nl_query}")
        return []
    except Exception as e:
        logging.warning(f"Exact match failed for node '{nl_query}': {e}")
        return []

def extract_keywords(nl_query):
    start_time = datetime.now()
    tokens = re.findall(r"\b\w+\b", nl_query.lower())
    keywords = [t for t in tokens if t not in STOPWORDS]
    logging.info(f"Extracted {len(keywords)} keywords in {(datetime.now() - start_time).total_seconds():.4f}s: {keywords}")
    return keywords

def construct_prompt(nl_query, examples, schema, resolved_nodes):
    # Preserve the professor's exact prompt content
    base_prompt = (
        "You are an expert at converting natural language queries into TRAPI JSON format. Recall that TRAPI ("
        "Translator Application Programming Interface) is the API format used by the Biomedical Data "
        "Translator Consortium. You will be provided a natural language query and will need to convert this "
        "to TRAPI. The TRAPI format is essentially a graph pattern representation expressed in JSON form. Some "
        "nodes are 'pinned' in that they correspond to specific biomedical entities and have an associated "
        "identifier/CURIE _and_ a category or categories. Some nodes are 'free' in that they have one (or more) "
        "categories assigned to them. "
        "Similarly, edges have predicates on them expressing the relationship type. The TRAPI query is then a "
        "pattern used to query knowledge graphs to satisfy the graph query (ensuring that nodes match the "
        "specified CURIE in 'pinned' nodes or the specified category of 'free' nodes, as well as edge "
        "relationships).\n"
        "So for this task, you will need to: 1. Identify the topology of the graph query (how many nodes, "
        "how many edges, and how they are connected), 2. Identify the entities and relationships in the natural"
        "language queries, and 3. determine which entities go with which nodes and edges. Note that we are only "
        "interested in the value of the `query_graph` field in the TRAPI JSON. The rest of the TRAPI JSON is "
        "not relevant for this task. "
        "We provide you with examples of natural language queries and their corresponding TRAPI JSON format. "
        "We also provide some context about nodes, their categories, and identifiers (CURIEs), as well as schema "
        "knowledge. Note: this context information is not always available and not every entry is relevant, but it"
        " can help you understand the nodes, which categories to assign, and which identifiers to use. DO NOT use "
        "CURIEs outside of this provided context. If you do not find a relevant CURIE in the context, you can "
        "return a question mark in it's place."
    )

    prompt_parts = [base_prompt]

    # Add examples section if available
    if examples and examples["documents"][0]:
        logging.info(f"Including {len(examples['documents'][0])} examples in prompt")
        prompt_parts.append("\n\n=== Examples ===\n")
        for i in range(len(examples["documents"][0])):
            nl = examples["documents"][0][i]
            trapi = examples["metadatas"][0][i].get("trapi_query", "N/A")
            prompt_parts.append(f"Example {i+1}:\nNL Query: {nl}\nTRAPI Query: {json.dumps(trapi, indent=2)}\n")

    # Add schema knowledge if available
    if schema and schema["documents"][0]:
        logging.info(f"Including {len(schema['documents'][0])} schema items in prompt")
        prompt_parts.append("\n=== Schema Knowledge ===\n")
        for i in range(len(schema["documents"][0])):
            k = schema["metadatas"][0][i].get("key", "Unknown")
            d = schema["documents"][0][i]
            prompt_parts.append(f"- {k}: {d}\n")

    # Add node context if available
    if resolved_nodes:
        logging.info(f"Including {len(resolved_nodes)} resolved nodes in prompt")
        prompt_parts.append("\n=== Node Context ===\n")
        for result in resolved_nodes:
            meta = result["metadatas"][0][0]
            name = meta.get("name", "")
            node_id = meta.get("id", "")
            cat = meta.get("category", "")
            desc = result["documents"][0][0]
            prompt_parts.append(f"{name} | {node_id} | {cat}: {desc}\n")

    # Add the final query section with clear instructions
    prompt_parts.append(
        f"\n=== Convert This Query ===\n"
        f"NL Query: {nl_query}\n\n"
        f"Generate exactly one TRAPI JSON response for this query. "
        f"Return only the complete JSON object wrapped in ```json markers like this:\n"
        f"```json\n"
        f"{{your TRAPI query_graph JSON here}}\n"
        f"```"
    )
    
    full_prompt = "\n".join(prompt_parts)
    logging.debug(f"Constructed prompt (first 500 chars):\n{full_prompt[:500]}...")
    return full_prompt

def initialize_mistral():
    logging.info("Loading Mistral model...")
    start_time = datetime.now()
    mistral_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        return_full_text=False
    )
    logging.info(f"Mistral loaded in {(datetime.now() - start_time).total_seconds():.2f}s")
    return mistral_pipeline

def extract_trapi_query(llm_response):
    try:
        logging.debug(f"Attempting to extract JSON from LLM response:\n{llm_response[:200]}...")
        
        # First try to find JSON between markers
        json_match = re.search(r'```json\n(.*?)\n```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            logging.debug(f"Found JSON between markers:\n{json_str}")
            return json.loads(json_str)
        
        # Fallback: try to find any complete JSON object
        json_str = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_str:
            logging.debug(f"Found JSON without markers:\n{json_str.group(0)}")
            return json.loads(json_str.group(0))
            
        logging.error("No JSON found in LLM response")
        return None
    except Exception as e:
        logging.error(f"Failed to parse TRAPI JSON: {e}\nResponse: {llm_response[:200]}...")
        return None

def generate_trapi(nl_query, examples, schema, nodes, mistral):
    """Generate TRAPI JSON with enhanced validation and debugging"""
    
    # Construct and display prompt
    prompt = construct_prompt(nl_query, examples, schema, nodes)
    print("\n=== LLM INPUT PROMPT ===\n")
    print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))  # Truncate long prompts
    logging.info("LLM Prompt (%d chars):\n%.1000s...", len(prompt), prompt)  # Log first 1000 chars

    try:
        # Generate response with timing
        start_time = datetime.now()
        response = mistral(
            prompt,
            do_sample=True,
            top_k=10,
            top_p=0.9,
            temperature=0.7,
            max_new_tokens=512,
            num_return_sequences=1,
            eos_token_id=mistral.tokenizer.eos_token_id
        )[0]["generated_text"]
        gen_time = (datetime.now() - start_time).total_seconds()
        
        # Display and log response
        print("\n=== LLM RAW OUTPUT ===\n")
        print(response)
        logging.info("LLM Response (%.2fs, %d tokens):\n%s", 
                    gen_time, len(response.split()), response)

        # Extract and validate
        trapi_query = extract_trapi_query(response)
        if not trapi_query:
            msg = "No valid JSON found in LLM response"
            print(f"\n⚠️ {msg}")
            logging.warning("%s. Response start:\n%.500s", msg, response)
            return None
            
        # Validate structure
        if not validate_trapi_structure(trapi_query):
            msg = "TRAPI structure validation failed"
            print(f"\n⚠️ {msg}")
            logging.warning("%s: %s", msg, json.dumps(trapi_query, indent=2))
            return None
            
        return trapi_query
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        print(f"\n❌ {error_msg}")
        logging.error(error_msg, exc_info=True)
        return None

def validate_trapi_structure(trapi):
    """Validate basic TRAPI structure"""
    if not isinstance(trapi, dict):
        return False
    if "nodes" not in trapi or "edges" not in trapi:
        return False
    return True

def main():
    nl_query = "What biological processes are related to GFAP"
    logging.info(f"Starting processing for query: '{nl_query}'")
    
    # Initialize databases
    db_examples = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db")
    db_nodeschema = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/chroma_db")

    # Retrieve examples
    examples = search_chroma(db_examples, "nl_to_trapi", nl_query, top_k=5)
    if examples:
        logging.info(f"Retrieved {len(examples['documents'][0])} examples")
    
    # Node resolution
    resolved_nodes = exact_node_match(db_nodeschema, nl_query)
    keywords = extract_keywords(nl_query)
    
    node_collection = db_nodeschema.get_collection(name="nodes_info", embedding_function=embedding_function)
    resolved_node_terms = set()
    if resolved_nodes:
        for result in resolved_nodes:
            meta = result["metadatas"][0][0]
            resolved_node_terms.add(meta.get("name", "").lower())

    # Resolve remaining keywords
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
                    logging.info(f"Resolved node term: {kw} (Category: {category})")
        except Exception as e:
            logging.warning(f"Failed to resolve node term: {kw} - {e}")
            continue

    # Schema fallback for unresolved terms
    unresolved_terms = [kw for kw in keywords if kw not in resolved_node_terms]
    schema = {"documents": [[]], "metadatas": [[]]}
    if unresolved_terms:
        logging.info(f"Attempting schema resolution for: {unresolved_terms}")
        schema_collection = db_nodeschema.get_collection(name="yaml_schema", embedding_function=embedding_function)
        for term in unresolved_terms:
            try:
                result = schema_collection.query(query_texts=[term], n_results=2)
                if result and result["documents"][0]:
                    schema["documents"][0].extend(result["documents"][0])
                    schema["metadatas"][0].extend(result["metadatas"][0])
                    logging.info(f"Resolved schema term: {term}")
            except Exception as e:
                logging.warning(f"Schema resolution failed for {term}: {e}")
                continue

    # Generate TRAPI query
    mistral = initialize_mistral()
    trapi_query = generate_trapi(nl_query, examples, schema, resolved_nodes, mistral)

    if trapi_query:
        print("\n✅ TRAPI Generation Successful")
        print("Final TRAPI Query Graph:")
        print(json.dumps(trapi_query, indent=2))
        logging.info("Successfully generated TRAPI:\n%s", 
                    json.dumps(trapi_query, indent=2))
    else:
        print("\n❌ TRAPI Generation Failed")
        print("Debugging information available:")
        print("- Full prompt input shown above")
        print("- Raw LLM output shown above")
        print("- Check log file for details:", log_filename)
        logging.error("TRAPI generation failed for query: %s", nl_query)

if __name__ == "__main__":
    main()
