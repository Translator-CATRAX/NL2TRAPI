import chromadb
import json
import torch
from transformers import pipeline
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# GPU-accelerated embedding
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device="cuda"
)

def initialize_chroma_db(path):
    return chromadb.PersistentClient(path=path)

def search_chroma(chroma_client, collection_name, query, top_k=5):
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
        return collection.query(query_texts=[query], n_results=top_k)
    except Exception as e:
        print(f"Search error in '{collection_name}': {e}")
        return None

def construct_prompt(nl_query, retrieved_examples, retrieved_schema, retrieved_nodes):
    example_text, schema_text, node_text = "", "", ""

    if retrieved_examples:
        for i in range(len(retrieved_examples["documents"][0])):
            nl = retrieved_examples["documents"][0][i]
            trapi = retrieved_examples["metadatas"][0][i].get("trapi_query", "N/A")
            example_text += f"Example {i+1}:\nNL Query: {nl}\nTRAPI Query: {trapi}\n\n"

    if retrieved_schema:
        for i in range(len(retrieved_schema["documents"][0])):
            schema_key = retrieved_schema["metadatas"][0][i].get("key", "Unknown")
            schema_text += f"- {schema_key}: {retrieved_schema['documents'][0][i]}\n"

    if retrieved_nodes:
        for i in range(len(retrieved_nodes["documents"][0])):
            meta = retrieved_nodes["metadatas"][0][i]
            name = meta.get("name", "")
            category = meta.get("category", "")
            node_id = meta.get("id", "")
            node_text += f"{name} | {node_id} | {category}: {retrieved_nodes['documents'][0][i]}\n"

    return f"""You are an expert at converting natural language questions into structured TRAPI JSON.

**Guidelines:**
- Use CURIE identifiers from the node metadata
- Use Biolink predicates from schema
- Keep your output strict JSON — no commentary

Examples:
{example_text}

Schema Knowledge:
{schema_text}

Node Context:
{node_text}

Natural Language Query: {nl_query}

TRAPI JSON:
"""

def initialize_mistral():
    print("Loading Mistral...")
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

def extract_trapi_query(llm_output):
    try:
        start = llm_output.find("TRAPI JSON:")
        if start != -1:
            json_part = llm_output[start + len("TRAPI JSON:"):].strip()
            return json.loads(json_part)
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
    except Exception as e:
        print("Unexpected Error:", e)
    return None

def validate_trapi(trapi_query, schema_collection, nodes_collection):
    for node in trapi_query.get("nodes", {}).values():
        for node_id in node.get("ids", []):
            result = nodes_collection.get(ids=[node_id])
            if not result['ids']:
                print(f"Invalid node ID: {node_id}")
                return False

    for edge in trapi_query.get("edges", {}).values():
        for predicate in edge.get("predicates", []):
            result = schema_collection.get(ids=[predicate])
            if not result['ids']:
                print(f"Invalid predicate: {predicate}")
                return False
    return True

def generate_trapi(nl_query, examples, schema, nodes, mistral):
    prompt = construct_prompt(nl_query, examples, schema, nodes)
    output = mistral(
        prompt,
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        top_k=10,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=mistral.tokenizer.eos_token_id
    )[0]["generated_text"]
    print("\n=== Raw LLM Output ===\n", output)
    return extract_trapi_query(output)

def main():
    query = "What biological processes are related to GFAP"

    db1 = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db")
    db2 = initialize_chroma_db("/scratch/vmm5481/NL2TRAPI/chroma_db")

    examples = search_chroma(db1, "nl_to_trapi", query, top_k=5)
    schema = search_chroma(db2, "yaml_schema", query, top_k=5)
    nodes = search_chroma(db2, "nodes_info", query, top_k=5)

    mistral = initialize_mistral()

    trapi = generate_trapi(query, examples, schema, nodes, mistral)

    if trapi:
        schema_col = db2.get_collection("yaml_schema", embedding_function=embedding_function)
        node_col = db2.get_collection("nodes_info", embedding_function=embedding_function)

        if validate_trapi(trapi, schema_col, node_col):
            print("\n✅ Final TRAPI Query:")
            print(json.dumps(trapi, indent=4))
        else:
            print("❌ Generated TRAPI query is invalid.")
    else:
        print("❌ LLM failed to generate a valid TRAPI query.")

if __name__ == "__main__":
    main()
