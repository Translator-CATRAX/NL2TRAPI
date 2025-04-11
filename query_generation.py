
import chromadb
from sentence_transformers import SentenceTransformer
import json
from transformers import pipeline
import torch

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma DB Instances
def initialize_chroma_db(persist_dir):
    return chromadb.PersistentClient(path=persist_dir)

# Perform Semantic Search on Chroma DB
def search_chroma(chroma_client, collection_name, query, top_k=3):
    try:
        collection = chroma_client.get_collection(name=collection_name)
        results = collection.query(query_texts=[query], n_results=top_k)
        print(f"Search results: {results}")
        return results
    except Exception as e:
        print(f"Error searching collection {collection_name}: {e}")
        return None

# Construct Prompt Dynamically
def construct_prompt(nl_query, retrieved_examples, retrieved_schema):
    # Format retrieved examples
    example_text = ""
    if retrieved_examples and 'documents' in retrieved_examples and retrieved_examples['documents']:
        for i in range(len(retrieved_examples['documents'])):
            example_text += f"Example {i + 1}:\n"
            example_text += f"NL Query: {retrieved_examples['documents'][i]}\n"
            # Accessing the correct index of the 'metadatas' list, which is nested in another list
            trapi_query = retrieved_examples['metadatas'][i][0].get('trapi_query', 'No TRAPI query available')
            example_text += f"TRAPI Query: {trapi_query}\n\n"

    # Format retrieved schema
    schema_text = ""
    if retrieved_schema and 'documents' in retrieved_schema and retrieved_schema['documents']:
        for i in range(len(retrieved_schema['documents'])):
            # Accessing the first element of the list in 'metadatas' to retrieve the schema 'key'
            schema_key = retrieved_schema['metadatas'][i][0].get('key', 'No key available')
            schema_text += f"- {schema_key}: {retrieved_schema['documents'][i]}\n"

    # Construct prompt
    prompt = f"""
    You are an expert at converting natural language queries into TRAPI JSON format.

    Here is an example of how similar queries were converted into TRAPI queries:

    {example_text}

    Here is some relevant schema information for your reference:
    {schema_text}

    Now, generate a TRAPI JSON for this natural language query:
    {nl_query}

    TRAPI JSON:
    """

    return prompt




# Initialize Mistral Model
def initialize_mistral():
    print("Initializing Mistral Model...")
    generator = pipeline('text-generation', model='mistralai/Mistral-7B-Instruct-v0.1', torch_dtype=torch.float16, device_map="auto")
    print("Mistral Model Initialized.")
    return generator

# Generate TRAPI Query Using Retrieved Data and LLM (Mistral)
def generate_trapi(nl_query, retrieved_examples, retrieved_schema, generator):
    prompt = construct_prompt(nl_query, retrieved_examples, retrieved_schema)
    
    sequences = generator(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=generator.tokenizer.eos_token_id,
        max_length=1024,
    )
    
    llm_response = sequences[0]['generated_text']
    print(f"LLM Response: {llm_response}")
    
    trapi_query = extract_trapi_query(llm_response)
    return trapi_query

# Extract TRAPI Query from LLM Response
def extract_trapi_query(llm_response):
    try:
        lines = llm_response.splitlines()
        start_index = None
        for i, line in enumerate(lines):
            if line.strip() == "TRAPI JSON:":
                start_index = i + 1
                break

        if start_index is not None:
            trapi_query_str = "\n".join(lines[start_index:])
            try:
                trapi_query = json.loads(trapi_query_str)
                return trapi_query
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return None
        else:
            print("TRAPI JSON not found in response.")
            return None
    except Exception as e:
        print(f"Error extracting TRAPI query: {e}")
        return None

# Main Function to Run the Pipeline
def main():
    # Initialize Chroma DB clients for DB1 and DB2
    nl_to_trapi_db = initialize_chroma_db("/data/vmm5481/nl_to_trapi_db") # set it to chromadb path with nl2trapi examples
    yaml_schema_db = initialize_chroma_db("/data/vmm5481/chroma_db") # set it to chromadb path with schema information

    # Initialize Mistral model
    generator = initialize_mistral()

    # Input Natural Language Query
    nl_query = "What biological processes are related to GFAP" # we can try out with any relevant NL query here

    # Retrieve relevant NL-to-TRAPI examples from DB1
    retrieved_examples = search_chroma(nl_to_trapi_db, collection_name="nl_to_trapi", query=nl_query)

    # Retrieve relevant schema information from DB2
    retrieved_schema = search_chroma(yaml_schema_db, collection_name="yaml_schema", query=nl_query)

    # Check if results are empty
    if not retrieved_examples or not retrieved_schema:
        print("No relevant data found. Exiting.")
        return

    # Generate TRAPI Query using LLM (Mistral)
    trapi_query = generate_trapi(nl_query, retrieved_examples, retrieved_schema, generator)

    # Print the generated TRAPI query
    if trapi_query:
        print(json.dumps(trapi_query, indent=4))
    else:
        print("Failed to generate a valid TRAPI query.")

if __name__ == "__main__":
    main()



