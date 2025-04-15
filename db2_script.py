import chromadb
from sentence_transformers import SentenceTransformer
import yaml
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding_function = SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )
# embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") #cpu

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cuda"  # This ensures embeddings are computed on the GPU
)

# Initialize Chroma DB (Persistent for saving data across runs)
def initialize_chroma_db():
    return chromadb.PersistentClient(path="/scratch/vmm5481/NL2TRAPI/chroma_db")  # Update path accordingly


# def initialize_chroma_db(persist_dir='/data/vmm5481/NL2TRAPI/chroma_db'):
#     embedding_function = SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )
#     return chromadb.PersistentClient(
#         path=persist_dir,
#         embedding_function=embedding_function
#     )

# Function to prevent duplicate IDs in ChromaDB
def add_to_collection_if_not_exists(collection, documents, metadatas, ids):
    existing_data = collection.get(ids=ids)
    existing_ids = set(existing_data['ids']) if 'ids' in existing_data and existing_data['ids'] else set()

    new_ids = set(ids)
    ids_to_add = list(new_ids - existing_ids)  # Remove existing IDs

    if ids_to_add:  # Add only new IDs
        filtered_documents = [documents[i] for i in range(len(ids)) if ids[i] in ids_to_add]
        filtered_metadatas = [metadatas[i] for i in range(len(ids)) if ids[i] in ids_to_add]
        
        collection.add(
            documents=filtered_documents,
            metadatas=filtered_metadatas,
            ids=ids_to_add
        )
    else:
        print(f"Skipping duplicate IDs: {existing_ids.intersection(new_ids)}")

# Load YAML Schema into Chroma DB
def load_yaml_to_chroma(yaml_file, chroma_client, collection_name):
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)

    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    # Extract Classes
    if 'classes' in yaml_data:
        for key, value in yaml_data['classes'].items():
            description = (value.get('description', '') or '').strip()
            if description:
                metadata = {
                    "key": key,
                    "description": description,
                    "slots": json.dumps(value.get('slots', [])),
                    "category": "class"
                }
                add_to_collection_if_not_exists(collection, [description], [metadata], [key])

    # Extract Predicates (Slots- Qualifiers and Predicates)
    # Extract Predicates (Slots) including Qualifiers and Notes
    if 'slots' in yaml_data:
        for key, value in yaml_data['slots'].items():
            # Get description
            description = (value.get('description', '') or '').strip()
            
            # Handle notes (list or string)
            notes = value.get('notes', '')
            if isinstance(notes, list):
                notes = ' '.join(notes).strip()  # Convert list to string and strip
            else:
                notes = (notes or '').strip()  # Ensure string and strip
            
            # Combine description and notes
            full_text = description
            if notes:
                full_text += f"\nNotes: {notes}"
            
            # Determine category
            category = "predicate"
            if 'is_a' in value and 'qualifier' in value['is_a']:
                category = "qualifier"
            
            # Create metadata
            metadata = {
                "key": key,
                "description": full_text,
                "category": category
            }
            
            # Add to collection
            add_to_collection_if_not_exists(
                collection=collection,
                documents=[full_text],
                metadatas=[metadata],
                ids=[key]
            )


    # if 'slots' in yaml_data:
    #     for key, value in yaml_data['slots'].items():
    #         description = (value.get('description', '') or '').strip()
    #         if description:
    #             category = "predicate"
    #             if 'is_a' in value and 'qualifier' in value['is_a']:
    #                 category = "qualifier"

    #             metadata = {
    #                 "key": key,
    #                 "description": description,
    #                 "category": category
    #             }
    #             add_to_collection_if_not_exists(collection, [description], [metadata], [key])

# Load Node Information from JSON File
def load_nodes_to_chroma(json_file, chroma_client, collection_name):
    with open(json_file, 'r') as file:
        nodes_data = json.load(file)

    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    for node in nodes_data:
        node_id = str(node.get('id'))
        if not node_id:  # Still skip nodes without a valid ID
            continue
        
        # Get description (allow empty strings)
        description = (node.get('description', '') or '').strip()

        # Create metadata even if description is empty
        metadata = {
            "id": node_id,
            "name": node.get('name', ''),
            "category": node.get('category', ''),
            # "all_names": json.dumps(node.get('all_names', [])),       #Uncomment if all_names is needed
            "description": description  # Will be empty string if no description
        }
        
        # Always add to collection regardless of description content
        add_to_collection_if_not_exists(
            collection=collection,
            documents=[description],  # Can be empty string
            metadatas=[metadata],
            ids=[node_id]
        )

# def load_nodes_to_chroma(json_file, chroma_client, collection_name):
#     with open(json_file, 'r') as file:
#         nodes_data = json.load(file)

#     collection = chroma_client.get_or_create_collection(name=collection_name)

#     for node in nodes_data:
#         node_id = str(node.get('id'))  
#         if not node_id:  # Skip nodes without a valid ID
#             continue
        
#         description = (node.get('description', '') or '').strip()
#         if not description:  # Skip empty descriptions
#             continue

#         metadata = {
#             "id": node_id,
#             "name": node.get('name', ''),
#             "category": node.get('category', ''),
#             "all_names": json.dumps(node.get('all_names', [])),  # Convert list to JSON string
#             "description": description
#         }
#         add_to_collection_if_not_exists(collection, [description], [metadata], [node_id])

# Main Function to Run the Pipeline
def main():
    # Initialize Chroma DB (Persistent)
    chroma_client = initialize_chroma_db()

    # File paths (update as per your server setup)
    yaml_file = "/scratch/vmm5481/NL2TRAPI/biolink-model.yaml"  # Replace this accordingly
    nodes_file = "/scratch/vmm5481/NL2TRAPI/nodes.json"         # Replace this accordingly

    # Load YAML schema into Chroma DB
    load_yaml_to_chroma(yaml_file, chroma_client, collection_name="yaml_schema")

    # Load Node Information into Chroma DB
    load_nodes_to_chroma(nodes_file, chroma_client, collection_name="nodes_info")

    print("Data successfully loaded into ChromaDB.")

if __name__ == "__main__":
    main()