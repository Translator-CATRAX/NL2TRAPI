import chromadb
import json
import yaml
import torch
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize GPU-optimized embedding function
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    device="cuda"
)

def initialize_chroma_db():
    return chromadb.PersistentClient(path="/scratch/vmm5481/NL2TRAPI/chroma_db")

# --- Batch Processing Functions ---

def process_node_batch(batch):
    docs, metas, ids = [], [], []
    for node in batch:
        node_id = str(node.get("id"))
        if not node_id:
            continue
        name = str(node.get("name", ""))
        category = str(node.get("category", ""))
        description = str(node.get("description", "")).strip()
        all_names = json.dumps(node.get("all_names", []))
        all_categories = json.dumps(node.get("all_categories", []))

        doc = f"{name} | {node_id} | {category}: {description}"
        meta = {
            "id": node_id,
            "name": name,
            "category": category,
            "description": description,
            "all_names": all_names,
            "all_categories": all_categories
        }

        docs.append(doc)
        metas.append(meta)
        ids.append(node_id)
    torch.cuda.empty_cache()  # Free GPU memory after batch
    return docs, metas, ids

def process_yaml_batch(batch, section):
    docs, metas, ids = [], [], []
    for key, value in batch:
        description = str(value.get("description", "")).strip()
        aliases = value.get("aliases", [])
        alias_text = f"Aliases: {', '.join(aliases)}" if aliases else ""
        slots = json.dumps(value.get("slots", [])) if 'slots' in value else ""
        notes = value.get("notes", [])
        notes_text = ' '.join(notes) if isinstance(notes, list) else str(notes).strip()

        full_text = f"{key} | {section}\nDescription: {description}"
        if notes_text:
            full_text += f"\nNotes: {notes_text}"
        if alias_text:
            full_text += f"\n{alias_text}"

        category = "qualifier" if value.get("is_a") == "qualifier" else section

        meta = {
            "key": key,
            "description": description,
            "aliases": ', '.join(aliases),
            "slots": slots,
            "category": category
        }

        docs.append(full_text)
        metas.append(meta)
        ids.append(key)
    return docs, metas, ids

# --- Parallel Loading Utility ---

def parallel_load(data, process_fn, collection, batch_size=5000, max_workers=6, desc="Loading"):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in tqdm(range(0, len(data), batch_size), desc=desc):
            batch = data[i:i+batch_size]
            futures.append(executor.submit(process_fn, batch))
        for batch_num, future in enumerate(tqdm(futures, desc="Committing")):
            try:
                docs, metas, ids = future.result()
                if ids:
                    collection.add(documents=docs, metadatas=metas, ids=ids)
            except Exception as e:
                print(f"Batch {batch_num} failed: {str(e)[:200]}")

# --- Loaders ---

def load_yaml_schema(chroma_client, yaml_file):
    print("Loading YAML schema...")
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    collection = chroma_client.get_or_create_collection(name="yaml_schema", embedding_function=embedding_function)

    if "classes" in data:
        class_data = list(data["classes"].items())
        parallel_load(class_data, partial(process_yaml_batch, section="class"), collection, desc="Classes")

    if "slots" in data:
        slot_data = list(data["slots"].items())
        parallel_load(slot_data, partial(process_yaml_batch, section="predicate"), collection, desc="Slots")

def load_nodes(chroma_client, nodes_file):
    print("Loading nodes...")
    with open(nodes_file, 'r') as f:
        data = json.load(f)

    collection = chroma_client.get_or_create_collection(name="nodes_info", embedding_function=embedding_function)
    parallel_load(data, process_node_batch, collection, batch_size=5000, max_workers=8, desc="Nodes")

# --- Verification ---

def verify_node(chroma_client, test_id="NCBIGene:2670"):
    print(f"\nVerifying existence of node {test_id}")
    collection = chroma_client.get_collection("nodes_info", embedding_function=embedding_function)
    result = collection.get(ids=[test_id])
    if result['ids']:
        print("Returned:", result['ids'])
        print("Metadata:", result['metadatas'][0])
        print("Document:", result['documents'][0])
    else:
        print("Node not found.")

    print(f"\nTotal nodes in collection: {collection.count()}")

# --- Main Entry Point ---

def main():
    client = initialize_chroma_db()
    load_yaml_schema(client, "/scratch/vmm5481/NL2TRAPI/biolink-model.yaml")
    load_nodes(client, "/scratch/vmm5481/NL2TRAPI/nodes.json")
    verify_node(client)

if __name__ == "__main__":
    main()

