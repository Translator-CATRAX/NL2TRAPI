# build_exact_index.py — fixed version
import json
import pickle
import re

def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()

def build_index(json_path):
    with open(json_path, "r") as f:
        nodes = json.load(f)

    index = {}
    skipped = 0
    for node in nodes:
        name = node.get("name")
        if not isinstance(name, str) or not name.strip():
            skipped += 1
            continue

        key = normalize(name)
        if key not in index:
            index[key] = {
                "id": node.get("id"),
                "name": name,
                "category": node.get("category", ""),
                "description": node.get("description", "")
            }

    print(f"✅ Built index with {len(index)} entries (skipped {skipped} invalid nodes)")
    return index

if __name__ == "__main__":
    index = build_index("/scratch/vmm5481/NL2TRAPI/nodes.json")
    with open("exact_index.pkl", "wb") as f:
        pickle.dump(index, f)
    print("✅ Saved exact_name_index → exact_index.pkl")
