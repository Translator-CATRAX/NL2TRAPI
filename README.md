# NL2TRAPI-LangGraph

Turn natural-language biomedical questions into TRAPI-compliant Query Graphs using a hybrid pipeline of SciSpaCy, LangGraph, Chroma embeddings and a local Node Normalization index.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Setup](#setup)
* [Data Preparation](#data-preparation)
* [Building Indices & Embeddings](#building-indices--embeddings)
* [Running the Agent](#running-the-agent)
* [Testing](#testing)
* [Configuration](#configuration)
* [License](#license)

---

## Features

* **Entity Recognition** with SciSpaCy (`en_ner_bionlp13cg_md`)
* **Schema & Predicate Disambiguation** by embedding the Biolink-Model YAML in Chroma
* **Few-shot Examples** stored in Chroma for prompt retrieval
* **Fast Exact-Name Lookup** via a local pickle index of 5 GB+ of Biolink nodes(CURIEs)
* **Fallbacks & Repairs** with LLM-based fixes if the initial graph is invalid

---

## Prerequisites

* Linux or macOS
* Python 3.10+
* CUDA 11.7+ (for GPU acceleration)
* `git`, `curl` or `wget`

---

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/YourOrg/NL2TRAPI-LangGraph.git
   cd NL2TRAPI-LangGraph
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the SciSpaCy NER model**

   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_ner_bionlp13cg_md-0.5.5.tar.gz
   ```

---

## Data Preparation

1. **Biolink Model YAML**
   Download the latest biolink_model and place it as `biolink-model.yaml` in the project root.

2. **Nodes Dump**
   Obtain the TRAPI-canonical `nodes.json` (≈4–5 GB, 7 M records) and save it under:

   ```
   data/nodes.json
   ```

3. **Few-Shot Examples**
   Edit or confirm your `examples.json` at the project root. Each entry should look like:

   ```json
   {
     "nl_query": "What proteins does acetaminophen interact with?",
     "trapi_query": { / TRAPI JSON / }
   }
   ```

---

## Building Indices & Embeddings

Run each of these **once** (or whenever the source files change):

1. **Exact-Name Index**

   ```bash
   python -m scripts.build_exact_index
   ```

   → outputs `data/exact_index.pkl`

2. **Embed Biolink YAML**

   ```bash
   python -m scripts.embed_biolink_yaml
   ```

   → populates Chroma collection `yaml_schema`

3. **Embed NL→TRAPI Examples**

   ```bash
   python -m scripts.embed_nl2trapi_examples
   ```

   → populates Chroma collection `nl_to_trapi`

4. **Load Nodes into Chroma**

   ```bash
   python -m scripts.load_nodes_info
   ```

   → populates Chroma collection `nodes_info` 

---

## Running the Agent

Once all indices are built:

```bash
python -m scripts.run_agent_cli "What proteins does acetaminophen interact with?"
```

**Output**: a TRAPI Query Graph JSON or an error list if validation fails.

---

## Testing

```bash
pytest -q
```

---

## Configuration

All configurable paths and model names live in `trapi_agent/config.py` (a Pydantic `Settings` model). It can be overridden via environment variables or edit defaults directly:

* `DATA_DIR`
* `CHROMA_PERSIST_PATH`
* `EXACT_INDEX_PKL`
* `LLM_NAME` (e.g. `"BioMistral/BioMistral-7B"`)
* `EMB_MODEL` (e.g. `"pritamdeka/BioBERT-mnli-snli-..."`)

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
