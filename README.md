# NL2TRAPI

This repository contains scripts for converting natural language queries into TRAPI format using Chroma DB and an LLM.

## Files
- `db1_script.py`: Script for populating DB1 with NL-to-TRAPI examples.
- `db2_script.py`: Script for populating DB2 with schema and biolink information.
- `query_generation.py`: Script for generating TRAPI queries from natural language inputs.
- `biolink-model.yaml`: YAML schema file.
- `nodes.json`: JSON file containing node information.

## Usage
1. pip install -r requirements.txt
2. Run `db1_script.py` and `db2_script.py` to populate Chroma DB.
3. Use `query_generation.py` to generate TRAPI queries from natural language inputs.

## Installation
Install dependencies using pip:
