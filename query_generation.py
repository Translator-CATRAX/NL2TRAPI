import re
import json
import torch
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import time
import chromadb
from transformers import pipeline
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configuration
@dataclass
class Config:
    examples_db_path: str = "/scratch/vmm5481/NL2TRAPI/nl_to_trapi_db"
    schema_db_path: str = "/scratch/vmm5481/NL2TRAPI/chroma_db"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    embedding_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    device: str = "cuda"
    top_k_examples: int = 3
    top_k_entities: int = 5
    top_k_schema: int = 3
    max_new_tokens: int = 500
    temperature: float = 0.3
    top_p: float = 0.95
    repetition_penalty: float = 1.4
    log_dir: str = "experiment_logs"

class Constants:
    STOPWORDS = {
        "what", "which", "is", "are", "a", "an", "the", "in", "of", "to", 
        "for", "and", "on", "with", "by", "how", "at", "that", "this",
        "from", "or", "but", "be", "have", "has", "do", "does", "can",
        "could", "should", "would", "will", "may", "might"
    }
    EXCLUDED_CATEGORIES = {
        "biolink:ClinicalAttribute", "biolink:InformationContentEntity",
        "biolink:Event", "biolink:ExposureEvent", "biolink:GeographicLocation",
        "biolink:PopulationOfIndividualOrganisms", "biolink:Publication",
        "biolink:RetrievalSource", "biolink:StudyPopulation", 
        "biolink:SubjectOfInvestigation"
    }
    BIOMEDICAL_INDICATORS = {
        "gene", "protein", "disease", "drug", "pathway", "phenotype",
        "biological process", "molecular function", "cellular component",
        "anatomy", "organism", "chemical", "variant"
    }

# Enhanced logging setup
def setup_logging(config: Config) -> logging.Logger:
    """Configure dual logging: file for debug, console for clean output"""
    Path(config.log_dir).mkdir(exist_ok=True)
    log_file = Path(config.log_dir) / f"nl2trapi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear any existing handlers to avoid duplicates
    logger = logging.getLogger("NL2TRAPI")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # File handler (debug level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (info level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Experiment log: {log_file}")
    return logger

class DebugLogger:
    """Structured logging for experiment tracking"""
    def __init__(self, config: Config):
        self.logger = setup_logging(config)
        self.config = config
        self.debug_data = {
            "query": None,
            "timing": {},
            "steps": {},
            "error": None
        }
    
    def log_operation(self, step_name: str, data: Dict, exec_time: float):
        """Log a single operation with timing and context"""
        self.debug_data["steps"][step_name] = {
            "data": data,
            "time_sec": round(exec_time, 3)
        }
        self.debug_data["timing"][step_name] = round(exec_time, 3)
        
        self.logger.debug(
            f"{step_name.upper()} ({exec_time:.3f}s)\n{json.dumps(data, indent=2)}"
        )
    
    def log_final_result(self, success: bool, result: Any = None):
        """Log final outcome of the pipeline"""
        self.debug_data["success"] = success
        self.debug_data["result"] = result
        
        total_time = sum(self.debug_data["timing"].values())
        self.debug_data["total_time_sec"] = round(total_time, 3)
        
        self.logger.debug("\nFINAL RESULT:\n" + json.dumps(
            self.debug_data, 
            indent=2, 
            default=str
        ))
        
        if success:
            self.logger.info(f"✅ Successfully generated TRAPI (total: {total_time:.3f}s)")
        else:
            self.logger.info(f"❌ Failed to generate valid TRAPI (total: {total_time:.3f}s)")

class DatabaseManager:
    def __init__(self, config: Config, logger: DebugLogger):
        self.config = config
        self.logger = logger
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model,
            device=config.device
        )
        self.clients = {}
    
    def get_client(self, db_type: str) -> chromadb.PersistentClient:
        if db_type not in self.clients:
            path = self.config.examples_db_path if db_type == "examples" else self.config.schema_db_path
            self.clients[db_type] = chromadb.PersistentClient(path=path)
        return self.clients[db_type]
    
    def search_collection(self, db_type: str, collection: str, query: str, **kwargs):
        start = time.time()
        try:
            client = self.get_client(db_type)
            col = client.get_collection(collection, embedding_function=self.embedding_fn)
            results = col.query(query_texts=[query], **kwargs)
            
            log_data = {
                "query": query,
                "results_count": len(results["documents"][0]) if results["documents"] else 0,
                "metadata": [m for m in results["metadatas"][0]] if results.get("metadatas") and results["metadatas"][0] else []
            }
            
            self.logger.log_operation(
                f"db_query_{db_type}_{collection}",
                log_data,
                time.time() - start
            )
            return results
            
        except Exception as e:
            self.logger.logger.error(f"Database error: {str(e)}")
            return None
    
    def exact_node_match(self, query: str) -> List[Dict]:
        """Attempt exact node name matching."""
        try:
            client = self.get_client("schema")
            collection = client.get_collection(
                name="nodes_info", 
                embedding_function=self.embedding_fn
            )
            results = collection.get(where={"name": query})
            
            if results and results.get("metadatas") and results["metadatas"]:
                self.logger.logger.debug(f"Exact match found for node: {query}")
                return [{
                    "documents": [results["documents"]],
                    "metadatas": [results["metadatas"]]
                }]
            return []
        except Exception as e:
            self.logger.logger.warning(f"Exact match failed for node '{query}': {e}")
            return []

class QueryProcessor:
    """Handles natural language query processing and entity extraction."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract_entities_and_keywords(self, nl_query: str) -> Dict[str, List[str]]:
        """Enhanced entity and keyword extraction."""
        # Normalize query
        normalized = re.sub(r'[^\w\s]', ' ', nl_query.lower())
        
        # Extract potential biomedical entities (multi-word phrases)
        entities = []
        for indicator in Constants.BIOMEDICAL_INDICATORS:
            pattern = rf'\b{re.escape(indicator)}\b[\w\s]*'
            matches = re.findall(pattern, normalized)
            entities.extend([m.strip() for m in matches if len(m.strip()) > len(indicator)])
        
        # Extract meaningful phrases (2-4 words)
        phrases = re.findall(r'\b(?:\w+\s+){1,3}\w+\b', normalized)
        meaningful_phrases = [
            p.strip() for p in phrases 
            if len(p.split()) >= 2 and 
            not all(w in Constants.STOPWORDS for w in p.split())
        ]
        
        # Extract individual keywords
        tokens = re.findall(r'\b\w+\b', normalized)
        keywords = [
            t for t in tokens 
            if t not in Constants.STOPWORDS and len(t) > 2
        ]
        
        # Remove duplicates and sort by length (longer first)
        all_terms = entities + meaningful_phrases + keywords
        unique_terms = sorted(set(all_terms), key=len, reverse=True)
        
        # Categorize terms
        result = {
            "entities": [t for t in unique_terms if len(t.split()) > 1],
            "keywords": [t for t in unique_terms if len(t.split()) == 1],
            "all_terms": unique_terms
        }
        
        self.logger.debug(f"Extracted entities: {result['entities']}")
        self.logger.debug(f"Extracted keywords: {result['keywords']}")
        return result
    
    def prioritize_search_terms(self, extracted_terms: Dict[str, List[str]]) -> List[str]:
        """Prioritize search terms based on biomedical relevance."""
        prioritized = []
        
        # First: entities with biomedical indicators
        for entity in extracted_terms["entities"]:
            if any(indicator in entity for indicator in Constants.BIOMEDICAL_INDICATORS):
                prioritized.append(entity)
        
        # Second: other entities
        for entity in extracted_terms["entities"]:
            if entity not in prioritized:
                prioritized.append(entity)
        
        # Third: keywords not in stopwords
        for keyword in extracted_terms["keywords"]:
            if keyword not in prioritized:
                prioritized.append(keyword)
        
        return prioritized[:10]  # Limit to top 10 terms

class EntityResolver:
    """Resolves entities to biomedical knowledge graph nodes."""
    
    def __init__(self, db_manager: DatabaseManager, logger: logging.Logger):
        self.db_manager = db_manager
        self.logger = logger
    
    def resolve_entities(self, nl_query: str, search_terms: List[str]) -> List[Dict]:
        """Resolve entities through multiple lookup strategies."""
        resolved = []
        seen_ids = set()
        
        # First: exact match for full query
        exact_results = self.db_manager.exact_node_match(nl_query)
        for result in exact_results:
            if result["metadatas"][0]:
                node_id = result["metadatas"][0][0].get("id")
                if node_id and node_id not in seen_ids:
                    resolved.append(result)
                    seen_ids.add(node_id)
        
        # Second: semantic search for each term
        for term in search_terms:
            if len(resolved) >= 5:  # Limit total resolved entities
                break
                
            try:
                results = self.db_manager.search_collection(
                    "schema", "nodes_info", term, 
                    n_results=1,
                    where={"category": {"$nin": list(Constants.EXCLUDED_CATEGORIES)}}
                )
                
                if results and results["documents"][0]:
                    metadata = results["metadatas"][0][0]
                    node_id = metadata.get("id")
                    
                    if node_id and node_id not in seen_ids:
                        resolved.append({
                            "documents": results["documents"],
                            "metadatas": results["metadatas"]
                        })
                        seen_ids.add(node_id)
                        self.logger.debug(f"Resolved '{term}' to {node_id}")
                        
            except Exception as e:
                self.logger.warning(f"Resolution failed for '{term}': {e}")
        
        self.logger.debug(f"Total resolved entities: {len(resolved)}")
        return resolved

class PromptBuilder:
    """Builds optimized prompts for TRAPI generation."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def build_prompt(self, nl_query: str, examples: Optional[Dict], 
                    schema: Optional[Dict], resolved_nodes: List[Dict]) -> str:
        """Build a comprehensive prompt for TRAPI generation."""
        prompt_parts = [
            "You are an expert at converting biomedical queries to TRAPI (Translator API) JSON format.",
            "",
            "RULES:",
            "1. Output ONLY valid JSON in query_graph format",
            "2. Use provided entity IDs when available",
            "3. Use '?' for unknown entities with appropriate categories",
            "4. Include both nodes and edges in the query_graph",
            "5. Ensure all edge references use valid node keys",
            "",
        ]
        
        # Add examples if available
        if examples and examples.get("documents") and examples["documents"][0]:
            prompt_parts.append("EXAMPLES:")
            for i in range(min(2, len(examples["documents"][0]))):
                nl_example = examples["documents"][0][i]
                trapi_data = examples["metadatas"][0][i].get("trapi_query", {})
                if trapi_data:
                    prompt_parts.extend([
                        f"Query: {nl_example}",
                        f"TRAPI: {json.dumps(trapi_data, indent=1)}",
                        ""
                    ])
        
        # Add resolved entities
        if resolved_nodes:
            prompt_parts.append("AVAILABLE ENTITIES:")
            for result in resolved_nodes:
                if result["metadatas"][0]:
                    meta = result["metadatas"][0][0]
                    name = meta.get("name", "")
                    node_id = meta.get("id", "")
                    category = meta.get("category", "")
                    prompt_parts.append(f"- {name}: {node_id} ({category})")
            prompt_parts.append("")
        
        # Add schema hints
        if schema and schema.get("documents") and schema["documents"]:
            prompt_parts.append("SCHEMA HINTS:")
            for i, doc_list in enumerate(schema["documents"][:3]):
                if doc_list:  # Check if document list is not empty
                    key = schema["metadatas"][i][0].get("key", "") if schema["metadatas"][i] else ""
                    desc = doc_list[0] if isinstance(doc_list, list) else doc_list
                    if len(desc) > 100:
                        desc = desc[:100] + "..."
                    prompt_parts.append(f"- {key}: {desc}")
            prompt_parts.append("")
        
        # Add the actual query
        prompt_parts.extend([
            f"Convert this query to TRAPI JSON: {nl_query}",
            "",
            "Output only the JSON for query_graph:"
        ])
        
        full_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Built prompt with {len(full_prompt)} characters")
        return full_prompt

class TrapiGenerator:
    """Generates TRAPI queries using language models."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
    
    def initialize_model(self):
        """Initialize the language model."""
        if self.model is None:
            self.logger.info(f"Loading model: {self.config.model_name}")
            try:
                self.model = pipeline(
                    "text-generation",
                    model=self.config.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    return_full_text=False,
                    trust_remote_code=True
                )
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise
    
    def generate(self, prompt: str) -> Optional[Dict]:
        """Generate TRAPI from prompt."""
        if self.model is None:
            self.initialize_model()
        
        try:
            self.logger.debug("Generating TRAPI response...")
            
            response = self.model(
                prompt,
                do_sample=True,
                top_k=50,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                num_return_sequences=1,
                eos_token_id=self.model.tokenizer.eos_token_id,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.model.tokenizer.eos_token_id
            )[0]["generated_text"]
            
            self.logger.debug("Raw response received from model")
            return self._extract_json(response)
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return None
    
    def _extract_json(self, response: str) -> Optional[Dict]:
        """Extract JSON from model response with multiple strategies."""
        # Strategy 1: Look for JSON between markers
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\{.*?\})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    parsed = json.loads(json_str)
                    if self._validate_trapi_structure(parsed):
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        self.logger.warning("Could not extract valid JSON from response")
        self.logger.debug(f"Raw response: {response[:500]}...")
        return None
    
    def _validate_trapi_structure(self, trapi: Any) -> bool:
        """Validate basic TRAPI structure."""
        if not isinstance(trapi, dict):
            return False
        
        # Check for query_graph structure
        if "query_graph" in trapi:
            qg = trapi["query_graph"]
        else:
            qg = trapi
        
        # Validate required fields
        if not all(key in qg for key in ["nodes", "edges"]):
            return False
        
        if not isinstance(qg["nodes"], dict) or not isinstance(qg["edges"], dict):
            return False
        
        # Validate node and edge references
        node_ids = set(qg["nodes"].keys())
        for edge_id, edge in qg["edges"].items():
            if not isinstance(edge, dict):
                return False
            subject = edge.get("subject")
            object_node = edge.get("object")
            if not (subject in node_ids and object_node in node_ids):
                return False
        
        return True

# Main NL2TRAPI class
class NL2TRAPI:
    def __init__(self, config: Config):
        self.config = config
        self.logger = DebugLogger(config)
        self.db_manager = DatabaseManager(config, self.logger)
        self.query_processor = QueryProcessor(self.logger.logger)
        self.entity_resolver = EntityResolver(self.db_manager, self.logger.logger)
        self.prompt_builder = PromptBuilder(self.logger.logger)
        self.trapi_generator = TrapiGenerator(config, self.logger.logger)

    def convert(self, nl_query: str) -> Optional[Dict]:
        self.logger.debug_data["query"] = nl_query
        self.logger.logger.info(f"\nProcessing query: {nl_query}")
        
        try:
            # Entity extraction
            start = time.time()
            terms = self.query_processor.extract_entities_and_keywords(nl_query)
            prioritized_terms = self.query_processor.prioritize_search_terms(terms)
            self.logger.log_operation(
                "entity_extraction",
                {"terms": prioritized_terms},
                time.time() - start
            )

            # Example retrieval
            start = time.time()
            examples = self.db_manager.search_collection(
                "examples", "nl_to_trapi", nl_query,
                n_results=self.config.top_k_examples
            )
            example_queries = examples["documents"][0] if examples and examples["documents"] else []
            self.logger.log_operation(
                "example_retrieval",
                {"example_queries": example_queries},
                time.time() - start
            )

            # Entity resolution
            start = time.time()
            resolved_nodes = self.entity_resolver.resolve_entities(nl_query, prioritized_terms)
            resolved_data = []
            if resolved_nodes:
                for n in resolved_nodes:
                    if n.get("metadatas") and n["metadatas"][0]:
                        meta = n["metadatas"][0][0]
                        resolved_data.append({
                            "name": meta.get("name", ""),
                            "id": meta.get("id", ""),
                            "category": meta.get("category", "")
                        })
            self.logger.log_operation(
                "entity_resolution",
                {"resolved_entities": resolved_data},
                time.time() - start
            )

            # Schema retrieval
            start = time.time()
            schema = self._retrieve_schema(prioritized_terms)
            schema_preview = []
            if schema and schema.get("documents"):
                for doc_list in schema["documents"][:3]:
                    if doc_list:
                        preview = doc_list[0] if isinstance(doc_list, list) else doc_list
                        schema_preview.append(preview[:100] + "..." if len(preview) > 100 else preview)
            self.logger.log_operation(
                "schema_retrieval",
                {"schema_entries": schema_preview},
                time.time() - start
            )

            # Prompt building
            start = time.time()
            prompt = self.prompt_builder.build_prompt(nl_query, examples, schema, resolved_nodes)
            self.logger.log_operation(
                "prompt_construction",
                {"prompt_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt},
                time.time() - start
            )

            # TRAPI generation
            start = time.time()
            trapi_result = self.trapi_generator.generate(prompt)
            self.logger.log_operation(
                "trapi_generation",
                {"generated": trapi_result is not None},
                time.time() - start
            )

            # Validate and return
            if trapi_result and self.trapi_generator._validate_trapi_structure(trapi_result):
                self.logger.log_final_result(True, trapi_result)
                return trapi_result
            
            self.logger.log_final_result(False)
            return None

        except Exception as e:
            self.logger.debug_data["error"] = str(e)
            self.logger.logger.error(f"Pipeline error: {str(e)}")
            self.logger.log_final_result(False)
            return None

    def _retrieve_schema(self, terms: List[str]) -> Dict:
        """Retrieve schema information for given terms."""
        schema = {"documents": [], "metadatas": []}
        for term in terms[:self.config.top_k_schema]:
            results = self.db_manager.search_collection("schema", "yaml_schema", term, n_results=1)
            if results and results.get("documents") and results["documents"][0]:
                schema["documents"].append(results["documents"][0])
                schema["metadatas"].append(results["metadatas"][0])
        return schema

def main():
    """Main function for testing the enhanced NL2TRAPI system."""
    config = Config()
    nl2trapi = NL2TRAPI(config)
    
    test_queries = [
        "What biological processes are related to GFAP?",
        # "What drugs treat Alzheimer's disease?",
        # "How does aspirin affect inflammation?",
        # "What pathways are involved in cancer progression?"
    ]
    
    print("="*80)
    print("NL2TRAPI Evaluation")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Processing: {query}")
        print("-" * 60)
        
        result = nl2trapi.convert(query)
        
        if result:
            print("\n✅ Generated TRAPI:")
            print(json.dumps(result, indent=2))
        else:
            print("\n❌ Failed to generate valid TRAPI")
        
        print("-" * 60)
    
    print("\n" + "="*80)
    print("Evaluation complete. Check logs for detailed analysis.")
    print("="*80)

if __name__ == "__main__":
    main()






