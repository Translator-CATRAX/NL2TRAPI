# trapi_agent/config.py

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Configuration for the NL→TRAPI agent.

    Attributes
    ----------
    DATA_DIR : Path
        Base directory for all data artifacts (nodes, embeddings, etc.).
    YAML_SCHEMA_COLLECTION : str
        Chroma collection name for embedded Biolink YAML schema.
    EXAMPLE_COLLECTION : str
        Chroma collection name for NL→TRAPI few-shot examples.
    NODES_COLLECTION : str
        Chroma collection name for embedded biomedical node names.
    EXACT_INDEX_PKL : Path
        Local path to the pickle file containing exact-match node index.
    CHROMA_PERSIST_PATH : Path
        Local path where ChromaDB stores vector index data.
    LLM_NAME : str
        HuggingFace model identifier used for text generation.
    EMB_MODEL : str
        HuggingFace model identifier used for embeddings.
    MAX_FIX_ATTEMPTS : int
        Maximum number of auto-repair attempts for invalid TRAPI graphs.
    """

    # Base data directory
    base_data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # Chroma collection names
    YAML_SCHEMA_COLLECTION: str = "yaml_schema"
    EXAMPLE_COLLECTION: str = "nl_to_trapi"
    NODES_COLLECTION: str = "nodes_info"

    # File paths derived from base_data_dir
    @property
    def EXACT_INDEX_PKL(self) -> Path:
        return self.base_data_dir / "exact_index.pkl"

    @property
    def CHROMA_PERSIST_PATH(self) -> Path:
        return self.base_data_dir / "chroma_indexes"

    # Model identifiers
    LLM_NAME: str = "BioMistral/BioMistral-7B"
    EMB_MODEL: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

    # Retry config
    MAX_FIX_ATTEMPTS: int = 3

    class Config:
        env_prefix = "TRAPI_AGENT_"
        env_file = ".env"
        extra = "ignore"


# Instantiate a global config object
settings = Settings()
