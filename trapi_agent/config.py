# trapi_agent/config.py

from pathlib import Path
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """
    Application configuration for NL→TRAPI agent.

    Attributes
    ----------
    DATA_DIR
        Filesystem path where all on‐disk data (nodes.json, exact_index.pkl,
        Chroma DB files, etc.) are stored.
    YAML_SCHEMA_COLLECTION
        Name of the Chroma collection holding embedded Biolink YAML (classes & slots).
    EXAMPLE_COLLECTION
        Name of the Chroma collection holding NL→TRAPI few-shot examples.
    NODES_COLLECTION
        Name of the Chroma collection holding embedded node names for semantic lookup.
    EXACT_INDEX_PKL
        Path to the exact-match pickle built from the canonical nodes dump.
    CHROMA_PERSIST_PATH
        Filesystem path where Chroma persists its vector indexes.
    LLM_NAME
        Hugging Face model identifier for the text-generation LLM.
    EMB_MODEL
        Hugging Face model identifier for the sentence-embedding encoder.
    MAX_FIX_ATTEMPTS
        Maximum number of auto-repair loops the agent will attempt on an invalid TRAPI graph.
    """

    # ─── Paths & Collections ───────────────────────────────────────────
    DATA_DIR: Path = Field(
        default=Path(__file__).parent.parent / "data",
        description="Base directory for all data artifacts."
    )
    YAML_SCHEMA_COLLECTION: str = Field(
        default="yaml_schema",
        description="Chroma collection name for Biolink schema embeddings."
    )
    EXAMPLE_COLLECTION: str = Field(
        default="nl_to_trapi",
        description="Chroma collection name for NL→TRAPI few-shot examples."
    )
    NODES_COLLECTION: str = Field(
        default="nodes_info",
        description="Chroma collection name for node name embeddings."
    )
    EXACT_INDEX_PKL: Path = Field(
        default=DATA_DIR / "exact_index.pkl",
        description="Path to exact-match index pickle for NodeResolver."
    )

    # ─── Chroma Settings ───────────────────────────────────────────────
    CHROMA_PERSIST_PATH: Path = Field(
        default=DATA_DIR / "chroma_indexes",
        description="Filesystem path for ChromaDB persistence."
    )

    # ─── Model Identifiers ─────────────────────────────────────────────
    LLM_NAME: str = Field(
        default="BioMistral/BioMistral-7B",
        description="Hugging Face model name for text generation."
    )
    EMB_MODEL: str = Field(
        default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        description="Hugging Face model name for sentence embeddings."
    )

    # ─── Retry Policy ─────────────────────────────────────────────────
    MAX_FIX_ATTEMPTS: int = Field(
        default=3,
        ge=0,
        description="Maximum TRAPI fix attempts before giving up."
    )

    class Config:
        # Allow overriding via environment variables if needed
        env_prefix = "TRAPI_AGENT_"
        env_file = ".env"
        extra = "ignore"


# Instantiate a singleton settings object for module-wide use
settings = Settings()
