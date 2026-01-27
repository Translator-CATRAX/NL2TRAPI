# trapi_agent/config.py
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Configuration for the NL→TRAPI agent.

    Data & indexes
    --------------
    base_data_dir          : Base directory for data artifacts (nodes, embeddings, pickles).
    YAML_SCHEMA_COLLECTION : Chroma collection for embedded Biolink YAML schema.
    EXAMPLE_COLLECTION     : Chroma collection for NL→TRAPI few-shot examples.
    NODES_COLLECTION       : Chroma collection for embedded biomedical node names.

    EXACT_INDEX_PKL        : (property) Path to exact-match node name index (pickle).
    CURIE2CAT_PKL          : (property) Path to CURIE→Biolink category map (pickle).
    CHROMA_PERSIST_PATH    : (property) Path where Chroma stores vector indexes.

    Models
    ------
    LLM_NAME               : HF model id for the generator (e.g., BioMistral-7B).
    EMB_MODEL              : HF model id for sentence embeddings.

    Runtime knobs
    -------------
    PREFER_GPU             : Try GPU first if available.
    USE_BITSANDBYTES       : Load model with bitsandbytes quantization when on GPU.
    QUANTIZATION           : "8bit" | "4bit" (honored only if USE_BITSANDBYTES=True).

    Validation / versions
    ---------------------
    MAX_FIX_ATTEMPTS       : Auto-repair attempts for invalid TRAPI graphs.
    TRAPI_VERSION          : Target ReasonerAPI (TRAPI) version.
    BIOLINK_VERSION        : Target Biolink Model version.

    All fields can be overridden via environment variables with prefix: TRAPI_AGENT_
    (e.g., TRAPI_AGENT_CURIE2CAT_PATH=/path/to/curie2cat.pkl)
    """

    # ── Base data directory ────────────────────────────────────────────────────
    base_data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )

    # ── Chroma collection names ───────────────────────────────────────────────
    YAML_SCHEMA_COLLECTION: str = "yaml_schema"
    EXAMPLE_COLLECTION: str = "nl_to_trapi"
    NODES_COLLECTION: str = "nodes_info"

    # Optional explicit paths (can be set via env); properties below
    # will fall back to sensible defaults under base_data_dir.
    exact_index_path: Path | None = None
    curie2cat_path: Path | None = None
    chroma_persist_dir: Path | None = None

    # ── Derived paths (do not set directly; use the *_path fields) ───────────
    @property
    def EXACT_INDEX_PKL(self) -> Path:
        """Exact name index (pickle)."""
        return Path(self.exact_index_path) if self.exact_index_path else self.base_data_dir / "exact_index.pkl"

    @property
    def CURIE2CAT_PKL(self) -> Path:
        """CURIE → Biolink category mapping (pickle)."""
        return Path(self.curie2cat_path) if self.curie2cat_path else self.base_data_dir / "curie2cat.pkl"

    @property
    def CHROMA_PERSIST_PATH(self) -> Path:
        """Chroma persistence directory."""
        return Path(self.chroma_persist_dir) if self.chroma_persist_dir else self.base_data_dir / "chroma_indexes"

    # ── Model identifiers ─────────────────────────────────────────────────────
    # Default to an instruction-tuned model suited for structured extraction.
    # Previous settings: "BioMistral/BioMistral-7B", "google/flan-t5-base"
    LLM_NAME: str = "microsoft/Phi-3.5-mini-instruct"
    EMB_MODEL: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

    # ── Runtime knobs (read by utils.llm, etc.) ───────────────────────────────
    PREFER_GPU: bool = True
    USE_BITSANDBYTES: bool = False
    QUANTIZATION: str = "8bit"  # "8bit" or "4bit" (used only when USE_BITSANDBYTES=True)

    # ── Validation / versions ─────────────────────────────────────────────────
    MAX_FIX_ATTEMPTS: int = 3
    TRAPI_VERSION: str = "1.4.0"
    BIOLINK_VERSION: str = "4.2.0"

    class Config:
        env_prefix = "TRAPI_AGENT_"
        env_file = ".env"
        extra = "ignore"


# Instantiate a global config object
settings = Settings()
