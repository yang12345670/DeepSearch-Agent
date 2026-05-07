"""Global configuration for DeepSearch Agent.

All settings can be overridden via environment variables or .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def _load_dotenv() -> None:
    """Load .env from project root if python-dotenv is available."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        # Manually parse simple KEY=VALUE lines as fallback
        import os
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            os.environ.setdefault(key, value)


_load_dotenv()

import os


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


class Settings(BaseModel):
    """Runtime settings — all values sourced from environment / .env."""

    # Supabase (chat history)
    supabase_url: str = Field(default_factory=lambda: _env("SUPABASE_URL", ""))
    supabase_key: str = Field(default_factory=lambda: _env("SUPABASE_KEY", ""))

    # LLM
    llm_provider: str = Field(default_factory=lambda: _env("LLM_PROVIDER", "local"))
    llm_api_key: str = Field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_base_url: str = Field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_model_name: str = Field(default_factory=lambda: _env("LLM_MODEL_NAME", "gpt-4o-mini"))
    llm_temperature: float = Field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = Field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))

    # Embedding
    embedding_model_name: str = Field(
        default_factory=lambda: _env("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    embedding_dim: int = Field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))

    # Reranker
    reranker_model_name: str = Field(
        default_factory=lambda: _env("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )

    # Chunker block classifier: "auto" | "rule" | "distilbert"
    # auto = use trained DistilBERT if models/block_classifier/ exists, else rule
    chunker_classifier: str = Field(
        default_factory=lambda: _env("CHUNKER_CLASSIFIER", "auto"),
    )

    # RAG
    rag_top_k: int = Field(default_factory=lambda: _env_int("RAG_TOP_K", 10))
    hybrid_alpha: float = Field(default_factory=lambda: _env_float("HYBRID_ALPHA", 0.5))
    hybrid_top_k_bm25: int = Field(default_factory=lambda: _env_int("HYBRID_TOP_K_BM25", 8))
    hybrid_top_k_dense: int = Field(default_factory=lambda: _env_int("HYBRID_TOP_K_DENSE", 8))
    hybrid_fusion_top_k: int = Field(default_factory=lambda: _env_int("HYBRID_FUSION_TOP_K", 8))

    # Data paths
    docs_dir: str = Field(default_factory=lambda: _env("DOCS_DIR", "data/docs"))
    index_dir: str = Field(default_factory=lambda: _env("INDEX_DIR", "data/index"))
    processed_dir: str = "data/processed"
    chunks_json_path: str = Field(default_factory=lambda: _env("INDEX_DIR", "data/index") + "/chunks.json")
    processed_chunks_json_path: str = "data/processed/chunks.json"
    faiss_index_path: str = Field(default_factory=lambda: _env("INDEX_DIR", "data/index") + "/faiss.index")
    embeddings_npy_path: str = Field(default_factory=lambda: _env("INDEX_DIR", "data/index") + "/embeddings.npy")

    # Context window / token budget
    context_window_tokens: int = Field(
        default_factory=lambda: _env_int("CONTEXT_WINDOW_TOKENS", 128000),
    )
    token_budget_evidence_ratio: float = Field(
        default_factory=lambda: _env_float("TOKEN_BUDGET_EVIDENCE_RATIO", 0.60),
    )
    token_budget_history_ratio: float = Field(
        default_factory=lambda: _env_float("TOKEN_BUDGET_HISTORY_RATIO", 0.25),
    )
    token_budget_memory_ratio: float = Field(
        default_factory=lambda: _env_float("TOKEN_BUDGET_MEMORY_RATIO", 0.15),
    )

    # Planner
    planner_enabled: bool = Field(
        default_factory=lambda: _env("PLANNER_ENABLED", "true").lower() in ("1", "true", "yes"),
    )

    # Memory
    redis_url: str = Field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    short_term_max_rounds: int = Field(default_factory=lambda: _env_int("SHORT_TERM_MAX_ROUNDS", 8))

    # Finance data sources (Tier 1: SEC EDGAR + Alpha Vantage)
    edgar_user_agent: str = Field(default_factory=lambda: _env("EDGAR_USER_AGENT", ""))
    alphavantage_api_key: str = Field(default_factory=lambda: _env("ALPHAVANTAGE_API_KEY", ""))
    finance_docs_dir: str = Field(default_factory=lambda: _env("FINANCE_DOCS_DIR", "data/docs_finance"))
    finance_state_path: str = "data/finance_source_state.json"
    finance_tracked_tickers: list[str] = Field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            "TSLA", "AVGO", "JPM", "V",
        ]
    )

    def ensure_data_dirs(self) -> None:
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def ensure_finance_dirs(self) -> None:
        Path(self.finance_docs_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.finance_docs_dir) / "edgar").mkdir(parents=True, exist_ok=True)
        (Path(self.finance_docs_dir) / "alphavantage").mkdir(parents=True, exist_ok=True)


settings = Settings()
