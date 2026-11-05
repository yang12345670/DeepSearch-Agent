"""Typed runtime settings loaded from environment, .env, then .env.example.

Defaults live in .env.example so deploy-time values have one editable source.
Configuration is read at startup; restart the process after changing .env.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from string import Formatter
from typing import Literal, get_origin
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_NAMES = {
    "embedding_force_hash": "DEEPSEARCH_EMBED_FORCE_HASH",
    "simple_reranker": "DEEPSEARCH_SIMPLE_RERANKER",
}


def _http_url(value: str, name: str, *, optional: bool = False) -> None:
    if optional and not value:
        return
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password)
        parsed.port  # Reject malformed ports.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an HTTP(S) URL without credentials")


def _template(value: str, name: str, placeholder: str) -> None:
    try:
        fields = [(field, spec, conversion) for _, field, spec, conversion
                  in Formatter().parse(value) if field is not None]
    except ValueError:
        raise ValueError(f"{name} contains an invalid template") from None
    if not fields or any(field != placeholder or spec or conversion
                         for field, spec, conversion in fields):
        raise ValueError(f"{name} must use only {{{placeholder}}} as a placeholder")


class Settings(BaseModel):
    """Validated configuration; explicit constructor arguments override file values."""

    model_config = ConfigDict(hide_input_in_errors=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def load_environment(cls, values):
        example = PROJECT_ROOT / ".env.example"
        if not example.is_file():
            raise ValueError("Missing .env.example in the project root")
        defaults = dotenv_values(example, interpolate=False)
        local = dotenv_values(PROJECT_ROOT / ".env", interpolate=False)
        resolved = dict(values)
        for name, field in cls.model_fields.items():
            if name in resolved:
                continue
            key = _ENV_NAMES.get(name, name.upper())
            raw = os.environ.get(key, local.get(key, defaults.get(key)))
            if raw is None:
                raise ValueError(f"Missing configuration: {key}")
            origin = get_origin(field.annotation)
            if origin in (list, dict):
                try:
                    if origin is list and not raw.lstrip().startswith("["):
                        raw = [item.strip() for item in raw.split(",") if item.strip()]
                    else:
                        raw = json.loads(raw)
                except (ValueError, TypeError):
                    raise ValueError(f"{key} must contain a valid JSON {'array' if origin is list else 'object'}") from None
            resolved[name] = raw
        return resolved

    # Server and browser / 服务与前端
    server_host: str = Field(min_length=1)
    server_port: int = Field(ge=1, le=65535)
    server_reload: bool
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    log_format: str = Field(min_length=1)
    cors_allow_origins: list[str]
    cors_allow_credentials: bool
    cors_allow_methods: list[str]
    cors_allow_headers: list[str]
    startup_auto_index: bool
    api_base_url: str
    frontend_use_rag: bool
    chat_db_path: str = Field(min_length=1)
    chat_db_timeout: float = Field(gt=0)

    # Models / 模型
    llm_provider: str = Field(min_length=1)
    llm_api_key: str = Field(repr=False)
    llm_base_url: str
    llm_model_name: str = Field(min_length=1)
    llm_temperature: float = Field(ge=0)
    llm_max_tokens: int = Field(gt=0)
    llm_request_timeout_seconds: float = Field(gt=0)
    llm_connect_timeout_seconds: float = Field(gt=0)
    llm_max_retries: int = Field(ge=0)
    llm_provider_base_urls: dict[str, str]
    embedding_model_name: str = Field(min_length=1)
    embedding_dim: int = Field(gt=0)
    embedding_batch_size: int = Field(gt=0)
    embedding_force_hash: bool
    reranker_model_name: str = Field(min_length=1)
    simple_reranker: bool
    reranker_top_n: int = Field(gt=0)

    # Documents and retrieval / 文档与检索
    docs_dir: str = Field(min_length=1)
    index_dir: str = Field(min_length=1)
    chunks_json_path: str
    faiss_index_path: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    chunk_overlap_sentences: int = Field(ge=0)
    min_chunk_size: int = Field(gt=0)
    chunker_classifier: Literal['auto', 'rule', 'distilbert']
    classifier_model_dir: str = Field(min_length=1)
    classifier_cache_size: int = Field(ge=0)
    classifier_max_length: int = Field(gt=0)
    rag_top_k: int = Field(gt=0)
    hybrid_alpha: float = Field(ge=0, le=1)
    hybrid_top_k_bm25: int = Field(ge=0)
    hybrid_top_k_dense: int = Field(ge=0)
    hybrid_fusion_top_k: int = Field(gt=0)
    rag_score_threshold: float
    rag_max_docs: int = Field(gt=0)
    rag_score_gap_ratio: float = Field(ge=0, le=1)
    rag_max_refine_rounds: int = Field(ge=0)
    refine_current_top_k: int = Field(gt=0)
    refine_expanded_top_k: int = Field(gt=0)
    refine_missing_points_limit: int = Field(ge=0)
    refine_unsupported_claims_limit: int = Field(ge=0)
    citation_excerpt_chars: int = Field(gt=0)
    verifier_evidence_threshold: float = Field(ge=0, le=1)
    verifier_coverage_threshold: float = Field(ge=0, le=1)
    verifier_uncertainty_threshold: float = Field(ge=0, le=1)

    # Agent and context / Agent 与上下文
    planner_enabled: bool
    react_max_rounds: int = Field(gt=0)
    context_window_tokens: int = Field(gt=0)
    token_budget_evidence_ratio: float = Field(ge=0, le=1)
    token_budget_history_ratio: float = Field(ge=0, le=1)
    tokenizer_model_name: str = Field(min_length=1)
    token_estimate_chars_per_token: int = Field(gt=0)
    token_output_reserve: int = Field(ge=0)
    token_query_overhead: int = Field(ge=0)
    token_evidence_header_overhead: int = Field(ge=0)
    token_min_evidence_tokens: int = Field(ge=0)
    history_recent_messages: int = Field(gt=0)
    history_summary_chars: int = Field(gt=0)

    # Memory / 记忆
    redis_url: str
    short_term_max_rounds: int = Field(gt=0)
    short_term_keep_rounds: int = Field(ge=0)
    short_term_max_traces: int = Field(gt=0)
    short_term_context_traces: int = Field(ge=0)
    short_term_trace_summary_chars: int = Field(gt=0)
    short_term_key_prefix: str = Field(min_length=1)
    memory_index_dir: str = Field(min_length=1)
    memory_dedup_threshold: float = Field(ge=0, le=1)
    memory_recall_top_k: int = Field(gt=0)
    memory_recall_candidate_multiplier: int = Field(gt=0)
    memory_confidence_weight: float = Field(ge=0)
    memory_importance_weight: float = Field(ge=0)
    memory_recency_weight: float = Field(ge=0)
    memory_gate_threshold: int = Field(ge=0)
    memory_task_summary_chars: int = Field(gt=0)

    # Financial sources / 金融数据源
    edgar_user_agent: str = Field(repr=False)
    alphavantage_api_key: str = Field(repr=False)
    finance_docs_dir: str = Field(min_length=1)
    finance_state_path: str = Field(min_length=1)
    financial_cache_dir: str = Field(min_length=1)
    finance_tracked_tickers: list[str]
    finance_ingest_sources: list[str]
    edgar_forms: list[str]
    edgar_backfill_days: int = Field(ge=0)
    alphavantage_backfill_days: int = Field(ge=0)
    sec_cache_ttl_seconds: float = Field(ge=0)
    sec_min_request_interval_seconds: float = Field(ge=0)
    sec_request_timeout_seconds: float = Field(gt=0)
    sec_request_attempts: int = Field(ge=1)
    sec_retry_backoff_base_seconds: float = Field(ge=0)
    sec_max_inline_retry_seconds: float = Field(ge=0)
    sec_tickers_url: str
    sec_company_facts_url_template: str
    alphavantage_api_url: str
    alphavantage_request_interval_seconds: float = Field(ge=0)
    alphavantage_request_timeout_seconds: float = Field(gt=0)
    alphavantage_limit_per_ticker: int = Field(ge=1)
    financial_default_limit: int = Field(ge=1, le=10)
    financial_default_metrics: list[str]
    financial_calculation_default_metrics: list[str]

    # PDF export / PDF 导出
    pdf_exports_dir: str = Field(min_length=1)
    pdf_font_name: str = Field(min_length=1)

    # Prompts / 提示词（保留输出标签和模板占位符）
    react_system_prompt: str = Field(min_length=1)
    rag_system_prompt: str = Field(min_length=1)
    chat_only_system_prompt: str = Field(min_length=1)
    summarizer_system_prompt: str = Field(min_length=1)
    planner_system_prompt: str = Field(min_length=1)
    memory_extract_system_prompt: str = Field(min_length=1)
    memory_extract_user_template: str = Field(min_length=1)
    planner_user_template: str = Field(min_length=1)
    rag_answer_instructions: str = Field(min_length=1)
    @model_validator(mode="after")
    def validate_runtime(self):
        if self.chunk_overlap >= self.chunk_size or self.min_chunk_size > self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE; MIN_CHUNK_SIZE must not exceed CHUNK_SIZE")
        if self.short_term_keep_rounds > self.short_term_max_rounds:
            raise ValueError("SHORT_TERM_KEEP_ROUNDS must not exceed SHORT_TERM_MAX_ROUNDS")
        if self.token_budget_evidence_ratio + self.token_budget_history_ratio > 1:
            raise ValueError("TOKEN_BUDGET_EVIDENCE_RATIO + TOKEN_BUDGET_HISTORY_RATIO must not exceed 1")
        if self.token_output_reserve >= self.context_window_tokens:
            raise ValueError("TOKEN_OUTPUT_RESERVE must be smaller than CONTEXT_WINDOW_TOKENS")
        if self.memory_confidence_weight + self.memory_importance_weight + self.memory_recency_weight <= 0:
            raise ValueError("At least one MEMORY_*_WEIGHT must be positive")
        for name in ("llm_base_url", "api_base_url"):
            value = getattr(self, name)
            if name == "api_base_url" and value.startswith("/") and not value.startswith("//"):
                continue
            _http_url(value, name.upper(), optional=True)
        for value in self.llm_provider_base_urls.values():
            _http_url(value, "LLM_PROVIDER_BASE_URLS")
        for name in ("sec_tickers_url", "sec_company_facts_url_template", "alphavantage_api_url"):
            _http_url(getattr(self, name), name.upper())
        _template(self.sec_company_facts_url_template, "SEC_COMPANY_FACTS_URL_TEMPLATE", "cik")
        _template(self.memory_extract_user_template, "MEMORY_EXTRACT_USER_TEMPLATE", "conversation")
        _template(self.planner_user_template, "PLANNER_USER_TEMPLATE", "query")
        if not self.finance_ingest_sources or set(self.finance_ingest_sources) - {"edgar", "av"}:
            raise ValueError("FINANCE_INGEST_SOURCES must contain edgar and/or av")
        fact_names = {"revenue", "net_income", "assets", "equity", "cash", "operating_cash_flow"}
        metric_names = {"revenue_growth_yoy", "net_income_growth_yoy", "net_margin", "operating_cash_flow_margin"}
        if not self.financial_default_metrics or set(self.financial_default_metrics) - fact_names:
            raise ValueError("FINANCIAL_DEFAULT_METRICS contains an unsupported financial fact")
        if not self.financial_calculation_default_metrics or set(self.financial_calculation_default_metrics) - metric_names:
            raise ValueError("FINANCIAL_CALCULATION_DEFAULT_METRICS contains an unsupported metric")
        for name in ("financial_default_metrics", "financial_calculation_default_metrics"):
            items = getattr(self, name)
            if len(items) != len(set(items)):
                raise ValueError(f"{name.upper()} must not contain duplicates")
        for name in ("edgar_forms", "finance_tracked_tickers"):
            if not getattr(self, name):
                raise ValueError(f"{name.upper()} must not be empty")
        self.chunks_json_path = self.chunks_json_path or str(Path(self.index_dir) / "chunks.json")
        self.faiss_index_path = self.faiss_index_path or str(Path(self.index_dir) / "faiss.index")
        classifier_dir = Path(self.classifier_model_dir).expanduser()
        if not classifier_dir.is_absolute():
            classifier_dir = PROJECT_ROOT / classifier_dir
        self.classifier_model_dir = str(classifier_dir)
        return self

    def ensure_data_dirs(self) -> None:
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chunks_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)

    def ensure_finance_dirs(self) -> None:
        Path(self.finance_docs_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.finance_docs_dir) / "edgar").mkdir(parents=True, exist_ok=True)
        (Path(self.finance_docs_dir) / "alphavantage").mkdir(parents=True, exist_ok=True)


settings = Settings()
