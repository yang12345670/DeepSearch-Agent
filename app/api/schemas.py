# -*- coding: utf-8 -*-
"""Pydantic schemas for API routes."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat request body."""

    query: str = Field(..., description="User query text.")
    session_id: Optional[str] = Field(
        default=None,
        description="Session id for short-term memory isolation.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User id for long-term memory isolation.",
    )
    use_rag: bool = Field(
        default=True,
        description="If False, skip retrieval and answer using conversation context only.",
    )
    mode: str = Field(
        default="orchestrator",
        description=(
            "Agent dispatch mode: 'orchestrator' (legacy Planner→RAG→Summarizer, "
            "default) or 'react' (LLM-driven tool-use loop with pluggable tools)."
        ),
    )


class ArtifactItem(BaseModel):
    """A side-effect produced by a ReAct tool call (file path, etc.)."""

    type: str = Field(..., description="Artifact type, e.g. 'pdf'.")
    file_path: str = Field(..., description="Server-side file path.")
    size_bytes: Optional[int] = None


class ToolCallSummary(BaseModel):
    """Compact view of one ReAct tool invocation."""

    name: str
    arguments: dict = Field(default_factory=dict)
    error: Optional[str] = None


class CitationItem(BaseModel):
    """One citation linking [n] in the answer to its source document."""

    id: int = Field(..., description="Citation number (matches [n] in answer).")
    source: str = Field(..., description="Source document filename.")
    text: str = Field(..., description="Evidence text snippet (truncated).")


class ChatResponse(BaseModel):
    """POST /chat response body."""

    answer: str = Field(
        ...,
        description="Clean user-facing answer (from FINAL_ANSWER block).",
    )
    debug_trace: Optional[str] = Field(
        default=None,
        description="Internal reasoning trace for debugging UI (collapsible).",
    )
    evidence_used: List[str] = Field(
        default_factory=list,
        description="Evidence snippets cited in the answer.",
    )
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="Source-linked citations for [n] references in the answer.",
    )
    mode: str = Field(default="orchestrator", description="Mode actually used.")
    artifacts: List[ArtifactItem] = Field(
        default_factory=list,
        description="Files produced by tools (e.g. generated PDFs).",
    )
    tool_calls: List[ToolCallSummary] = Field(
        default_factory=list,
        description="Compact ReAct tool-call trace (empty for orchestrator mode).",
    )


class UploadResponse(BaseModel):
    """POST /knowledge/upload response body."""

    success: bool
    message: str
    files_saved: List[str] = Field(default_factory=list)
    chunks_indexed: int = 0


class EvalRequest(BaseModel):
    """POST /eval/query request body."""

    question: str = Field(..., description="Evaluation question.")
    top_k: int = Field(default=5, description="Number of chunks to retrieve.")


class EvalResponse(BaseModel):
    """POST /eval/query response body."""

    answer: str = Field(default="", description="Model-generated answer.")
    retrieved_context: List[str] = Field(
        default_factory=list,
        description="Top-k retrieved chunk texts (before LLM, after rerank).",
    )
    retrieved_scores: List[float] = Field(
        default_factory=list,
        description="Corresponding reranker scores.",
    )


class SessionInfo(BaseModel):
    """One chat session."""

    session_id: str
    title: str = "New Chat"
    created_at: str = ""
    last_active: str = ""
    preview: str = ""


class SessionListResponse(BaseModel):
    """GET /chat/sessions response."""

    sessions: List[SessionInfo] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """GET /chat/history/{session_id} response."""

    session_id: str
    messages: List[dict] = Field(default_factory=list)


class KnowledgeFile(BaseModel):
    """One file in the knowledge base listing."""

    filename: str
    size: int
    modified: str


class KnowledgeListResponse(BaseModel):
    """GET /knowledge/list response body."""

    files: List[KnowledgeFile] = Field(default_factory=list)
    total: int = 0
