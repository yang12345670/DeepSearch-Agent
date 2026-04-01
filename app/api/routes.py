# -*- coding: utf-8 -*-
"""HTTP routes."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile

from app.agent.pipeline import get_agent_pipeline, reset_agent_pipeline
from app.api.schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    EvalRequest,
    EvalResponse,
    KnowledgeFile,
    KnowledgeListResponse,
    SessionInfo,
    SessionListResponse,
    UploadResponse,
)
from app.storage import chat_store
from app.config import settings
from app.rag.auto_index import rebuild_index

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed file extensions per category (extensible for future multimodal support)
_ALLOWED_EXTENSIONS = {
    "document": {".md"},
    # Future: "image": {".png", ".jpg", ".jpeg", ".webp"},
    # Future: "pdf": {".pdf"},
}

_ALL_ALLOWED = set()
for _exts in _ALLOWED_EXTENSIONS.values():
    _ALL_ALLOWED |= _exts


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint with optional session_id."""
    pipeline = get_agent_pipeline()
    session_id = getattr(req, "session_id", None) or "default"

    # Ensure session exists in Supabase (auto-create with first user message as title)
    chat_store.create_session(session_id, title=req.query[:30] or "New Chat")

    result = pipeline.run(query=req.query, session_id=session_id)
    return ChatResponse(
        answer=result.answer,
        debug_trace=result.debug_trace or None,
        evidence_used=result.evidence_used,
    )


# ------------------------------------------------------------------
# Chat history (Supabase)
# ------------------------------------------------------------------

@router.get("/chat/sessions", response_model=SessionListResponse)
def get_sessions() -> SessionListResponse:
    """List all chat sessions, ordered by last active."""
    sessions = chat_store.list_sessions()
    return SessionListResponse(
        sessions=[SessionInfo(**s) for s in sessions],
    )


@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
def get_history(session_id: str) -> ChatHistoryResponse:
    """Get all messages for a session."""
    messages = chat_store.get_messages(session_id)
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.delete("/chat/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a session and all its messages."""
    ok = chat_store.delete_session(session_id)
    return {"success": ok, "session_id": session_id}


# ------------------------------------------------------------------
# Knowledge base management
# ------------------------------------------------------------------

@router.post("/knowledge/upload", response_model=UploadResponse)
async def upload_knowledge(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Upload files to the knowledge base, then rebuild the index.

    Currently supports: .md
    Designed to be extended for images, PDFs, etc.
    """
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    skipped: List[str] = []

    for f in files:
        filename = f.filename or "unknown"
        ext = Path(filename).suffix.lower()

        if ext not in _ALL_ALLOWED:
            skipped.append(filename)
            logger.warning("upload: skipped unsupported file type: %s", filename)
            continue

        # Save file to data/docs/
        dest = docs_dir / filename
        content = await f.read()
        dest.write_bytes(content)
        saved.append(filename)
        logger.info("upload: saved %s (%d bytes)", filename, len(content))

    if not saved:
        return UploadResponse(
            success=False,
            message=f"No supported files uploaded. Skipped: {', '.join(skipped) or 'none'}. "
                    f"Currently supported: {', '.join(sorted(_ALL_ALLOWED))}",
            files_saved=[],
            chunks_indexed=0,
        )

    # Rebuild index in a thread to avoid blocking the async event loop
    chunks_count = await asyncio.to_thread(rebuild_index)

    # Reset pipeline singleton so next /chat request picks up the new index
    reset_agent_pipeline()

    msg = f"Uploaded {len(saved)} file(s), indexed {chunks_count} chunks."
    if skipped:
        msg += f" Skipped unsupported: {', '.join(skipped)}."

    return UploadResponse(
        success=True,
        message=msg,
        files_saved=saved,
        chunks_indexed=chunks_count,
    )


@router.get("/knowledge/list", response_model=KnowledgeListResponse)
def list_knowledge() -> KnowledgeListResponse:
    """List all files currently in the knowledge base."""
    docs_dir = Path(settings.docs_dir)
    if not docs_dir.exists():
        return KnowledgeListResponse(files=[], total=0)

    items: List[KnowledgeFile] = []
    for p in sorted(docs_dir.rglob("*")):
        if not p.is_file():
            continue
        stat = p.stat()
        items.append(
            KnowledgeFile(
                filename=str(p.relative_to(docs_dir)),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                .strftime("%Y-%m-%d %H:%M"),
            )
        )

    return KnowledgeListResponse(files=items, total=len(items))


# ------------------------------------------------------------------
# Evaluation endpoint
# ------------------------------------------------------------------

@router.post("/eval/query", response_model=EvalResponse)
def eval_query(req: EvalRequest) -> EvalResponse:
    """Evaluation endpoint: returns raw retrieved chunks + model answer.

    Unlike /chat, this endpoint:
    - Skips short-term/long-term memory (pure RAG evaluation)
    - Returns the raw retrieved chunk texts and scores
    - Does NOT save to memory (no side effects)
    """
    from app.agent.context_builder import (
        SYSTEM_PROMPT,
        build_final_context,
        filter_retrieved_docs,
        parse_llm_response,
    )

    pipeline = get_agent_pipeline()
    kb = pipeline.knowledge_base

    # 1. Retrieve + rerank with requested top_k (not limited by settings.rag_top_k)
    candidates = kb.retrieve(query=req.question, top_n=req.top_k)

    # 2. Get top_k results
    top_k_results = candidates[: req.top_k]

    retrieved_context = [chunk.text for chunk, _ in top_k_results]
    retrieved_scores = [round(float(score), 4) for _, score in top_k_results]

    # 3. Filter for LLM context (normal threshold)
    filtered = filter_retrieved_docs(candidates, score_threshold=0.3, max_docs=5)
    evidence_texts = [chunk.text for chunk, _ in filtered]

    # 4. Generate answer via LLM
    final_context = build_final_context(
        query=req.question,
        evidence_texts=evidence_texts,
    )
    raw = pipeline.llm.generate_with_context(
        system_prompt=SYSTEM_PROMPT,
        user_message=final_context,
    )
    _, final_answer, _ = parse_llm_response(raw)

    return EvalResponse(
        answer=final_answer,
        retrieved_context=retrieved_context,
        retrieved_scores=retrieved_scores,
    )
