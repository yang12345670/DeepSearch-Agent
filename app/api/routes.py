# -*- coding: utf-8 -*-
"""HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from app.agent.pipeline import get_agent_pipeline, get_react_executor, reset_agent_pipeline
from app.api.schemas import (
    ArtifactItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    EvalRequest,
    EvalResponse,
    KnowledgeFile,
    KnowledgeListResponse,
    SessionInfo,
    SessionListResponse,
    ToolCallSummary,
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
    """Chat endpoint — dispatches to orchestrator (legacy) or react (tool-use)."""
    session_id = getattr(req, "session_id", None) or "default"
    user_id = getattr(req, "user_id", None) or "default"
    mode = (getattr(req, "mode", None) or "orchestrator").lower()

    # Ensure session exists (auto-create with first user message as title)
    chat_store.create_session(session_id, title=req.query[:30] or "New Chat")

    if mode == "react":
        executor = get_react_executor()
        result = executor.run(req.query)

        # Persist user query + final answer to chat history (skip ReAct internals)
        chat_store.save_message(session_id, "user", req.query)
        chat_store.save_message(session_id, "assistant", result.answer)

        return ChatResponse(
            answer=result.answer,
            debug_trace=result.debug_trace or None,
            evidence_used=[],
            citations=[],
            mode="react",
            artifacts=[ArtifactItem(**a) for a in result.artifacts],
            tool_calls=[
                ToolCallSummary(
                    name=tc.name,
                    arguments=tc.arguments,
                    error=tc.error,
                )
                for tc in result.tool_calls
            ],
        )

    # Default: legacy orchestrator path
    pipeline = get_agent_pipeline()
    use_rag = getattr(req, "use_rag", True)
    result = pipeline.run(
        query=req.query, session_id=session_id, user_id=user_id, use_rag=use_rag,
    )
    citations = [
        {"id": c.id, "source": c.source, "text": c.text}
        for c in getattr(result, "citations", [])
    ]
    return ChatResponse(
        answer=result.answer,
        debug_trace=result.debug_trace or None,
        evidence_used=result.evidence_used,
        citations=citations,
        mode="orchestrator",
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE streaming chat endpoint.

    Event types:
      - ``evidence``: retrieved evidence list (sent once before generation)
      - ``token``:     incremental LLM token
      - ``done``:      generation complete, carries debug_trace
      - ``error``:     on failure
    """
    from app.agent.context_builder import (
        SYSTEM_PROMPT,
        build_final_context,
        filter_retrieved_docs,
        parse_llm_response,
    )
    from app.agent.orchestrator import _CHAT_ONLY_SYSTEM_PROMPT

    pipeline = get_agent_pipeline()
    session_id = getattr(req, "session_id", None) or "default"
    user_id = getattr(req, "user_id", None) or "default"
    use_rag = getattr(req, "use_rag", True)

    chat_store.create_session(session_id, title=req.query[:30] or "New Chat")

    def event_generator():
        try:
            if not use_rag:
                # Chat-only branch: no retrieval, no evidence event.
                short_term = pipeline._recall_short_term(session_id)
                long_term = pipeline._recall_long_term(req.query, user_id)
                parts = []
                if long_term.strip():
                    parts.append("## 长期记忆\n" + long_term.strip())
                if short_term.strip():
                    parts.append("## 对话历史\n" + short_term.strip())
                parts.append("## 用户问题\n" + req.query.strip())
                user_message = "\n\n".join(parts)

                yield _sse("evidence", {"evidence_used": [], "citations": []})

                raw_parts = []
                for token in pipeline.llm.stream_generate(
                    system_prompt=_CHAT_ONLY_SYSTEM_PROMPT,
                    user_message=user_message,
                ):
                    raw_parts.append(token)
                    yield _sse("token", {"text": token})

                final_answer = "".join(raw_parts).strip()
                chat_store.save_message(session_id, "user", req.query)
                chat_store.save_message(session_id, "assistant", final_answer)
                pipeline.memory.save_message(session_id, role="user", content=req.query)
                pipeline.memory.save_message(session_id, role="assistant", content=final_answer)

                yield _sse("done", {
                    "debug_trace": "chat-only mode (RAG disabled)",
                    "evidence_used": [],
                })
                return

            # 1. Retrieve + filter (same as rag_answer)
            kb = pipeline.knowledge_base
            candidates = kb.retrieve(req.query)
            filtered = filter_retrieved_docs(
                candidates, score_threshold=0.1, max_docs=5,
            )
            evidence_texts = [chunk.text for chunk, _ in filtered]

            # 2. Build citations from filtered chunks
            citations = [
                {"id": i + 1, "source": chunk.metadata.get("source", "unknown"), "text": chunk.text[:200].strip()}
                for i, (chunk, _) in enumerate(filtered)
            ]

            # 3. Send evidence + citations to client first
            yield _sse("evidence", {"evidence_used": evidence_texts, "citations": citations})

            # 3. Build context
            final_context = build_final_context(
                query=req.query,
                evidence_texts=evidence_texts,
            )

            # 4. Stream LLM tokens
            raw_parts = []
            for token in pipeline.llm.stream_generate(
                system_prompt=SYSTEM_PROMPT,
                user_message=final_context,
            ):
                raw_parts.append(token)
                yield _sse("token", {"text": token})

            # 5. Parse the full response for debug_trace
            raw = "".join(raw_parts)
            debug_trace, final_answer, evidence_from_llm = parse_llm_response(raw)

            # 6. Save to memory / chat store (embed citations as metadata)
            chat_store.save_message(session_id, "user", req.query)
            citations_tag = ""
            if citations:
                citations_tag = "\n<!--CITATIONS:" + json.dumps(citations, ensure_ascii=False) + "-->"
            chat_store.save_message(session_id, "assistant", final_answer + citations_tag)

            yield _sse("done", {
                "debug_trace": debug_trace,
                "evidence_used": evidence_from_llm or evidence_texts,
            })
        except Exception as e:
            logger.exception("Stream error")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    """Format one SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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


@router.patch("/chat/sessions/{session_id}")
def rename_session(session_id: str, body: dict) -> dict:
    """Rename a session."""
    title = body.get("title", "").strip()
    if not title:
        return {"success": False, "error": "title is required"}
    chat_store.update_session(session_id, title=title)
    return {"success": True, "session_id": session_id}


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
    filtered = filter_retrieved_docs(candidates, score_threshold=0.1, max_docs=5, score_gap_ratio=0.4)
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
