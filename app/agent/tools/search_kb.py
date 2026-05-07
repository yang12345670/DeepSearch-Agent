"""search_knowledge_base tool — wraps the existing Orchestrator as a single tool.

The Orchestrator runs Planner (mode + query rewrite) → RAG Agent
(retrieve + rerank + verify + refine) → Summarizer. From the LLM's
perspective, this is a single black-box: it asks `search_knowledge_base(query)`
and gets back a synthesized answer + citations.

Each call uses an isolated session_id so the agent's short-term memory
doesn't accumulate noise across multiple ReAct rounds. The LLM in the
ReAct loop is responsible for integrating multiple search results.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from app.agent.tools import register_tool

logger = logging.getLogger(__name__)


@register_tool(
    name="search_knowledge_base",
    description=(
        "Search the finance knowledge base (SEC filings + financial news) "
        "with a natural language query. Returns a synthesized answer with "
        "supporting citations. Use this when you need factual information "
        "about a company's filings, earnings, risks, or recent news. "
        "Call multiple times with different queries to cover multiple angles "
        "(e.g. one query for risks, another for outlook)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query, e.g. "
                    "'AAPL Q1 revenue guidance' or "
                    "'NVIDIA risk factors related to China export controls'"
                ),
            },
        },
        "required": ["query"],
    },
)
def search_knowledge_base(query: str) -> Dict[str, Any]:
    # Lazy imports to avoid circular dependency on module load
    from app.agent.pipeline import get_agent_pipeline
    from app.storage import chat_store

    orch = get_agent_pipeline()
    isolated_session = f"__react_tool__{uuid.uuid4().hex[:8]}"
    # Pre-create the session so chat_store FK constraints don't fail when
    # Orchestrator.run() persists user/assistant messages internally.
    chat_store.create_session(isolated_session, title=f"react:{query[:30]}")

    result = orch.run(
        query,
        session_id=isolated_session,
        user_id="default",
        use_rag=True,
    )

    logger.info(
        "search_knowledge_base: query=%s -> %d citations, %d evidence",
        query[:60], len(result.citations), len(result.evidence_used),
    )

    return {
        "answer": result.answer,
        "citations": [
            {"id": c.id, "source": c.source, "snippet": c.text[:200]}
            for c in result.citations
        ],
        "evidence_count": len(result.evidence_used),
    }
