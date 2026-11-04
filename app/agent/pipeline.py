# -*- coding: utf-8 -*-
"""Shared factories for the orchestrator and ReAct runtime."""

from __future__ import annotations

import logging
from typing import Optional

from app.agent.orchestrator import AgentOrchestrator
from app.config import settings
from app.llm.client import LLMClient
from app.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

_pipeline_instance: Optional[AgentOrchestrator] = None
_react_executor_instance = None


def reset_agent_pipeline() -> None:
    """Reset the singleton so the next call to get_agent_pipeline() rebuilds it."""
    global _pipeline_instance, _react_executor_instance
    _pipeline_instance = None
    _react_executor_instance = None
    logger.info("Agent pipeline singleton reset.")


def get_react_executor():
    """Singleton factory — returns the ReActExecutor (LLM-driven tool-use loop)."""
    global _react_executor_instance
    if _react_executor_instance is None:
        # Lazy import to avoid circulars when only orchestrator path is used
        from app.agent.react_executor import ReActExecutor
        _react_executor_instance = ReActExecutor(LLMClient())
        logger.info("ReActExecutor singleton initialized.")
    return _react_executor_instance


def get_agent_pipeline() -> AgentOrchestrator:
    """Singleton factory — returns the AgentOrchestrator (which uses RAG as a tool)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        kb = KnowledgeBase.from_persisted_index()
        if kb is None:
            kb = KnowledgeBase(
                chunks=[],
                embedding_model_name=settings.embedding_model_name,
                embedding_backend=None,
                embedding_dim=settings.embedding_dim,
                faiss_index_path=None,
            )
            logger.info("No persisted index found; using an empty knowledge base.")
        _pipeline_instance = AgentOrchestrator(kb)
    return _pipeline_instance
