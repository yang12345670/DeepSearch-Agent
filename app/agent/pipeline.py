# -*- coding: utf-8 -*-
"""RAG pipeline — pure stateless retrieval + generation tool.

This module is a TOOL for the Agent layer (orchestrator.py).
It does NOT do planning, decomposition, or multi-step reasoning.

Single-pass flow: retrieve -> filter -> build context -> LLM answer -> parse.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agent.context_builder import rag_answer, RAGResult
from app.agent.orchestrator import AgentOrchestrator
from app.llm.client import LLMClient
from app.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

SEED_DOCUMENTS = [
    "DeepSearch Agent 是一个结合检索与生成的问答系统。",
    "RAG 典型流程包括文档切分、检索、融合重排和大模型回答。",
    "FastAPI 可以快速搭建 Python Web API，并通过 /chat 接口提供服务。",
]


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
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_backend=None,
                embedding_dim=384,
                faiss_index_path=None,
            )
            from app.rag.chunker import split_documents
            from app.rag.bm25_retriever import BM25Retriever
            from app.rag.dense_retriever import DenseRetriever
            from app.rag.hybrid_retriever import HybridRetriever

            kb.chunks = split_documents(SEED_DOCUMENTS, chunk_size=500, overlap=100)
            kb.bm25 = BM25Retriever(kb.chunks)
            kb.dense = DenseRetriever(
                kb.chunks,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_dim=384,
            )
            kb.hybrid = HybridRetriever(kb.bm25, kb.dense)
        _pipeline_instance = AgentOrchestrator(kb)
    return _pipeline_instance
