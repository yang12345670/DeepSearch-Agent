# -*- coding: utf-8 -*-
"""Agent pipeline.

Single-pass RAG flow: retrieve -> filter -> build context -> LLM answer.

Memory flow:
  Store: LLM extract → gate → dedup → structure → embed → FAISS persist
  Recall: query → embed → FAISS search → filter by type → format
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agent.context_builder import RAGResult, rag_answer
from app.llm.client import LLMClient
from app.memory.extractor import extract_high_value_memories
from app.memory.gate import gate_filter
from app.memory.long_term import LongTermMemory
from app.storage import chat_store
from app.memory.short_term import ShortTermMemory
from app.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

SEED_DOCUMENTS = [
    "DeepSearch Agent \u662f\u4e00\u4e2a\u7ed3\u5408\u68c0\u7d22\u4e0e\u751f\u6210\u7684\u95ee\u7b54\u7cfb\u7edf\u3002",
    "RAG \u5178\u578b\u6d41\u7a0b\u5305\u62ec\u6587\u6863\u5207\u5206\u3001\u68c0\u7d22\u3001\u878d\u5408\u91cd\u6392\u548c\u5927\u6a21\u578b\u56de\u7b54\u3002",
    "FastAPI \u53ef\u4ee5\u5feb\u901f\u642d\u5efa Python Web API\uff0c\u5e76\u901a\u8fc7 /chat \u63a5\u53e3\u63d0\u4f9b\u670d\u52a1\u3002",
]


class AgentPipeline:
    """Single-turn agent -- single-pass evidence-grounded answering."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.llm = LLMClient()
        self.memory = ShortTermMemory(max_rounds=8)
        self.long_term_memory = LongTermMemory()
        self.knowledge_base = knowledge_base

    def run(
        self,
        query: str,
        *,
        session_id: str = "default",
        user_id: str = "default",
    ) -> RAGResult:
        """Run one /chat turn.

        Flow:
          1. Recall long-term memory (FAISS search + type filter)
          2. Recall short-term memory (filtered: history + traces + task state)
          3. RAG answer (retrieve -> filter -> build context -> generate)
          4. Save to short-term memory
          5. Extract -> Gate -> Dedup -> Store to long-term memory
        """
        # 0. Save user message BEFORE recall so current query is in context
        self.memory.save_message(session_id, role="user", content=query)
        chat_store.save_message(session_id, role="user", content=query)

        # 1. Recall long-term memory with type filtering
        recalled = self.long_term_memory.recall_memories(
            query,
            top_k=5,
            filter_types=["user_preference", "research_direction", "task_conclusion"],
            user_id=user_id if user_id != "default" else None,
        )
        long_term_context = self.long_term_memory.format_memories_for_context(recalled)

        # 2. Recall short-term memory (filtered)
        short_term_context = self.memory.get_recent_context(
            session_id=session_id,
            include_history=True,
            include_traces=True,
            include_task_state=True,
        )

        # 3. Single-pass RAG answer
        result = rag_answer(
            query=query,
            knowledge_base=self.knowledge_base,
            llm_client=self.llm,
            long_term_memory=long_term_context,
            short_term_memory=short_term_context,
        )

        # 4. Save assistant reply to short-term memory (user msg already saved in step 0)
        self.memory.save_message(session_id, role="assistant", content=result.answer)
        chat_store.save_message(session_id, role="assistant", content=result.answer)

        # 5. Long-term memory: LLM extract → gate → dedup → store
        memory_candidates = extract_high_value_memories(
            session_id=session_id,
            user_id=user_id,
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": result.answer},
            ],
            task_result=result.answer,
            llm_client=self.llm,
        )
        logger.info("Extracted %d memory candidates", len(memory_candidates))

        # Gate: score and filter
        gated = gate_filter(memory_candidates, threshold=1)
        logger.info("After gate: %d candidates", len(gated))

        # Dedup + structure + embed + persist (handled inside add_memories)
        added = self.long_term_memory.add_memories(gated)
        logger.info("Stored %d new long-term memories", len(added))

        return result


_pipeline_instance: Optional[AgentPipeline] = None


def reset_agent_pipeline() -> None:
    """Reset the singleton so the next call to get_agent_pipeline() rebuilds it."""
    global _pipeline_instance
    _pipeline_instance = None
    logger.info("Agent pipeline singleton reset.")


def get_agent_pipeline() -> AgentPipeline:
    """Singleton pipeline factory."""
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
        _pipeline_instance = AgentPipeline(kb)
    return _pipeline_instance
