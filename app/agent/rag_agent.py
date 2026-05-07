# -*- coding: utf-8 -*-
"""RAG Agent — stateless retrieval-augmented generation.

This is Stage 2 of the three-stage multi-agent architecture:
  Stage 1: Planner Agent (planner.py)
  Stage 2: RAG Agent (this module)  ← you are here
  Stage 3: Answer Agent / Orchestrator (orchestrator.py)

RAG Agent is a stateless tool. It receives a single clean query and returns
an evidence-grounded answer. It does NOT know about:
  - Whether it's being called in single/parallel/sequential mode
  - How many sub-questions exist
  - What other RAG calls returned

Internal pipeline:
  query
    → BM25 retrieval (top-20) ─┐
    → Dense retrieval (top-20) ─┤  parallel
    → Hybrid fusion (α=0.7)  ──┘  coarse ranking
    → Cross-Encoder reranker      fine ranking
    → Score filter (≥0.1, gap ratio 0.4, max 5)
    → Token Budget context assembly
    → LLM generation
    → Response parsing
    → Verifier (evidence_consistency ≥ 0.5, coverage ≥ 0.8)
    → Refine loop (max 2 rounds if verification fails)
    → RAGResult
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.agent.context_builder import (
    RAGResult,
    Citation,
    SYSTEM_PROMPT,
    build_final_context,
    filter_retrieved_docs,
    parse_llm_response,
    count_tokens,
)
from app.agent.verifier import Verifier
from app.agent.refine import Refiner
from app.llm.client import LLMClient
from app.rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Max refinement rounds to prevent infinite loops
_MAX_REFINE_ROUNDS = 2


class RAGAgent:
    """Stateless RAG Agent: query in → evidence-grounded answer out.

    Encapsulates the full retrieval → rank → filter → generate → verify → refine pipeline.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_client: LLMClient,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.llm = llm_client
        self.verifier = Verifier()
        self.refiner = Refiner()

    def run(
        self,
        query: str,
        *,
        long_term_memory: str = "",
        short_term_memory: str = "",
        score_threshold: float = 0.1,
        max_docs: int = 5,
        exclude_chunk_ids: Optional[set] = None,
    ) -> RAGResult:
        """Execute the full RAG pipeline with verification and refinement.

        Args:
            query: clean, rewritten query from Planner Agent.
            long_term_memory: formatted long-term memory text.
            short_term_memory: formatted conversation history text.
            score_threshold: minimum reranker score.
            max_docs: maximum evidence chunks for LLM context.
            exclude_chunk_ids: chunk IDs from prior sequential steps to skip,
                so retrieval budget is spent on new information only.

        Returns:
            RAGResult with answer, debug_trace, evidence_used, citations.
        """
        current_query = query
        current_top_k = max_docs
        _exclude = exclude_chunk_ids or set()

        for round_idx in range(_MAX_REFINE_ROUNDS + 1):
            is_refine = round_idx > 0
            logger.info(
                "RAG Agent: %s (round %d/%d, query=%s, exclude=%d chunks)",
                "REFINE" if is_refine else "INITIAL",
                round_idx + 1, _MAX_REFINE_ROUNDS + 1,
                current_query[:80], len(_exclude),
            )

            # --- Retrieve: BM25 + Dense parallel → Hybrid fusion → Reranker ---
            candidates = self.knowledge_base.retrieve(current_query)

            # --- Exclude chunks already used in prior sequential steps ---
            if _exclude:
                before = len(candidates)
                candidates = [(c, s) for c, s in candidates if c.chunk_id not in _exclude]
                logger.info(
                    "RAG Agent: excluded %d prior chunks, %d → %d",
                    before - len(candidates), before, len(candidates),
                )

            # --- Score filter ---
            filtered = filter_retrieved_docs(
                candidates,
                score_threshold=score_threshold,
                max_docs=current_top_k,
            )
            evidence_texts = [chunk.text for chunk, _ in filtered]

            logger.info(
                "RAG Agent retrieve: %d candidates → %d after filter",
                len(candidates), len(filtered),
            )

            # --- Token Budget context assembly ---
            final_context = build_final_context(
                query=current_query,
                evidence_texts=evidence_texts,
                long_term_memory=long_term_memory if not is_refine else "",
                short_term_memory=short_term_memory if not is_refine else "",
            )

            # --- LLM generation ---
            raw = self.llm.generate_with_context(
                system_prompt=SYSTEM_PROMPT,
                user_message=final_context,
            )
            debug_trace, final_answer, evidence_from_llm = parse_llm_response(raw)
            evidence_used = evidence_from_llm if evidence_from_llm else evidence_texts

            logger.info(
                "RAG Agent LLM: answer=%d chars, evidence=%d items",
                len(final_answer), len(evidence_used),
            )

            # --- Build citations ---
            citations = [
                Citation(
                    id=i + 1,
                    source=chunk.metadata.get("source", "unknown"),
                    text=chunk.text[:200].strip(),
                )
                for i, (chunk, _) in enumerate(filtered)
            ]

            result = RAGResult(
                answer=final_answer,
                debug_trace=debug_trace,
                evidence_used=evidence_used,
                citations=citations,
                used_chunk_ids=[chunk.chunk_id for chunk, _ in filtered],
            )

            # --- Verify ---
            verification = self.verifier.verify_answer(
                original_query=query,
                sub_questions=[query],
                retrieved_evidence={
                    query: [{"text": chunk.text} for chunk, _ in filtered]
                },
                draft_answer=final_answer,
            )

            passed = verification.get("pass", False)
            logger.info(
                "RAG Agent verify (round %d): pass=%s, consistency=%.2f, coverage=%.2f",
                round_idx + 1, passed,
                verification.get("evidence_consistency_score", 0),
                verification.get("coverage_score", 0),
            )

            # --- Pass → return; Fail → refine or give up ---
            if passed or round_idx >= _MAX_REFINE_ROUNDS:
                if not passed:
                    logger.warning(
                        "RAG Agent: verification failed after %d rounds, returning best effort",
                        round_idx + 1,
                    )
                return result

            # --- Refine: rewrite query + expand retrieval ---
            if not self.refiner.should_refine(verification):
                return result

            refine_result = self.refiner.refine_once(
                original_query=query,
                verifier_result=verification,
                current_top_k=current_top_k,
                expanded_top_k=15,
            )
            current_query = refine_result["refined_query"]
            current_top_k = refine_result["top_k"]
            logger.info(
                "RAG Agent refine: new query=%s, top_k=%d",
                current_query[:80], current_top_k,
            )

        # Should never reach here, but just in case
        return result
