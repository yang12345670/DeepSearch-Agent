# -*- coding: utf-8 -*-
"""Agent Orchestrator — three-stage multi-agent coordination.

  Stage 1: Planner Agent   → classify mode + rewrite queries
  Stage 2: RAG Agent       → stateless retrieval + generation (with verify + refine)
  Stage 3: Summarizer      → merge sub-answers (only for parallel/sequential)

Orchestrator = dispatcher. It does not understand questions, retrieve docs,
or generate answers. It only reads Planner output, calls RAG Agent, and
collects results.

Flow by mode:
  single:     1 query  → RAG Agent → done
  parallel:   N queries → each RAG Agent independently → Summarizer → done
  sequential: N queries → RAG Agent chained (prior answer injected) → Summarizer → done
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from app.agent.context_builder import RAGResult, Citation
from app.agent.planner import Planner
from app.agent.rag_agent import RAGAgent
from app.llm.client import LLMClient
from app.memory.extractor import extract_high_value_memories
from app.memory.gate import gate_filter
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.rag.knowledge_base import KnowledgeBase
from app.storage import chat_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Summarizer prompt
# ---------------------------------------------------------------------------

_CHAT_ONLY_SYSTEM_PROMPT = """\
你是一个友好的中文对话助手。基于对话历史和长期记忆与用户自然交流。
- 回答简洁、自然，不需要任何调试标签或结构化格式。
- 对于事实性问题，如果不确定就直说不知道，不要编造。
- 不要输出 [DEBUG_TRACE] / [FINAL_ANSWER] / [EVIDENCE_USED] 等标签。"""


_SUMMARIZER_SYSTEM_PROMPT = """\
你是一个答案合并器。你会收到一个原始问题和若干子问题的回答。
请将这些子回答合并为一个完整、连贯、无重复的最终回答。

规则：
1. 保留每个子回答中有证据支持的内容，不要丢失信息。
2. 如果某个子回答说"证据不足"，在最终回答中保留这一标注。
3. 不要编造任何子回答中没有的内容。
4. 使用中文，结构清晰。
5. 如果子回答之间有矛盾，以证据更充分的为准，并标注分歧。

输出格式：
[FINAL_ANSWER]
（合并后的完整回答）
[/FINAL_ANSWER]"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """Dispatcher: Planner → RAG Agent → Summarizer."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.llm = LLMClient()
        self.memory = ShortTermMemory(max_rounds=8)
        self.long_term_memory = LongTermMemory()
        self.knowledge_base = knowledge_base
        self.planner = Planner(llm_client=self.llm)
        self.rag_agent = RAGAgent(knowledge_base, self.llm)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        session_id: str = "default",
        user_id: str = "default",
        use_rag: bool = True,
    ) -> RAGResult:
        """Full agent turn: plan → dispatch → summarize → memorize.

        If use_rag is False, skip planner/RAG and answer using conversation
        history + long-term memory only (pure chat mode).
        """

        # 0. Save user message (eviction may happen here too)
        _ok, evicted_on_user = self.memory.save_message(session_id, role="user", content=query)
        chat_store.save_message(session_id, role="user", content=query)

        # 1. Recall memory
        long_term_context = self._recall_long_term(query, user_id)
        short_term_context = self._recall_short_term(session_id)

        if not use_rag:
            # Pure chat path — no planner, no retrieval, no verifier.
            result = self._run_chat_only(query, long_term_context, short_term_context)
        else:
            # 2. Planner Agent
            plan = self.planner.make_plan(query)
            mode = plan.get("mode", "single")
            rewritten_queries = plan.get("rewritten_queries", [query])
            logger.info("Orchestrator: mode=%s, rewritten_queries=%s", mode, rewritten_queries)

            # 3. Dispatch by mode
            if mode == "single" or len(rewritten_queries) <= 1:
                result = self.rag_agent.run(
                    rewritten_queries[0] if rewritten_queries else query,
                    long_term_memory=long_term_context,
                    short_term_memory=short_term_context,
                )
            elif mode == "sequential":
                result = self._execute_sequential(
                    query, rewritten_queries, long_term_context, short_term_context,
                )
            else:  # parallel
                result = self._execute_parallel(
                    query, rewritten_queries, long_term_context, short_term_context,
                )

        # 4. Save assistant reply — eviction typically triggers here
        _ok, evicted_on_assistant = self.memory.save_message(
            session_id, role="assistant", content=result.answer,
        )
        chat_store.save_message(session_id, role="assistant", content=result.answer)

        # 5. Long-term memory persistence — only when messages were evicted
        all_evicted = evicted_on_user + evicted_on_assistant
        if all_evicted:
            self._persist_memories(session_id, user_id, all_evicted)

        return result

    # ------------------------------------------------------------------
    # Chat-only: no retrieval, just LLM with memory
    # ------------------------------------------------------------------

    def _run_chat_only(
        self,
        query: str,
        long_term_context: str,
        short_term_context: str,
    ) -> RAGResult:
        parts: List[str] = []
        if long_term_context.strip():
            parts.append("## 长期记忆\n" + long_term_context.strip())
        if short_term_context.strip():
            parts.append("## 对话历史\n" + short_term_context.strip())
        parts.append("## 用户问题\n" + query.strip())
        user_message = "\n\n".join(parts)

        try:
            answer = self.llm.generate_with_context(
                system_prompt=_CHAT_ONLY_SYSTEM_PROMPT,
                user_message=user_message,
            ).strip()
        except Exception as e:
            logger.error("Chat-only generation failed: %s", e)
            answer = "抱歉，我现在没法回答这个问题，请稍后再试。"

        return RAGResult(
            answer=answer,
            debug_trace="chat-only mode (RAG disabled)",
            evidence_used=[],
            citations=[],
        )

    # ------------------------------------------------------------------
    # Parallel: each sub-question → independent RAG Agent call
    # ------------------------------------------------------------------

    def _execute_parallel(
        self,
        original_query: str,
        rewritten_queries: List[str],
        long_term_context: str,
        short_term_context: str,
    ) -> RAGResult:
        sub_answers: List[str] = []
        all_citations: List[Citation] = []
        all_evidence: List[str] = []

        for i, rq in enumerate(rewritten_queries):
            logger.info("Parallel [%d/%d]: %s", i + 1, len(rewritten_queries), rq)
            sub_result = self.rag_agent.run(
                rq,
                long_term_memory=long_term_context if i == 0 else "",
                short_term_memory=short_term_context if i == 0 else "",
            )
            sub_answers.append(sub_result.answer)
            all_citations.extend(sub_result.citations)
            all_evidence.extend(sub_result.evidence_used)

        final_answer, debug_trace = self._summarize(
            original_query, rewritten_queries, sub_answers,
        )
        return RAGResult(
            answer=final_answer,
            debug_trace=debug_trace,
            evidence_used=all_evidence,
            citations=self._dedup_citations(all_citations),
        )

    # ------------------------------------------------------------------
    # Sequential: chained RAG Agent calls, prior answer → next query
    # ------------------------------------------------------------------

    def _execute_sequential(
        self,
        original_query: str,
        rewritten_queries: List[str],
        long_term_context: str,
        short_term_context: str,
    ) -> RAGResult:
        sub_answers: List[str] = []
        all_citations: List[Citation] = []
        all_evidence: List[str] = []
        chain_context = ""
        used_chunks: set = set()  # accumulate chunk IDs across steps

        for i, rq in enumerate(rewritten_queries):
            if chain_context:
                augmented_query = f"{rq}\n\n[前序推理结果]\n{chain_context}"
            else:
                augmented_query = rq

            logger.info(
                "Sequential [%d/%d]: %s (chain=%d chars, exclude=%d chunks)",
                i + 1, len(rewritten_queries), rq, len(chain_context), len(used_chunks),
            )

            sub_result = self.rag_agent.run(
                augmented_query,
                long_term_memory=long_term_context if i == 0 else "",
                short_term_memory=short_term_context if i == 0 else "",
                exclude_chunk_ids=used_chunks if used_chunks else None,
            )
            sub_answers.append(sub_result.answer)
            all_citations.extend(sub_result.citations)
            all_evidence.extend(sub_result.evidence_used)

            # Accumulate used chunks so next step skips them
            used_chunks.update(sub_result.used_chunk_ids)
            chain_context += f"\n步骤{i+1} ({rq}): {sub_result.answer}"

        final_answer, debug_trace = self._summarize(
            original_query, rewritten_queries, sub_answers,
        )
        return RAGResult(
            answer=final_answer,
            debug_trace=debug_trace,
            evidence_used=all_evidence,
            citations=self._dedup_citations(all_citations),
        )

    # ------------------------------------------------------------------
    # Summarizer
    # ------------------------------------------------------------------

    def _summarize(
        self,
        original_query: str,
        sub_questions: List[str],
        sub_answers: List[str],
    ) -> tuple[str, str]:
        if len(sub_answers) == 1:
            return sub_answers[0], ""

        parts = [f"原始问题：{original_query}\n"]
        for i, (sq, sa) in enumerate(zip(sub_questions, sub_answers)):
            parts.append(f"子问题{i+1}：{sq}\n回答{i+1}：{sa}\n")

        user_message = "\n".join(parts)

        try:
            raw = self.llm.generate_with_context(
                system_prompt=_SUMMARIZER_SYSTEM_PROMPT,
                user_message=user_message,
            )
            m = re.search(
                r"\[FINAL_ANSWER\]\s*\n?(.*?)\n?\s*\[/FINAL_ANSWER\]", raw, re.DOTALL,
            )
            answer = m.group(1).strip() if m else raw.strip()

            debug_trace = (
                f"Orchestrator: {len(sub_questions)} sub-questions, summarized\n"
                + "\n".join(f"  sub_q{i+1}: {sq}" for i, sq in enumerate(sub_questions))
            )
            return answer, debug_trace

        except Exception as e:
            logger.error("Summarizer failed: %s, concatenating sub-answers", e)
            combined = "\n\n".join(
                f"**{sq}**\n{sa}" for sq, sa in zip(sub_questions, sub_answers)
            )
            return combined, f"Summarizer error: {e}"

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _recall_long_term(self, query: str, user_id: str) -> str:
        recalled = self.long_term_memory.recall_memories(
            query,
            top_k=5,
            filter_types=["user_preference", "research_direction", "task_conclusion"],
            user_id=user_id if user_id != "default" else None,
        )
        return self.long_term_memory.format_memories_for_context(recalled)

    def _recall_short_term(self, session_id: str) -> str:
        ctx = self.memory.get_recent_context(
            session_id=session_id,
            include_history=True,
            include_traces=True,
            include_task_state=True,
        )
        if not ctx.strip() and session_id != "default":
            logger.warning("Short-term context empty for session %s", session_id)
        return ctx

    def _persist_memories(
        self,
        session_id: str,
        user_id: str,
        evicted_messages: List[Dict[str, str]],
    ) -> None:
        """Extract long-term memories from messages evicted by the sliding window."""
        memory_candidates = extract_high_value_memories(
            session_id=session_id,
            user_id=user_id,
            messages=evicted_messages,
            task_result=None,
            llm_client=self.llm,
        )
        gated = gate_filter(memory_candidates, threshold=1)
        added = self.long_term_memory.add_memories(gated)
        logger.info(
            "Eviction triggered: %d messages → %d candidates → %d gated → %d stored",
            len(evicted_messages), len(memory_candidates), len(gated), len(added),
        )

    @staticmethod
    def _dedup_citations(citations: List[Citation]) -> List[Citation]:
        seen = set()
        deduped: List[Citation] = []
        for c in citations:
            key = (c.source, c.text)
            if key not in seen:
                seen.add(key)
                deduped.append(Citation(id=len(deduped) + 1, source=c.source, text=c.text))
        return deduped
