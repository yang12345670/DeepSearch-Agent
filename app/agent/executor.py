"""Executor stage.

Supports two levels:
1) simple `execute(query)` for one-shot retrieval + answer
2) structured `execute_with_plan(...)`:
   - input: original_query, sub_questions, retrieved_evidence
   - answer each sub-question based on evidence
   - aggregate to final answer
   - return JSON with used evidence ids
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.llm.client import LLMClient
from app.llm.prompts import build_executor_prompt, build_executor_summary_prompt
from app.rag.knowledge_base import KnowledgeBase  # type: ignore


class Executor:
    """Executes retrieval and generation."""

    def __init__(self, knowledge_base: KnowledgeBase, llm_client: LLMClient) -> None:
        self.knowledge_base = knowledge_base
        self.llm = llm_client

    def execute(self, query: str, *, recent_context: Optional[str] = None) -> str:
        contexts = self.knowledge_base.search(query)
        return self.llm.generate(query, contexts, recent_context=recent_context)

    def execute_with_plan(
        self,
        *,
        original_query: str,
        sub_questions: List[str],
        retrieved_evidence: Dict[str, List[Dict[str, Any]]],
        recent_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute sub-question answering from provided evidence and aggregate.

        Args:
            original_query: user's original question
            sub_questions: decomposed sub questions
            retrieved_evidence: mapping sub_question -> list of evidence items
                each item should include:
                    - "chunk_id": str (optional but recommended)
                    - "text": str

        Returns JSON:
            {
              "sub_answers": [...],
              "final_answer": "...",
              "used_evidence_ids": [...]
            }
        """
        sub_answers: List[str] = []
        used_evidence_ids: List[str] = []
        seen_ids = set()

        if not sub_questions:
            sub_questions = [original_query]

        for sq in sub_questions:
            items = retrieved_evidence.get(sq, [])
            contexts: List[str] = []
            for item in items:
                text = str(item.get("text", "")).strip()
                if text:
                    contexts.append(text)
                cid = item.get("chunk_id")
                if isinstance(cid, str) and cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    used_evidence_ids.append(cid)

            # Enforce "no fabrication": explicit insufficiency when evidence is absent
            if not contexts:
                sub_answers.append(f"子问题：{sq}\n证据不足，无法回答该子问题。")
                continue

            prompt = build_executor_prompt(original_query, sq)
            sub_answer = self.llm.generate(prompt, contexts, recent_context=recent_context)
            sub_answers.append(sub_answer)

        # Aggregate final answer from sub-answers only.
        summary_prompt = build_executor_summary_prompt(original_query, sub_answers)
        # Pass sub_answers as "contexts" so fallback LLM remains evidence-grounded.
        final_answer = self.llm.generate(summary_prompt, sub_answers, recent_context=recent_context)

        return {
            "sub_answers": sub_answers,
            "final_answer": final_answer,
            "used_evidence_ids": used_evidence_ids,
        }

    def generate_from_evidence(
        self,
        original_query: str,
        evidence_contexts: List[str],
        *,
        recent_context: Optional[str] = None,
    ) -> str:
        """Generate final answer from aggregated evidence contexts."""
        return self.llm.generate(original_query, evidence_contexts, recent_context=recent_context)

