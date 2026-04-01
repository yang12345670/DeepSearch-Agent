"""Prompt templates for LLM client."""

from __future__ import annotations

from typing import Optional


def build_answer_prompt(query: str, contexts: list[str], recent_context: Optional[str] = None) -> str:
    """Build answer prompt."""
    ctx_block = "\n".join(f"- {c}" for c in contexts[:3]) if contexts else "(no context)"
    history_block = recent_context.strip() if recent_context else "(no recent history)"
    return (
        f"[Conversation history]\n{history_block}\n\n"
        f"Question: {query}\n"
        f"Context:\n{ctx_block}\n\n"
        "Answer in Chinese, grounded ONLY in the given context and history."
    )


def build_planner_prompt(original_query: str) -> str:
    """Decompose query into sub-questions."""
    return (
        "You are a planning assistant. Decompose the user's question into 2~4 focused sub-questions.\n"
        "Rules:\n"
        "1) If the question is simple, return exactly 1 sub_question identical to the original question.\n"
        "2) Otherwise, return 2 to 4 sub_questions that cover different aspects.\n"
        "3) Output ONLY valid JSON with keys: original_query, sub_questions, strategy.\n"
        "4) Do not include any extra text.\n\n"
        f"User question:\n{original_query}\n"
    )


def build_executor_prompt(original_query: str, sub_question: str) -> str:
    """Prompt for evidence-grounded sub-question answering."""
    return (
        "You are an evidence-grounded QA executor.\n"
        "Rules:\n"
        "1) You MUST answer the sub_question using ONLY the given evidence contexts.\n"
        "2) If the evidence does not support the answer, reply exactly:\n"
        "   证据不足，无法回答该子问题。\n"
        "3) Do not guess, do not fabricate.\n\n"
        f"Original query: {original_query}\n"
        f"Sub question: {sub_question}\n"
        "Answer:"
    )


def build_executor_summary_prompt(original_query: str, sub_answers: list[str]) -> str:
    """Prompt for composing final answer from sub-answers.

    Must stay evidence-grounded: if some sub-answer says evidence is insufficient,
    final answer should explicitly keep that uncertainty.
    """
    answers_block = "\n".join(f"- {a}" for a in sub_answers) if sub_answers else "(no sub answers)"
    return (
        "You are an evidence-grounded summarizer.\n"
        "Use ONLY the provided sub-answers (which were generated from evidence).\n"
        "Do not invent any new claim beyond those sub-answers.\n"
        "If any sub-answer indicates insufficient evidence, keep that limitation explicitly.\n\n"
        f"Original query: {original_query}\n"
        f"Sub answers:\n{answers_block}\n"
        "Final answer:"
    )


def build_verifier_prompt(original_query: str, sub_questions: list[str], draft_answer: str) -> str:
    """Prompt for verifying draft answer against evidence.

    The verifier should score:
    - evidence_consistency
    - coverage
    - uncertainty
    and return strict JSON.
    """
    subs = "\n".join(f"- {sq}" for sq in sub_questions) if sub_questions else "(none)"
    return (
        "You are a strict evidence-grounded verifier.\n"
        "You will be given:\n"
        "1) original_query\n"
        "2) sub_questions\n"
        "3) draft_answer\n"
        "4) EVIDENCE contexts are provided separately.\n\n"
        "Task:\n"
        "Evaluate the draft_answer with respect to the evidence contexts.\n"
        "Check these dimensions:\n"
        "1) evidence_consistency: whether claims are supported by evidence\n"
        "2) coverage: whether all sub_questions are covered\n"
        "3) uncertainty: whether there are obvious uncertain conclusions\n"
        "Do not invent facts beyond evidence.\n\n"
        "Return ONLY JSON with exactly these keys:\n"
        "{\n"
        '  "pass": true/false,\n'
        '  "evidence_consistency_score": 0~1,\n'
        '  "coverage_score": 0~1,\n'
        '  "uncertainty_score": 0~1,\n'
        '  "missing_points": [...],\n'
        '  "unsupported_claims": [...],\n'
        '  "suggestion": "..." \n'
        "}\n\n"
        f"original_query:\n{original_query}\n\n"
        f"sub_questions:\n{subs}\n\n"
        f"draft_answer:\n{draft_answer}\n"
    )


def build_refine_prompt(
    original_query: str,
    suggestion: str,
    missing_points: list[str],
    unsupported_claims: list[str],
) -> str:
    """Prompt to refine query for better evidence coverage."""
    mp = "\n".join(f"- {x}" for x in missing_points[:5]) if missing_points else "(none)"
    uc = "\n".join(f"- {x}" for x in unsupported_claims[:5]) if unsupported_claims else "(none)"
    return (
        "You are a query refiner for an evidence-grounded QA system.\n"
        "Rewrite the original_query to improve coverage and reduce unsupported claims.\n"
        "Rules:\n"
        "1) Output ONLY valid JSON with key refined_query.\n"
        "2) The refined_query should focus on missing_points and avoid unsupported_claims.\n\n"
        f"original_query:\n{original_query}\n\n"
        f"suggestion:\n{suggestion}\n\n"
        f"missing_points:\n{mp}\n\n"
        f"unsupported_claims:\n{uc}\n"
    )

