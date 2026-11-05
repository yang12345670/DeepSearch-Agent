# -*- coding: utf-8 -*-
"""Planner Agent — query understanding, rewriting, and decomposition.

Three-stage multi-agent architecture:
  Stage 1: Planner Agent (this module)
    - Classify query mode: single / parallel / sequential
    - Query Rewrite: strip filler, extract retrieval intent
    - Output: clean, search-ready queries for the RAG Agent
  Stage 2: RAG Agent (context_builder.py)
  Stage 3: Answer Agent (orchestrator.py)

Planner output contract:
  {
    "original_query": str,
    "mode": "single" | "parallel" | "sequential",
    "rewritten_queries": [str, ...],
  }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = settings.planner_system_prompt


def _build_planner_user_message(query: str) -> str:
    return settings.planner_user_template.format(query=query)


# ---------------------------------------------------------------------------
# Rule-based fallback (no LLM)
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    re.compile(r"^(请问|我想知道|帮我查一下|你能告诉我|麻烦问下|请帮我|我想了解一下|我想问一下)\s*[，,]?\s*"),
    re.compile(r"[吧呢啊嘛哈噢哦呀]+([？?。，,；;]|$)"),
    re.compile(r"(到底|究竟|具体来说|一下下?|这个那个)"),
    re.compile(r"^(嗯+|额+|那个+)\s*[，,]?\s*"),
]

_COMPLEX_PATTERNS = [
    re.compile(r"和|与|以及|还有"),
    re.compile(r"对比|比较|区别|不同|异同"),
    re.compile(r"为什么.*怎么|怎么.*为什么"),
    re.compile(r"哪些.*分别|分别.*哪些"),
    re.compile(r"优缺点|利弊|优势.*劣势"),
    re.compile(r"compare|difference|vs\.?|versus", re.IGNORECASE),
]

_SEQUENTIAL_PATTERNS = [
    re.compile(r"然后|接着|之后|进而|从而|基于此"),
    re.compile(r"为什么.*怎么办|原因.*影响"),
    re.compile(r"是什么.*有什么用|是什么.*怎么影响"),
]


def _rule_based_rewrite(query: str) -> str:
    """Strip filler words and noise from query."""
    q = query.strip()
    for pattern in _FILLER_PATTERNS:
        q = pattern.sub("", q)
    q = re.sub(r"^[，,。.、\s]+", "", q)
    q = re.sub(r"[，,。.、\s]+$", "", q)
    return q.strip() or query.strip()


def _rule_based_plan(query: str) -> Dict[str, Any]:
    """Heuristic fallback when LLM is unavailable."""
    q = query.strip()
    if not q:
        return {"original_query": "", "mode": "single", "rewritten_queries": []}

    is_complex = any(p.search(q) for p in _COMPLEX_PATTERNS) and len(q) > 10
    if is_complex:
        parts = re.split(r"[，,；;]|(?:和|与|以及)", q)
        parts = [_rule_based_rewrite(p) for p in parts if len(p.strip()) > 4]
        if 2 <= len(parts) <= 4:
            is_sequential = any(p.search(q) for p in _SEQUENTIAL_PATTERNS)
            return {
                "original_query": q,
                "mode": "sequential" if is_sequential else "parallel",
                "rewritten_queries": parts,
            }

    return {
        "original_query": q,
        "mode": "single",
        "rewritten_queries": [_rule_based_rewrite(q)],
    }


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_planner_response(raw: str, original_query: str) -> Dict[str, Any]:
    """Parse LLM JSON response with robust fallback."""
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            queries = data.get("rewritten_queries", [])
            mode = data.get("mode", "single")

            if isinstance(queries, list) and 1 <= len(queries) <= 6:
                queries = [str(s).strip() for s in queries if str(s).strip()]
                if queries:
                    return {
                        "original_query": original_query,
                        "mode": mode if mode in ("single", "parallel", "sequential") else "single",
                        "rewritten_queries": queries,
                    }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    logger.warning("Failed to parse planner response, falling back to rule-based: %.200s", raw)
    return _rule_based_plan(original_query)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def plan_query(query: str, llm_client=None) -> Dict[str, Any]:
    """Planner Agent: classify + rewrite + decompose.

    Returns:
        {
            "original_query": str,
            "mode": "single" | "parallel" | "sequential",
            "rewritten_queries": [str, ...],
        }
    """
    q = (query or "").strip()
    marker = "当前用户问题："
    if marker in q:
        q = q.split(marker)[-1].strip()
    if not q:
        return {"original_query": "", "mode": "single", "rewritten_queries": []}

    if not settings.planner_enabled or llm_client is None:
        logger.info("Planner Agent: LLM disabled, using rule-based fallback")
        return _rule_based_plan(q)

    try:
        raw = llm_client.generate_with_context(
            system_prompt=settings.planner_system_prompt,
            user_message=_build_planner_user_message(q),
        )
        logger.info("Planner Agent LLM response: %.300s", raw)
        result = _parse_planner_response(raw, q)
        logger.info(
            "Planner Agent: mode=%s, rewritten_queries=%s",
            result["mode"], result["rewritten_queries"],
        )
        return result
    except Exception as e:
        logger.warning("Planner Agent LLM call failed: %s, falling back", e)
        return _rule_based_plan(q)


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------

class Planner:
    """Planner Agent class wrapper."""

    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client

    def make_plan(self, query: str) -> Dict[str, Any]:
        return plan_query(query, llm_client=self.llm_client)
