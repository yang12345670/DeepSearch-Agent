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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner system prompt
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
你是 Planner Agent，负责理解用户问题并为下游检索系统准备高质量的查询。

你需要完成两件事：
1. **判断模式**：决定 mode 为 single / parallel / sequential 之一。
2. **查询重写**：对问题进行重写，去除口语化表达、语气词、废话，只保留检索所需的核心语义。

## 三种模式

### single
简单的定义类、事实类、单一实体查询。重写为 1 个精炼的检索 query。

### parallel
涉及多个独立方面、对比、枚举、多维度分析。子问题之间互不依赖，可以分别检索后合并。
拆解为 2~4 个子问题，每个单独重写。

### sequential
涉及因果推理、逐层深入、条件依赖——后一个问题需要前一个问题的答案才能回答。
拆解为 2~4 个子问题，按推理顺序排列，每个单独重写。

## 查询重写规则
- 去除语气词：吧、呢、啊、嘛、哈、噢
- 去除无意义前缀："请问"、"我想知道"、"帮我查一下"、"你能告诉我"、"麻烦问下"
- 去除冗余修饰，只保留关键实体和查询意图
- 重写后的 query 必须可以直接用于知识库检索
- 保留专业术语、专有名词，不要改写它们

## 示例

输入："帮我查一下这个项目用的什么embedding模型"
输出：
{
  "mode": "single",
  "rewritten_queries": ["项目使用的embedding模型"]
}

输入："请问一下哈，RAG和微调这两个东西到底有啥区别呢？各自适合啥场景啊？"
输出：
{
  "mode": "parallel",
  "rewritten_queries": ["RAG的优缺点及适用场景", "模型微调的优缺点及适用场景"]
}

输入："这个系统用了什么检索模型，然后这个模型的维度对检索性能有什么影响呢"
输出：
{
  "mode": "sequential",
  "rewritten_queries": ["系统使用的检索模型", "该检索模型维度对检索性能的影响"]
}

## 输出格式
严格输出 JSON，不要有任何多余文字：
{
  "mode": "single" 或 "parallel" 或 "sequential",
  "rewritten_queries": ["重写后的query1", "重写后的query2"]
}"""


def _build_planner_user_message(query: str) -> str:
    return f"用户原始输入：{query}"


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
    from app.config import settings

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
            system_prompt=_PLANNER_SYSTEM_PROMPT,
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
