# -*- coding: utf-8 -*-
"""Long-term memory extractor — LLM-based.

Flow: conversation messages → LLM extraction → candidate list.

Replaces the old regex-only approach with LLM-driven extraction that
identifies three memory types:
  - user_preference:      用户偏好（阅读习惯、关注领域、交互风格）
  - research_direction:   用户研究方向（RAG、Memory、MCP、transformer 等）
  - task_conclusion:      重要任务结论
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from app.config import settings

logger = logging.getLogger(__name__)

MemoryType = Literal["user_preference", "research_direction", "task_conclusion"]

# --- LLM extraction prompt ---

_EXTRACT_SYSTEM_PROMPT = settings.memory_extract_system_prompt

_EXTRACT_USER_TEMPLATE = settings.memory_extract_user_template


def extract_high_value_memories(
    *,
    session_id: str,
    user_id: str = "default",
    messages: List[Dict[str, str]],
    task_result: Optional[str] = None,
    llm_client=None,
) -> List[Dict[str, Any]]:
    """Extract memory candidates via LLM, with regex fallback.

    Args:
        session_id: current session id.
        user_id: user identifier for structured storage.
        messages: list of {"role": ..., "content": ...} from this turn.
        task_result: the final answer produced this turn.
        llm_client: LLMClient instance. If None or LLM fails, uses regex fallback.

    Returns:
        List of candidate dicts ready for gate → dedup → add_memories.
    """
    candidates: List[Dict[str, Any]] = []

    # --- Try LLM extraction first ---
    if llm_client is not None:
        try:
            conversation = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )
            user_msg = settings.memory_extract_user_template.format(conversation=conversation)
            raw = llm_client.generate_with_context(
                system_prompt=settings.memory_extract_system_prompt,
                user_message=user_msg,
            )
            extracted = _parse_llm_extraction(raw)
            for item in extracted:
                mt = item.get("memory_type", "")
                content = str(item.get("content", "")).strip()
                if mt in ("user_preference", "research_direction", "task_conclusion") and content:
                    candidates.append({
                        "memory_type": mt,
                        "content": content,
                        "metadata": {
                            "session_id": session_id,
                            "user_id": user_id,
                            "source": "llm_extraction",
                        },
                    })
            if candidates:
                logger.info("LLM extracted %d memory candidates: %s",
                            len(candidates), [c["content"][:40] for c in candidates])
                return candidates
        except Exception as e:
            logger.warning("LLM extraction failed, falling back to regex: %s", e)

    # --- Regex fallback ---
    candidates = _regex_fallback(
        messages=messages,
        task_result=task_result,
        session_id=session_id,
        user_id=user_id,
    )
    logger.info("Regex fallback extracted %d memory candidates", len(candidates))
    return candidates


def _parse_llm_extraction(raw: str) -> List[Dict[str, Any]]:
    """Parse JSON array from LLM output, tolerant of surrounding text."""
    # Find JSON array in the output
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _regex_fallback(
    *,
    messages: List[Dict[str, str]],
    task_result: Optional[str],
    session_id: str,
    user_id: str,
) -> List[Dict[str, Any]]:
    """Regex-based extraction as fallback when LLM is unavailable."""
    candidates: List[Dict[str, Any]] = []
    meta = {"session_id": session_id, "user_id": user_id, "source": "regex_fallback"}

    # --- user_preference patterns ---
    pref_patterns = [
        (r"记住[:：]\s*(.+)", "user_preference"),
        (r"我的偏好是[:：]?\s*(.+)", "user_preference"),
        (r"我偏好(.+)", "user_preference"),
        (r"我喜欢(.+)", "user_preference"),
    ]
    # --- research_direction patterns ---
    research_patterns = [
        (r"我(?:在)?(?:研究|关注|学习|探索)(.+)", "research_direction"),
        (r"我更关注(.+)", "research_direction"),
        (r"我对(.+?)(?:感兴趣|很关注|在研究)", "research_direction"),
    ]

    all_patterns = pref_patterns + research_patterns

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        for pat, mtype in all_patterns:
            m = re.search(pat, content)
            if m:
                extracted = m.group(1).strip(" \u3002,.!\uff01?\uff1f")
                if extracted:
                    candidates.append({
                        "memory_type": mtype,
                        "content": extracted,
                        "metadata": dict(meta),
                    })
                break

    # --- task_conclusion: extract from assistant messages ---
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        result_text = str(msg.get("content", "")).strip()
        if result_text and result_text != "证据不足，无法回答该问题。":
            limit = settings.memory_task_summary_chars
            summary = result_text if len(result_text) <= limit else result_text[:limit] + "..."
            candidates.append({
                "memory_type": "task_conclusion",
                "content": summary,
                "metadata": dict(meta, source="evicted_assistant_msg"),
            })

    return candidates
