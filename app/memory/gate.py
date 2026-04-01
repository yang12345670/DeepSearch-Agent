# -*- coding: utf-8 -*-
"""Memory gate — quality scoring for long-term memory candidates.

Scoring criteria (each worth 1 point):
  1. Is it a technical decision?          (技术决策)
  2. Is it system design knowledge?       (系统设计知识)
  3. Is it NOT a one-time question?       (非一次性问题)

Score range: 0 ~ 3.  Threshold default = 1.
Candidates scoring below threshold are rejected.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of memory gate evaluation."""
    passed: bool
    score: int
    is_technical_decision: bool
    is_system_design: bool
    is_persistent: bool  # True = NOT a one-time question


# --- Keyword lists for rule-based scoring ---

_TECHNICAL_DECISION_KEYWORDS = [
    "选择", "决定", "采用", "使用", "切换", "迁移", "替换", "升级",
    "偏好", "配置", "设置", "策略", "方案", "架构",
    "choose", "decide", "adopt", "switch", "migrate", "prefer",
    "tradeoff", "trade-off",
]

_SYSTEM_DESIGN_KEYWORDS = [
    "架构", "设计", "模块", "流水线", "pipeline", "系统",
    "RAG", "retrieval", "embedding", "FAISS", "reranker", "向量",
    "Memory", "MCP", "transformer", "agent", "LLM",
    "索引", "检索", "融合", "重排", "门控", "去重",
    "微服务", "分布式", "缓存", "队列", "数据库",
]

_ONETIME_PATTERNS = [
    r"^你好",
    r"^hello",
    r"^hi\b",
    r"^谢谢",
    r"^thanks",
    r"^测试",
    r"^test\b",
    r"今天天气",
    r"几点了",
    r"^ok\b",
    r"^好的$",
]


def score_memory(content: str) -> GateResult:
    """Score a single memory candidate on 3 criteria.

    Returns GateResult with score 0~3.
    """
    text = content.lower()

    # Criterion 1: technical decision
    is_tech = any(kw.lower() in text for kw in _TECHNICAL_DECISION_KEYWORDS)

    # Criterion 2: system design knowledge
    is_design = any(kw.lower() in text for kw in _SYSTEM_DESIGN_KEYWORDS)

    # Criterion 3: NOT a one-time question (persistent value)
    is_onetime = any(re.search(pat, content, re.IGNORECASE) for pat in _ONETIME_PATTERNS)
    is_persistent = not is_onetime

    score = int(is_tech) + int(is_design) + int(is_persistent)

    return GateResult(
        passed=(score >= 1),  # threshold applied by gate_filter
        score=score,
        is_technical_decision=is_tech,
        is_system_design=is_design,
        is_persistent=is_persistent,
    )


def gate_filter(
    candidates: List[Dict[str, Any]],
    *,
    threshold: int = 1,
) -> List[Dict[str, Any]]:
    """Filter memory candidates through the quality gate.

    Args:
        candidates: list of candidate dicts with "content" key.
        threshold: minimum score to pass (0~3, default 1).

    Returns:
        Candidates that passed the gate, with gate_score added to metadata.
    """
    passed: List[Dict[str, Any]] = []

    for c in candidates:
        content = str(c.get("content", "")).strip()
        if not content:
            continue

        result = score_memory(content)

        if result.score >= threshold:
            # Inject gate metadata
            meta = dict(c.get("metadata", {}))
            meta["gate_score"] = result.score
            meta["gate_tech_decision"] = result.is_technical_decision
            meta["gate_system_design"] = result.is_system_design
            meta["gate_persistent"] = result.is_persistent
            c["metadata"] = meta
            passed.append(c)
            logger.debug(
                "GATE PASS [%d/3]: %s  (tech=%s design=%s persist=%s)",
                result.score, content[:50],
                result.is_technical_decision, result.is_system_design, result.is_persistent,
            )
        else:
            logger.debug(
                "GATE REJECT [%d/3]: %s  (tech=%s design=%s persist=%s)",
                result.score, content[:50],
                result.is_technical_decision, result.is_system_design, result.is_persistent,
            )

    logger.info("Memory gate: %d/%d candidates passed (threshold=%d)",
                len(passed), len(candidates), threshold)
    return passed
