"""Planner stage.

For the minimal runnable skeleton, planner returns a plan where each
question maps to a single sub-question (no complex decomposition yet).
"""

from __future__ import annotations

from typing import Any, Dict, List


def plan_query(query: str) -> Dict[str, Any]:
    """Return planner JSON.

    Schema:
    {
      "original_query": "...",
      "sub_questions": ["..."],
      "strategy": "single"
    }
    """
    q = (query or "").strip()
    marker = "当前用户问题："
    if marker in q:
        q = q.split(marker)[-1].strip()
    if not q:
        return {"original_query": "", "sub_questions": [], "strategy": "empty"}
    return {"original_query": q, "sub_questions": [q], "strategy": "single"}


class Planner:
    """Planner class wrapper."""

    def make_plan(self, query: str) -> Dict[str, Any]:
        return plan_query(query)

