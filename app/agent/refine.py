"""Refine stage.

Provides:
- should_refine(verifier_result)
- refine_once(...)

Policy:
- if missing_points or unsupported_claims exists -> refine
- rewrite query
- expand retrieval top_k (e.g. 8 -> 15)
- pipeline should cap rounds to max 2
"""

from __future__ import annotations

from typing import Any, Dict, List


def should_refine(verifier_result: Dict[str, Any]) -> bool:
    """Return True when verifier signals missing/unsupported points."""
    missing = verifier_result.get("missing_points") or []
    unsupported = verifier_result.get("unsupported_claims") or []
    return bool(missing or unsupported)


def refine_once(
    *,
    original_query: str,
    verifier_result: Dict[str, Any],
    current_top_k: int = 8,
    expanded_top_k: int = 15,
) -> Dict[str, Any]:
    """One refinement step: rewrite query + increase retrieval top_k."""
    missing = verifier_result.get("missing_points") or []
    unsupported = verifier_result.get("unsupported_claims") or []
    suggestion = str(verifier_result.get("suggestion") or "").strip()

    additions: List[str] = []
    if missing:
        additions.append("补充要点：" + "；".join(str(x) for x in missing[:3]))
    if unsupported:
        additions.append("避免无证据断言：" + "；".join(str(x) for x in unsupported[:2]))
    if suggestion:
        additions.append("修正策略：" + suggestion)

    refined_query = original_query
    if additions:
        refined_query = f"{original_query}\n\n[Refine]\n" + "\n".join(f"- {x}" for x in additions)

    next_top_k = max(int(current_top_k), int(expanded_top_k))

    return {
        "refined_query": refined_query,
        "top_k": next_top_k,
    }


class Refiner:
    """Refiner wrapper for agent pipeline."""

    def refine(self, contexts: List[str]) -> List[str]:
        """Legacy helper: keep top contexts."""
        return contexts[:3]

    def should_refine(self, verifier_result: Dict[str, Any]) -> bool:
        return should_refine(verifier_result)

    def refine_once(
        self,
        *,
        original_query: str,
        verifier_result: Dict[str, Any],
        current_top_k: int = 8,
        expanded_top_k: int = 15,
    ) -> Dict[str, Any]:
        return refine_once(
            original_query=original_query,
            verifier_result=verifier_result,
            current_top_k=current_top_k,
            expanded_top_k=expanded_top_k,
        )

