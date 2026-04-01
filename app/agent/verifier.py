"""Verifier stage.

Implements structured verification on three dimensions:
- evidence_consistency
- coverage
- uncertainty
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class Verifier:
    """Verifier with both legacy and structured APIs."""

    def verify(self, contexts: List[str]) -> bool:
        """Legacy boolean verifier used by old flow."""
        return len(contexts) > 0

    def verify_answer(
        self,
        *,
        original_query: str,
        sub_questions: List[str],
        retrieved_evidence: Dict[str, List[Dict[str, Any]]],
        draft_answer: str,
    ) -> Dict[str, Any]:
        """Structured verifier required by agent pipeline.

        Args:
            original_query: original user query
            sub_questions: decomposed questions
            retrieved_evidence: mapping sub_question -> evidence list
            draft_answer: current answer text

        Returns:
            {
              "pass": true/false,
              "evidence_consistency_score": 0~1,
              "coverage_score": 0~1,
              "uncertainty_score": 0~1,
              "missing_points": [...],
              "unsupported_claims": [...],
              "suggestion": "..."
            }
        """
        _ = original_query
        if not sub_questions:
            sub_questions = [original_query] if original_query else []

        # 1) coverage: each sub-question should be addressed by draft_answer.
        missing_points: List[str] = []
        covered = 0
        answer_lower = (draft_answer or "").lower()
        for sq in sub_questions:
            sq_terms = self._extract_terms(sq)
            hit = any(term.lower() in answer_lower for term in sq_terms[:4]) if sq_terms else False
            if hit:
                covered += 1
            else:
                missing_points.append(sq)

        coverage_score = (covered / len(sub_questions)) if sub_questions else 0.0

        # 2) evidence_consistency: answer terms should overlap with evidence terms.
        evidence_texts = []
        for sq in sub_questions:
            for item in retrieved_evidence.get(sq, []):
                t = str(item.get("text", "")).strip()
                if t:
                    evidence_texts.append(t)

        evidence_blob = "\n".join(evidence_texts)
        evidence_terms = set(self._extract_terms(evidence_blob))
        answer_terms = self._extract_terms(draft_answer)

        if not evidence_terms or not answer_terms:
            evidence_consistency_score = 0.0
        else:
            supported = sum(1 for t in answer_terms if t in evidence_terms)
            evidence_consistency_score = min(1.0, supported / max(1, len(answer_terms)))

        # 3) uncertainty: detect hedging/uncertain wording.
        uncertainty_markers = [
            "可能",
            "大概",
            "不确定",
            "也许",
            "猜测",
            "无法确认",
            "might",
            "maybe",
            "uncertain",
        ]
        if not draft_answer:
            uncertainty_score = 1.0
        else:
            marker_hits = sum(1 for m in uncertainty_markers if m in draft_answer)
            uncertainty_score = min(1.0, marker_hits / 4.0)

        # unsupported claims: rough heuristic from low evidence consistency.
        unsupported_claims: List[str] = []
        if evidence_consistency_score < 0.5 and draft_answer.strip():
            unsupported_claims.append("部分结论与证据重合度较低，可能存在无证据支撑的断言。")
        if not evidence_texts:
            unsupported_claims.append("未提供可用证据，答案无法被证据验证。")

        # pass rule from requirement:
        # if evidence insufficient OR coverage insufficient => pass=false
        evidence_enough = len(evidence_texts) > 0 and evidence_consistency_score >= 0.5
        coverage_enough = coverage_score >= 0.8
        passed = evidence_enough and coverage_enough

        suggestion_parts: List[str] = []
        if not evidence_enough:
            suggestion_parts.append("补充更相关证据后再回答，或显式标注证据不足。")
        if not coverage_enough:
            suggestion_parts.append("逐一覆盖每个子问题，避免漏答。")
        if uncertainty_score > 0.6:
            suggestion_parts.append("减少不确定措辞，改为基于证据的确定表述。")
        suggestion = " ".join(suggestion_parts) if suggestion_parts else "答案可通过当前验证。"

        return {
            "pass": passed,
            "evidence_consistency_score": float(round(evidence_consistency_score, 4)),
            "coverage_score": float(round(coverage_score, 4)),
            "uncertainty_score": float(round(uncertainty_score, 4)),
            "missing_points": missing_points,
            "unsupported_claims": unsupported_claims,
            "suggestion": suggestion,
        }

    @staticmethod
    def _extract_terms(text: str) -> List[str]:
        """Simple multilingual term extraction for heuristic matching."""
        if not text:
            return []
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{2,}", text.lower())
        # keep order + dedupe
        seen = set()
        out: List[str] = []
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out[:80]

