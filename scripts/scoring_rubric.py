# -*- coding: utf-8 -*-
"""
Answer-Layer Scoring Rubric for DeepSearch Agent.

Six metrics, each with a precise formula, auto/manual classification,
and per-case-type expected behavior.

Usage:
    from scoring_rubric import Rubric, score_sample_v2

    rubric = Rubric()
    result = rubric.score(sample, model_answer, retrieved_context, retrieved_scores)
    print(result)

Or run standalone for a dry-run demo:
    python scripts/scoring_rubric.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ====================================================================
# Constants
# ====================================================================

REFUSAL_PHRASES = [
    "无法回答", "没有相关信息", "未找到", "不确定", "无法确定",
    "未提及", "没有提及", "不在知识库", "无相关", "无法判断",
    "知识库中没有", "没有足够的信息",
    "not found", "no relevant", "cannot answer", "don't have",
    "not mentioned", "no information", "unable to answer",
]

CAVEAT_PHRASES = [
    "但文档中未", "但未提及", "但没有", "但未找到",
    "不过文档中", "不过未提及", "尚未涉及", "未涵盖",
    "关于这一点文档", "文档没有涉及", "目前资料中未",
    "however", "not mentioned", "no information about",
    "not covered in", "beyond the scope",
]


# ====================================================================
# Text Helpers
# ====================================================================

def _norm(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return " ".join(text.lower().split())


def _split_claims(answer: str) -> List[str]:
    """Split answer into sentence-level claims.

    Splits on Chinese/English sentence terminators and newlines.
    Filters out empty or very short fragments.
    """
    parts = re.split(r'[。！？\n.!?;；]', answer)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def _extract_distinctive_terms(text: str, min_len: int = 4) -> List[str]:
    """Extract distinctive terms from a text for noise-leak detection."""
    return [t.strip() for t in re.split(r'[,，.。;；:：()\s]+', text)
            if len(t.strip()) >= min_len]


# ====================================================================
# Metric 1: Answer Accuracy  (M1)
# ====================================================================
#
# Definition:
#   Fraction of expected key_points that appear (substring match)
#   in the model answer.
#
# Formula:
#   M1 = |{kp ∈ key_points : kp ⊂ answer}| / |key_points|
#   If key_points is empty → M1 = 1.0 (vacuously correct)
#
# Judgment: AUTOMATIC
# ====================================================================

def metric_answer_accuracy(key_points: List[str], answer: str) -> Dict[str, Any]:
    if not key_points:
        return {"score": 1.0, "hits": [], "method": "auto"}
    ans = _norm(answer)
    hits = [_norm(kp) in ans for kp in key_points]
    score = sum(hits) / len(hits)
    return {"score": round(score, 4), "hits": hits, "method": "auto"}


# ====================================================================
# Metric 2: Evidence Groundedness  (M2)
# ====================================================================
#
# Definition:
#   Fraction of sentence-level claims in the answer that can be
#   traced back to at least one gold_evidence_text (substring overlap).
#
# Formula:
#   claims = split_claims(answer)
#   grounded(c) = ∃ e ∈ gold_evidence_texts : overlap(c, e)
#   M2 = |{c : grounded(c)}| / |claims|
#
#   overlap(c, e): at least one 6-char substring of c appears in e,
#   OR at least one 6-char substring of e appears in c.
#
# Judgment: AUTOMATIC (heuristic) + MANUAL (spot-check recommended)
# ====================================================================

def _has_overlap(claim: str, evidence: str, window: int = 6) -> bool:
    """Check if claim and evidence share a common substring of length >= window."""
    c = _norm(claim)
    e = _norm(evidence)
    # Check claim substrings in evidence
    for i in range(len(c) - window + 1):
        if c[i:i + window] in e:
            return True
    # Check evidence substrings in claim
    for i in range(len(e) - window + 1):
        if e[i:i + window] in c:
            return True
    return False


def metric_evidence_groundedness(
    answer: str,
    gold_evidence_texts: List[str],
) -> Dict[str, Any]:
    claims = _split_claims(answer)
    if not claims:
        return {"score": 1.0, "total_claims": 0, "grounded_claims": 0,
                "ungrounded": [], "method": "auto_heuristic"}
    grounded = []
    ungrounded = []
    for c in claims:
        if any(_has_overlap(c, e) for e in gold_evidence_texts):
            grounded.append(c)
        else:
            ungrounded.append(c)
    score = len(grounded) / len(claims) if claims else 1.0
    return {
        "score": round(score, 4),
        "total_claims": len(claims),
        "grounded_claims": len(grounded),
        "ungrounded": ungrounded,
        "method": "auto_heuristic",
    }


# ====================================================================
# Metric 3: Unsupported Claim Rate  (M3)
# ====================================================================
#
# Definition:
#   Fraction of claims in the answer that are NOT supported by
#   ANY retrieved context chunk (not just gold evidence).
#   This catches hallucinations beyond evidence scope.
#
# Formula:
#   claims = split_claims(answer)
#   supported(c) = ∃ ctx ∈ retrieved_context : overlap(c, ctx)
#   M3 = |{c : ¬supported(c)}| / |claims|
#   Lower is better. 0.0 = all claims supported.
#
# Judgment: AUTOMATIC (heuristic) + MANUAL (for ambiguous cases)
# ====================================================================

def metric_unsupported_claim_rate(
    answer: str,
    retrieved_context: List[str],
) -> Dict[str, Any]:
    claims = _split_claims(answer)
    if not claims:
        return {"score": 0.0, "total_claims": 0, "unsupported_claims": 0,
                "unsupported": [], "method": "auto_heuristic"}
    unsupported = []
    for c in claims:
        if not any(_has_overlap(c, ctx) for ctx in retrieved_context):
            unsupported.append(c)
    score = len(unsupported) / len(claims)
    return {
        "score": round(score, 4),
        "total_claims": len(claims),
        "unsupported_claims": len(unsupported),
        "unsupported": unsupported,
        "method": "auto_heuristic",
    }


# ====================================================================
# Metric 4: Correct Refusal Rate  (M4)
# ====================================================================
#
# Definition:
#   Binary — did the model correctly refuse when it should have?
#
# Formula:
#   if case_type == "unsupported":
#       M4 = 1.0 if answer contains refusal phrase, else 0.0
#   if case_type == "fully_supported":
#       M4 = 1.0 if answer does NOT contain refusal phrase, else 0.0
#       (false refusal is also a failure)
#   if case_type == "partially_supported":
#       M4 = 1.0 (always pass — both answering and refusing are valid)
#   if case_type == "noisy_context":
#       M4 = 1.0 if answer does NOT contain refusal phrase, else 0.0
#
# Judgment: AUTOMATIC
# ====================================================================

def metric_correct_refusal(
    answer: str,
    case_type: str,
) -> Dict[str, Any]:
    has_refusal = any(_norm(p) in _norm(answer) for p in REFUSAL_PHRASES)

    if case_type == "unsupported":
        score = 1.0 if has_refusal else 0.0
        reason = "should_refuse"
    elif case_type == "fully_supported":
        score = 1.0 if not has_refusal else 0.0
        reason = "should_not_refuse"
    elif case_type == "noisy_context":
        score = 1.0 if not has_refusal else 0.0
        reason = "should_not_refuse"
    else:  # partially_supported
        score = 1.0
        reason = "either_acceptable"

    return {
        "score": score,
        "has_refusal": has_refusal,
        "reason": reason,
        "method": "auto",
    }


# ====================================================================
# Metric 5: Partial Answer Compliance  (M5)
# ====================================================================
#
# Definition:
#   For partially_supported questions: does the model
#   (a) answer the supported sub-parts AND
#   (b) explicitly acknowledge the unsupported sub-parts?
#
# Formula:
#   answered = key_point_recall >= 0.5   (covers supported parts)
#   flagged  = answer contains caveat phrase
#   M5 = 1.0  if answered AND flagged
#        0.5  if answered but NOT flagged   (correct but overconfident)
#        0.25 if NOT answered but flagged   (flagged but wrong content)
#        0.0  if neither
#
#   For non-partial case types → M5 = 1.0 (not applicable)
#
# Judgment: AUTOMATIC
# ====================================================================

def metric_partial_answer_compliance(
    answer: str,
    key_points: List[str],
    case_type: str,
) -> Dict[str, Any]:
    if case_type != "partially_supported":
        return {"score": 1.0, "answered": True, "flagged": True,
                "reason": "not_applicable", "method": "auto"}

    kp_recall = metric_answer_accuracy(key_points, answer)["score"]
    answered = kp_recall >= 0.5
    flagged = any(_norm(p) in _norm(answer) for p in CAVEAT_PHRASES)

    if answered and flagged:
        score = 1.0
        reason = "correct_partial_with_caveat"
    elif answered and not flagged:
        score = 0.5
        reason = "correct_but_overconfident"
    elif not answered and flagged:
        score = 0.25
        reason = "flagged_but_wrong_content"
    else:
        score = 0.0
        reason = "neither_answered_nor_flagged"

    return {
        "score": score,
        "answered": answered,
        "flagged": flagged,
        "kp_recall": round(kp_recall, 4),
        "reason": reason,
        "method": "auto",
    }


# ====================================================================
# Metric 6: Context Noise Ratio  (M6)
# ====================================================================
#
# Definition:
#   Measures how much noise from distractor chunks leaked into the
#   final answer.
#
# Formula:
#   noise_terms = extract_distinctive_terms(noise_texts)
#   gold_terms  = extract_distinctive_terms(gold_evidence_texts)
#   leaked = {t ∈ noise_terms : t ⊂ answer AND t ∉ gold_terms}
#   M6 = |leaked| / max(|noise_terms|, 1)
#   Lower is better. 0.0 = no noise leaked.
#
#   For non-noisy case types → M6 = 0.0 (clean by definition)
#
# Judgment: AUTOMATIC
# ====================================================================

def metric_context_noise_ratio(
    answer: str,
    noise_texts: List[str],
    gold_evidence_texts: List[str],
    case_type: str,
) -> Dict[str, Any]:
    if case_type != "noisy_context" or not noise_texts:
        return {"score": 0.0, "leaked_terms": [], "total_noise_terms": 0,
                "reason": "not_applicable", "method": "auto"}

    ans = _norm(answer)
    gold_combined = _norm(" ".join(gold_evidence_texts))

    noise_terms = []
    for nt in noise_texts:
        noise_terms.extend(_extract_distinctive_terms(nt))

    # Deduplicate
    noise_terms = list(set(noise_terms))

    leaked = []
    for t in noise_terms:
        tn = _norm(t)
        if tn in ans and tn not in gold_combined:
            leaked.append(t)

    score = len(leaked) / max(len(noise_terms), 1)
    return {
        "score": round(score, 4),
        "leaked_terms": leaked,
        "total_noise_terms": len(noise_terms),
        "reason": "noise_leak_check",
        "method": "auto",
    }


# ====================================================================
# Composite Score
# ====================================================================

@dataclass
class RubricResult:
    """Full rubric result for one sample."""
    id: str
    case_type: str
    question: str
    model_answer: str

    # Individual metric results
    m1_accuracy: Dict[str, Any] = field(default_factory=dict)
    m2_groundedness: Dict[str, Any] = field(default_factory=dict)
    m3_unsupported: Dict[str, Any] = field(default_factory=dict)
    m4_refusal: Dict[str, Any] = field(default_factory=dict)
    m5_partial: Dict[str, Any] = field(default_factory=dict)
    m6_noise: Dict[str, Any] = field(default_factory=dict)

    # Composite
    composite_score: float = 0.0
    all_pass: bool = False

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "case_type": self.case_type,
            "question": self.question,
            "model_answer": self.model_answer[:200],
            "m1_accuracy": self.m1_accuracy,
            "m2_groundedness": self.m2_groundedness,
            "m3_unsupported": self.m3_unsupported,
            "m4_refusal": self.m4_refusal,
            "m5_partial": self.m5_partial,
            "m6_noise": self.m6_noise,
            "composite_score": self.composite_score,
            "all_pass": self.all_pass,
            "error": self.error,
        }


# ====================================================================
# Rubric Weights per Case Type
# ====================================================================
#
# Each case type emphasizes different metrics.
# Weights sum to 1.0 within each case type.
#
#                     M1     M2     M3     M4     M5     M6
# fully_supported   [0.40,  0.30,  0.15,  0.15,  0.00,  0.00]
# partially_sup     [0.25,  0.15,  0.10,  0.10,  0.40,  0.00]
# unsupported       [0.00,  0.00,  0.00,  1.00,  0.00,  0.00]
# noisy_context     [0.30,  0.20,  0.10,  0.10,  0.00,  0.30]
#
# Note: M3 (unsupported claim rate) and M6 (noise ratio) are
# "lower is better" — we invert them: contribution = 1 - score
# ====================================================================

WEIGHTS = {
    "fully_supported":     {"m1": 0.40, "m2": 0.30, "m3": 0.15, "m4": 0.15, "m5": 0.00, "m6": 0.00},
    "partially_supported": {"m1": 0.25, "m2": 0.15, "m3": 0.10, "m4": 0.10, "m5": 0.40, "m6": 0.00},
    "unsupported":         {"m1": 0.00, "m2": 0.00, "m3": 0.00, "m4": 1.00, "m5": 0.00, "m6": 0.00},
    "noisy_context":       {"m1": 0.30, "m2": 0.20, "m3": 0.10, "m4": 0.10, "m5": 0.00, "m6": 0.30},
}

# Pass thresholds per metric
THRESHOLDS = {
    "m1": 0.8,   # ≥80% key points hit
    "m2": 0.6,   # ≥60% claims grounded
    "m3": 0.3,   # ≤30% claims unsupported  (inverted: score ≤ 0.3)
    "m4": 1.0,   # must be exactly correct
    "m5": 0.5,   # ≥0.5 partial compliance
    "m6": 0.1,   # ≤10% noise leaked  (inverted: score ≤ 0.1)
}


# ====================================================================
# Main Rubric Class
# ====================================================================

class Rubric:
    """Stateless scorer. Call rubric.score() per sample."""

    def score(
        self,
        sample: Dict[str, Any],
        model_answer: str,
        retrieved_context: Optional[List[str]] = None,
        retrieved_scores: Optional[List[float]] = None,
    ) -> RubricResult:
        case = sample["case_type"]
        kps = sample.get("key_points", [])
        gold = sample.get("gold_evidence_texts", [])
        noise = sample.get("noise_texts", [])

        r = RubricResult(
            id=sample["id"],
            case_type=case,
            question=sample["question"],
            model_answer=model_answer,
        )

        # M1: Answer Accuracy
        r.m1_accuracy = metric_answer_accuracy(kps, model_answer)

        # M2: Evidence Groundedness
        r.m2_groundedness = metric_evidence_groundedness(model_answer, gold)

        # M3: Unsupported Claim Rate
        ctx = retrieved_context or []
        r.m3_unsupported = metric_unsupported_claim_rate(model_answer, ctx)

        # M4: Correct Refusal
        r.m4_refusal = metric_correct_refusal(model_answer, case)

        # M5: Partial Answer Compliance
        r.m5_partial = metric_partial_answer_compliance(model_answer, kps, case)

        # M6: Context Noise Ratio
        r.m6_noise = metric_context_noise_ratio(model_answer, noise, gold, case)

        # Composite score (weighted, M3 and M6 inverted)
        w = WEIGHTS.get(case, WEIGHTS["fully_supported"])
        composite = (
            w["m1"] * r.m1_accuracy["score"]
            + w["m2"] * r.m2_groundedness["score"]
            + w["m3"] * (1.0 - r.m3_unsupported["score"])  # invert: lower is better
            + w["m4"] * r.m4_refusal["score"]
            + w["m5"] * r.m5_partial["score"]
            + w["m6"] * (1.0 - r.m6_noise["score"])        # invert: lower is better
        )
        r.composite_score = round(composite, 4)

        # All-pass check (per applicable thresholds)
        r.all_pass = (
            (r.m1_accuracy["score"] >= THRESHOLDS["m1"] or w["m1"] == 0)
            and (r.m2_groundedness["score"] >= THRESHOLDS["m2"] or w["m2"] == 0)
            and (r.m3_unsupported["score"] <= THRESHOLDS["m3"] or w["m3"] == 0)
            and (r.m4_refusal["score"] >= THRESHOLDS["m4"] or w["m4"] == 0)
            and (r.m5_partial["score"] >= THRESHOLDS["m5"] or w["m5"] == 0)
            and (r.m6_noise["score"] <= THRESHOLDS["m6"] or w["m6"] == 0)
        )

        return r


# ====================================================================
# Demo / Dry-Run
# ====================================================================

def _demo():
    """Dry-run with synthetic data to verify rubric logic."""
    rubric = Rubric()

    # --- fully_supported: good answer ---
    r1 = rubric.score(
        sample={
            "id": "demo-fs",
            "case_type": "fully_supported",
            "question": "Transformer 相比 RNN 的优势？",
            "key_points": ["并行处理", "长距离依赖"],
            "gold_evidence_texts": ["Transformer 的核心优势：并行处理能力和长距离依赖建模"],
            "noise_texts": [],
        },
        model_answer="Transformer 相比 RNN 有两个核心优势：并行处理能力和长距离依赖建模。",
        retrieved_context=["Transformer 的核心优势：并行处理能力和长距离依赖建模"],
    )

    # --- unsupported: good refusal ---
    r2 = rubric.score(
        sample={
            "id": "demo-us",
            "case_type": "unsupported",
            "question": "如何用 Terraform 部署？",
            "key_points": [],
            "gold_evidence_texts": [],
            "noise_texts": [],
        },
        model_answer="抱歉，知识库中没有相关信息，无法回答关于 Terraform 部署的问题。",
        retrieved_context=[],
    )

    # --- partially_supported: good partial ---
    r3 = rubric.score(
        sample={
            "id": "demo-ps",
            "case_type": "partially_supported",
            "question": "Dify 支持多少模型？水平扩展方案？",
            "key_points": ["100+"],
            "gold_evidence_texts": ["Dify 支持100+开源和商用模型"],
            "noise_texts": [],
            "unsupported_subquestion": "水平扩展方案",
        },
        model_answer="Dify 支持100+开源和商用模型。但文档中未提及水平扩展方案的具体细节。",
        retrieved_context=["Dify 支持100+开源和商用模型"],
    )

    # --- noisy_context: answer with noise leak ---
    r4 = rubric.score(
        sample={
            "id": "demo-nc",
            "case_type": "noisy_context",
            "question": "PSSH 是由谁提出的？",
            "key_points": ["纽厄尔", "西蒙", "1976"],
            "gold_evidence_texts": ["PSSH由纽厄尔和西蒙于1976年提出"],
            "noise_texts": [
                "ELIZA由魏泽鲍姆于1966年开发",
                "ReAct由Shunyu Yao于2022年提出",
            ],
        },
        model_answer="PSSH由纽厄尔和西蒙于1976年提出。另外ELIZA由魏泽鲍姆于1966年开发。",
        retrieved_context=["PSSH由纽厄尔和西蒙于1976年提出", "ELIZA由魏泽鲍姆于1966年开发"],
    )

    print("=" * 65)
    print("  SCORING RUBRIC DRY-RUN")
    print("=" * 65)

    for r in [r1, r2, r3, r4]:
        print(f"\n--- {r.id} ({r.case_type}) ---")
        print(f"  Q: {r.question}")
        print(f"  A: {r.model_answer[:80]}")
        print(f"  M1 Accuracy:     {r.m1_accuracy['score']:.2f}")
        print(f"  M2 Groundedness: {r.m2_groundedness['score']:.2f}")
        print(f"  M3 Unsupported:  {r.m3_unsupported['score']:.2f}  (lower=better)")
        print(f"  M4 Refusal:      {r.m4_refusal['score']:.2f}  ({r.m4_refusal['reason']})")
        print(f"  M5 Partial:      {r.m5_partial['score']:.2f}  ({r.m5_partial['reason']})")
        print(f"  M6 Noise:        {r.m6_noise['score']:.2f}  (lower=better)")
        print(f"  Composite:       {r.composite_score:.2f}")
        print(f"  All Pass:        {r.all_pass}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    _demo()
