# -*- coding: utf-8 -*-
"""
Answer-Layer Evaluation Script for DeepSearch Agent.

Evaluates five dimensions beyond retrieval:
  1. Answer Correctness   — do key_points appear in the model answer?
  2. Evidence Groundedness — does the answer stick to provided evidence?
  3. Context Noise Ratio   — can the model ignore irrelevant context?
  4. Correct Refusal       — does the model refuse unsupported questions?
  5. Partial-Answer        — does the model answer what it can + flag gaps?

Usage:
  python scripts/eval_answer.py
  python scripts/eval_answer.py --api http://127.0.0.1:8000 --tag v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Rubric integration (detailed 6-metric scoring)
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from scoring_rubric import Rubric


# ====================================================================
# Configuration
# ====================================================================

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_BENCHMARK = "data/answer_eval_dataset.jsonl"
DEFAULT_TOP_K = 5

# Refusal indicator phrases (Chinese + English)
REFUSAL_PHRASES = [
    "无法回答", "没有相关信息", "未找到", "不确定", "无法确定",
    "未提及", "没有提及", "不在知识库", "无相关", "无法判断",
    "not found", "no relevant", "cannot answer", "don't have",
    "not mentioned", "no information",
]

CAVEAT_PHRASES = [
    "但文档中未", "但未提及", "但没有", "但未找到",
    "不过文档中", "不过未提及", "尚未涉及", "未涵盖",
    "however", "not mentioned", "no information about",
]


# ====================================================================
# Helpers
# ====================================================================

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def key_point_hit(key_points: List[str], answer: str) -> List[bool]:
    """Check which key_points appear in the answer."""
    ans = normalize(answer)
    return [normalize(kp) in ans for kp in key_points]


def key_point_recall(key_points: List[str], answer: str) -> float:
    """Fraction of key_points found in answer."""
    if not key_points:
        return 1.0
    hits = key_point_hit(key_points, answer)
    return sum(hits) / len(hits)


def contains_refusal(answer: str) -> bool:
    """Check if the answer contains refusal language."""
    ans = normalize(answer)
    return any(normalize(p) in ans for p in REFUSAL_PHRASES)


def contains_caveat(answer: str) -> bool:
    """Check if the answer flags an information gap."""
    ans = normalize(answer)
    return any(normalize(p) in ans for p in CAVEAT_PHRASES)


def contains_hallucination_of(answer: str, noise_texts: List[str]) -> List[str]:
    """Check if answer incorporates content from noise texts but not gold."""
    ans = normalize(answer)
    incorporated = []
    for nt in noise_texts:
        # Extract distinctive terms from noise (4+ char segments)
        terms = [t for t in re.split(r'[,，.。;；:：\s]+', nt) if len(t) >= 4]
        for term in terms:
            if normalize(term) in ans:
                incorporated.append(term)
    return incorporated


# ====================================================================
# Agent Caller
# ====================================================================

def call_eval_endpoint(question: str, *, api_base: str, top_k: int) -> Dict[str, Any]:
    """Call /eval/query — returns answer + retrieved_context."""
    resp = requests.post(
        f"{api_base}/eval/query",
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "retrieved_context": data.get("retrieved_context", []),
        "retrieved_scores": data.get("retrieved_scores", []),
    }


# ====================================================================
# Per-Sample Scoring
# ====================================================================

@dataclass
class SampleScore:
    id: str
    question: str
    case_type: str
    model_answer: str

    # Correctness
    kp_recall: float = 0.0
    kp_hits: List[bool] = field(default_factory=list)

    # Refusal / Caveat
    has_refusal: bool = False
    has_caveat: bool = False

    # Noise
    noise_leaks: List[str] = field(default_factory=list)

    # Verdicts
    correctness_pass: bool = False
    groundedness_pass: bool = False
    refusal_pass: bool = False
    noise_pass: bool = False
    partial_pass: bool = False

    error: Optional[str] = None


def score_sample(sample: Dict[str, Any], model_answer: str) -> SampleScore:
    """Score one benchmark sample against the model answer."""
    sc = SampleScore(
        id=sample["id"],
        question=sample["question"],
        case_type=sample["case_type"],
        model_answer=model_answer,
    )

    case = sample["case_type"]
    kps = sample.get("key_points", [])
    expected = sample.get("expected_behavior", "")

    sc.kp_hits = key_point_hit(kps, model_answer)
    sc.kp_recall = key_point_recall(kps, model_answer)
    sc.has_refusal = contains_refusal(model_answer)
    sc.has_caveat = contains_caveat(model_answer)

    if case == "fully_supported":
        sc.correctness_pass = sc.kp_recall >= 0.8
        sc.groundedness_pass = not sc.has_refusal  # should not refuse
        sc.refusal_pass = True  # N/A
        sc.noise_pass = True  # N/A
        sc.partial_pass = True  # N/A

    elif case == "partially_supported":
        sc.correctness_pass = sc.kp_recall >= 0.5
        sc.groundedness_pass = True  # hard to auto-check
        sc.refusal_pass = True  # N/A
        sc.noise_pass = True  # N/A
        # Should answer what it can AND flag the gap
        sc.partial_pass = sc.kp_recall >= 0.5 and (sc.has_caveat or sc.has_refusal)

    elif case == "unsupported":
        sc.correctness_pass = True  # N/A
        sc.groundedness_pass = True  # N/A
        sc.refusal_pass = sc.has_refusal  # must refuse
        sc.noise_pass = True  # N/A
        sc.partial_pass = True  # N/A

    elif case == "noisy_context":
        noise_texts = sample.get("noise_texts", [])
        sc.noise_leaks = contains_hallucination_of(model_answer, noise_texts)
        sc.correctness_pass = sc.kp_recall >= 0.8
        sc.groundedness_pass = len(sc.noise_leaks) == 0
        sc.refusal_pass = True  # N/A
        sc.noise_pass = sc.correctness_pass and sc.groundedness_pass
        sc.partial_pass = True  # N/A

    return sc


# ====================================================================
# Aggregate Metrics
# ====================================================================

@dataclass
class AnswerEvalMetrics:
    total: int = 0
    by_case: Dict[str, int] = field(default_factory=dict)
    correctness_pass: int = 0
    groundedness_pass: int = 0
    refusal_pass: int = 0
    noise_pass: int = 0
    partial_pass: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0
    scores: List[SampleScore] = field(default_factory=list)

    # Per case-type breakdowns
    case_correctness: Dict[str, List[bool]] = field(default_factory=dict)
    case_kp_recalls: Dict[str, List[float]] = field(default_factory=dict)

    # Rubric detailed results
    rubric_results: List[Dict[str, Any]] = field(default_factory=list)
    rubric_composite_scores: List[float] = field(default_factory=list)

    def rate(self, count: int) -> float:
        return count / self.total if self.total else 0.0

    @property
    def avg_composite(self) -> float:
        if not self.rubric_composite_scores:
            return 0.0
        return sum(self.rubric_composite_scores) / len(self.rubric_composite_scores)


# ====================================================================
# Main Evaluation
# ====================================================================

def run_answer_eval(
    benchmark: Dict[str, Any],
    *,
    api_base: str,
    top_k: int,
) -> AnswerEvalMetrics:
    samples = benchmark["samples"]
    metrics = AnswerEvalMetrics(total=len(samples))
    rubric = Rubric()

    start = time.time()

    for i, sample in enumerate(samples):
        sid = sample["id"]
        question = sample["question"]
        case_type = sample["case_type"]

        metrics.by_case[case_type] = metrics.by_case.get(case_type, 0) + 1

        print(f"  [{i+1:02d}/{len(samples)}] ({case_type:20s}) {question[:45]}...", end=" ", flush=True)

        try:
            result = call_eval_endpoint(question, api_base=api_base, top_k=top_k)
            model_answer = result["answer"]
            retrieved_context = result.get("retrieved_context", [])

            # --- Simple pass/fail scoring ---
            sc = score_sample(sample, model_answer)

            if sc.correctness_pass:
                metrics.correctness_pass += 1
            if sc.groundedness_pass:
                metrics.groundedness_pass += 1
            if sc.refusal_pass:
                metrics.refusal_pass += 1
            if sc.noise_pass:
                metrics.noise_pass += 1
            if sc.partial_pass:
                metrics.partial_pass += 1

            # Track per case-type
            metrics.case_correctness.setdefault(case_type, []).append(sc.correctness_pass)
            metrics.case_kp_recalls.setdefault(case_type, []).append(sc.kp_recall)

            # --- Detailed rubric scoring ---
            rr = rubric.score(
                sample, model_answer,
                retrieved_context=retrieved_context,
                retrieved_scores=result.get("retrieved_scores"),
            )
            metrics.rubric_results.append(rr.to_dict())
            metrics.rubric_composite_scores.append(rr.composite_score)

            # Status line
            flags = []
            flags.append("C" if sc.correctness_pass else "-")
            flags.append("G" if sc.groundedness_pass else "-")
            flags.append("R" if sc.refusal_pass else "-")
            flags.append("N" if sc.noise_pass else "-")
            flags.append("P" if sc.partial_pass else "-")
            print(f"{' '.join(flags)}  composite={rr.composite_score:.2f}")

        except Exception as e:
            metrics.errors += 1
            sc = SampleScore(
                id=sid,
                question=question,
                case_type=case_type,
                model_answer="",
                error=str(e),
            )
            print(f"ERROR: {e}")

        metrics.scores.append(sc)

    metrics.elapsed_sec = time.time() - start
    return metrics


# ====================================================================
# Report
# ====================================================================

def print_report(metrics: AnswerEvalMetrics) -> None:
    w = 65
    print()
    print("=" * w)
    print("  ANSWER-LAYER EVALUATION REPORT")
    print("=" * w)
    print(f"  Total samples:       {metrics.total}")
    print(f"  Errors:              {metrics.errors}")
    print(f"  Time:                {metrics.elapsed_sec:.1f}s")
    print("-" * w)

    print("  OVERALL PASS RATES")
    print(f"    Correctness:       {metrics.correctness_pass:3d}/{metrics.total}  "
          f"= {metrics.rate(metrics.correctness_pass):.1%}")
    print(f"    Groundedness:      {metrics.groundedness_pass:3d}/{metrics.total}  "
          f"= {metrics.rate(metrics.groundedness_pass):.1%}")
    print(f"    Refusal:           {metrics.refusal_pass:3d}/{metrics.total}  "
          f"= {metrics.rate(metrics.refusal_pass):.1%}")
    print(f"    Noise Resistance:  {metrics.noise_pass:3d}/{metrics.total}  "
          f"= {metrics.rate(metrics.noise_pass):.1%}")
    print(f"    Partial-Answer:    {metrics.partial_pass:3d}/{metrics.total}  "
          f"= {metrics.rate(metrics.partial_pass):.1%}")

    print(f"    Avg Composite:     {metrics.avg_composite:.2f}")
    print("-" * w)
    print("  PER CASE-TYPE BREAKDOWN")
    for ct in ["fully_supported", "partially_supported", "unsupported", "noisy_context"]:
        n = metrics.by_case.get(ct, 0)
        if n == 0:
            continue
        corr = sum(metrics.case_correctness.get(ct, []))
        recalls = metrics.case_kp_recalls.get(ct, [])
        avg_kp = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"    {ct:25s}  n={n:2d}  correct={corr}/{n}  avg_kp_recall={avg_kp:.2f}")

    print("=" * w)

    # Failures
    fails = [s for s in metrics.scores if not s.error and not all([
        s.correctness_pass, s.groundedness_pass, s.refusal_pass,
        s.noise_pass, s.partial_pass
    ])]
    if fails:
        print()
        print(f"--- Failures ({len(fails)}) ---")
        for s in fails[:15]:
            flags = []
            if not s.correctness_pass:
                flags.append("correctness")
            if not s.groundedness_pass:
                flags.append("groundedness")
            if not s.refusal_pass:
                flags.append("refusal")
            if not s.noise_pass:
                flags.append("noise")
            if not s.partial_pass:
                flags.append("partial")
            print(f"  [{s.id}] ({s.case_type}) failed: {', '.join(flags)}")
            print(f"    Q: {s.question[:60]}")
            print(f"    KP recall: {s.kp_recall:.2f}  refusal={s.has_refusal}  caveat={s.has_caveat}")
            if s.noise_leaks:
                print(f"    Noise leaks: {s.noise_leaks[:3]}")
            print(f"    Answer: {s.model_answer[:100]}")
            print()


def save_results(metrics: AnswerEvalMetrics, output_path: str) -> None:
    out = {
        "summary": {
            "total": metrics.total,
            "correctness_pass": metrics.correctness_pass,
            "groundedness_pass": metrics.groundedness_pass,
            "refusal_pass": metrics.refusal_pass,
            "noise_pass": metrics.noise_pass,
            "partial_pass": metrics.partial_pass,
            "errors": metrics.errors,
            "elapsed_sec": round(metrics.elapsed_sec, 1),
            "avg_composite": round(metrics.avg_composite, 4),
            "by_case": metrics.by_case,
        },
        "rubric_details": metrics.rubric_results,
        "details": [
            {
                "id": s.id,
                "question": s.question,
                "case_type": s.case_type,
                "model_answer": s.model_answer,
                "kp_recall": round(s.kp_recall, 4),
                "kp_hits": s.kp_hits,
                "has_refusal": s.has_refusal,
                "has_caveat": s.has_caveat,
                "noise_leaks": s.noise_leaks,
                "correctness_pass": s.correctness_pass,
                "groundedness_pass": s.groundedness_pass,
                "refusal_pass": s.refusal_pass,
                "noise_pass": s.noise_pass,
                "partial_pass": s.partial_pass,
                "error": s.error,
            }
            for s in metrics.scores
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDetailed results saved to: {output_path}")


def generate_report_md(metrics: AnswerEvalMetrics, *, tag: str) -> str:
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"eval_answer_{date_str}_{tag}.md"
    report_dir = Path("data/eval_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename

    lines = [
        f"# Answer-Layer Evaluation: {tag}",
        "",
        f"- **Date**: {time_str}",
        f"- **Samples**: {metrics.total}",
        f"- **Errors**: {metrics.errors}",
        f"- **Time**: {metrics.elapsed_sec:.1f}s",
        "",
        "## Overall",
        "",
        "| Metric | Pass | Rate |",
        "|--------|------|------|",
        f"| Correctness | {metrics.correctness_pass}/{metrics.total} | {metrics.rate(metrics.correctness_pass):.1%} |",
        f"| Groundedness | {metrics.groundedness_pass}/{metrics.total} | {metrics.rate(metrics.groundedness_pass):.1%} |",
        f"| Refusal | {metrics.refusal_pass}/{metrics.total} | {metrics.rate(metrics.refusal_pass):.1%} |",
        f"| Noise Resistance | {metrics.noise_pass}/{metrics.total} | {metrics.rate(metrics.noise_pass):.1%} |",
        f"| Partial-Answer | {metrics.partial_pass}/{metrics.total} | {metrics.rate(metrics.partial_pass):.1%} |",
        f"| **Avg Composite** | **{metrics.avg_composite:.2f}** | - |",
        "",
        "## Per Case-Type",
        "",
        "| Case Type | n | Correct | Avg KP Recall |",
        "|-----------|---|---------|---------------|",
    ]

    for ct in ["fully_supported", "partially_supported", "unsupported", "noisy_context"]:
        n = metrics.by_case.get(ct, 0)
        if n == 0:
            continue
        corr = sum(metrics.case_correctness.get(ct, []))
        recalls = metrics.case_kp_recalls.get(ct, [])
        avg_kp = sum(recalls) / len(recalls) if recalls else 0.0
        lines.append(f"| {ct} | {n} | {corr}/{n} | {avg_kp:.2f} |")

    lines.append("")

    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    print(f"Report saved to: {report_path}")
    return str(report_path)


# ====================================================================
# CLI
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Answer-layer evaluation for DeepSearch Agent.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark JSON path.")
    parser.add_argument("--api", default=DEFAULT_API_BASE, help="Agent API base URL.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-K for retrieval.")
    parser.add_argument("--output", default="data/eval_answer_results.json", help="Output JSON path.")
    parser.add_argument("--tag", default="run", help="Tag for this eval run.")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_file():
        print(f"Benchmark not found: {benchmark_path}")
        sys.exit(1)

    # Support both .jsonl (one record per line) and .json ({"samples": [...]})
    raw = benchmark_path.read_text(encoding="utf-8")
    if benchmark_path.suffix == ".jsonl":
        samples = [json.loads(line) for line in raw.splitlines() if line.strip()]
        benchmark = {"samples": samples}
    else:
        benchmark = json.loads(raw)
        samples = benchmark["samples"]
    print(f"Loaded {len(samples)} samples from {benchmark_path}")

    from collections import Counter
    ct = Counter(s["case_type"] for s in samples)
    for k, v in ct.items():
        print(f"  {k}: {v}")
    print(f"API: {args.api}/eval/query | Top-K: {args.top_k} | Tag: {args.tag}")
    print()

    metrics = run_answer_eval(benchmark, api_base=args.api, top_k=args.top_k)
    print_report(metrics)
    save_results(metrics, args.output)
    generate_report_md(metrics, tag=args.tag)


if __name__ == "__main__":
    main()
