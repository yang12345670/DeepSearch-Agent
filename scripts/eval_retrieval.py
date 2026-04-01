# -*- coding: utf-8 -*-
"""
Retrieval & Answer Evaluation Script for DeepSearch Agent.

Evaluates two dimensions:
  (A) Retrieval Hit Rate — does the gold evidence appear in retrieved chunks?
  (B) Answer Accuracy    — does the model answer contain the ground truth?

Usage:
  # Start the server first:  python main.py
  # Then run evaluation:
  python scripts/eval_retrieval.py

  # Options:
  python scripts/eval_retrieval.py --top_k 5 --api http://127.0.0.1:8000 --dataset data/eval_qa_dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ====================================================================
# Configuration
# ====================================================================

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_DATASET = "data/eval_qa_dataset.json"
DEFAULT_TOP_K = 5


# ====================================================================
# Agent Caller
# ====================================================================

def call_agent(question: str, *, api_base: str, top_k: int) -> Dict[str, Any]:
    """Call the /eval/query endpoint and return answer + retrieved_context.

    To swap in a different agent, replace this function body.
    """
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
# Matching Helpers
# ====================================================================

def normalize(text: str) -> str:
    """Lowercase + strip whitespace for fuzzy substring matching."""
    return " ".join(text.lower().split())


def chunk_contains_answer(answer: str, context_list: List[str]) -> bool:
    """Check if the ground-truth answer appears in any retrieved chunk.

    This is the primary retrieval hit criterion: a chunk is considered
    relevant if it contains the answer content, regardless of whether
    the full evidence sentence is present.
    """
    ans = normalize(answer)
    for ctx in context_list:
        if ans in normalize(ctx):
            return True
    return False


def retrieval_hit_at_k(answer: str, context_list: List[str], k: int) -> bool:
    """Check retrieval hit within top-k chunks (answer-in-chunk criterion)."""
    return chunk_contains_answer(answer, context_list[:k])


def answer_match(ground_truth: str, model_answer: str) -> bool:
    """Check if ground truth is a substring of model answer (case-insensitive)."""
    return normalize(ground_truth) in normalize(model_answer)


# ====================================================================
# Result Structures
# ====================================================================

@dataclass
class SampleResult:
    idx: int
    question: str
    ground_truth: str
    evidence: str
    model_answer: str
    retrieved_context: List[str]
    retrieved_scores: List[float]
    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False
    answer_correct: bool = False
    error: Optional[str] = None


@dataclass
class EvalMetrics:
    total: int = 0
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    answer_correct: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0
    results: List[SampleResult] = field(default_factory=list)

    @property
    def hit_rate_1(self) -> float:
        return self.hit_at_1 / self.total if self.total else 0.0

    @property
    def hit_rate_3(self) -> float:
        return self.hit_at_3 / self.total if self.total else 0.0

    @property
    def hit_rate_5(self) -> float:
        return self.hit_at_5 / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.answer_correct / self.total if self.total else 0.0


# ====================================================================
# Main Evaluation
# ====================================================================

def run_evaluation(
    dataset: List[Dict[str, Any]],
    *,
    api_base: str,
    top_k: int,
) -> EvalMetrics:
    metrics = EvalMetrics(total=len(dataset))
    start = time.time()

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["answer"]
        evidence = item["evidence"]

        print(f"  [{i+1:02d}/{len(dataset)}] {question[:50]}...", end=" ", flush=True)

        try:
            result = call_agent(question, api_base=api_base, top_k=top_k)
            model_answer = result["answer"]
            retrieved = result["retrieved_context"]
            scores = result.get("retrieved_scores", [])

            h1 = retrieval_hit_at_k(ground_truth, retrieved, 1)
            h3 = retrieval_hit_at_k(ground_truth, retrieved, 3)
            h5 = retrieval_hit_at_k(ground_truth, retrieved, 5)
            ac = answer_match(ground_truth, model_answer)

            if h1:
                metrics.hit_at_1 += 1
            if h3:
                metrics.hit_at_3 += 1
            if h5:
                metrics.hit_at_5 += 1
            if ac:
                metrics.answer_correct += 1

            sr = SampleResult(
                idx=i,
                question=question,
                ground_truth=ground_truth,
                evidence=evidence,
                model_answer=model_answer,
                retrieved_context=retrieved,
                retrieved_scores=scores,
                hit_at_1=h1,
                hit_at_3=h3,
                hit_at_5=h5,
                answer_correct=ac,
            )

            status = []
            status.append("H1" if h1 else "--")
            status.append("H3" if h3 else "--")
            status.append("H5" if h5 else "--")
            status.append("AC" if ac else "--")
            print(" | ".join(status))

        except Exception as e:
            metrics.errors += 1
            sr = SampleResult(
                idx=i,
                question=question,
                ground_truth=ground_truth,
                evidence=evidence,
                model_answer="",
                retrieved_context=[],
                retrieved_scores=[],
                error=str(e),
            )
            print(f"ERROR: {e}")

        metrics.results.append(sr)

    metrics.elapsed_sec = time.time() - start
    return metrics


# ====================================================================
# Report
# ====================================================================

def print_report(metrics: EvalMetrics, *, top_k: int) -> None:
    w = 60
    print()
    print("=" * w)
    print("  RETRIEVAL & ANSWER EVALUATION REPORT")
    print("=" * w)
    print(f"  Total samples:     {metrics.total}")
    print(f"  Top-K:             {top_k}")
    print(f"  Errors:            {metrics.errors}")
    print(f"  Time:              {metrics.elapsed_sec:.1f}s "
          f"({metrics.elapsed_sec / max(metrics.total, 1):.1f}s/sample)")
    print("-" * w)
    print("  RETRIEVAL HIT RATE")
    print(f"    Hit@1:           {metrics.hit_at_1:3d}/{metrics.total}  "
          f"= {metrics.hit_rate_1:.1%}")
    print(f"    Hit@3:           {metrics.hit_at_3:3d}/{metrics.total}  "
          f"= {metrics.hit_rate_3:.1%}")
    print(f"    Hit@5:           {metrics.hit_at_5:3d}/{metrics.total}  "
          f"= {metrics.hit_rate_5:.1%}")
    print("-" * w)
    print("  ANSWER ACCURACY")
    print(f"    Correct:         {metrics.answer_correct:3d}/{metrics.total}  "
          f"= {metrics.accuracy:.1%}")
    print("=" * w)

    # Failed cases
    retrieval_fails = [r for r in metrics.results if not r.hit_at_5 and not r.error]
    answer_fails = [r for r in metrics.results if not r.answer_correct and not r.error]
    error_cases = [r for r in metrics.results if r.error]

    if retrieval_fails:
        print()
        print(f"--- Retrieval Misses (top 10 of {len(retrieval_fails)}) ---")
        for r in retrieval_fails[:10]:
            print(f"  [{r.idx+1:02d}] Q: {r.question[:60]}")
            print(f"       Evidence: {r.evidence[:80]}")
            if r.retrieved_context:
                print(f"       Top-1 retrieved: {r.retrieved_context[0][:80]}")
            else:
                print(f"       (no chunks retrieved)")
            print()

    if answer_fails:
        print(f"--- Answer Misses (top 10 of {len(answer_fails)}) ---")
        for r in answer_fails[:10]:
            print(f"  [{r.idx+1:02d}] Q: {r.question[:60]}")
            print(f"       Expected: {r.ground_truth[:60]}")
            print(f"       Got:      {r.model_answer[:80]}")
            print()

    if error_cases:
        print(f"--- Errors ({len(error_cases)}) ---")
        for r in error_cases[:10]:
            print(f"  [{r.idx+1:02d}] {r.question[:50]}  -> {r.error}")
        print()


def save_results(metrics: EvalMetrics, output_path: str) -> None:
    """Save detailed results to JSON for further analysis."""
    out = {
        "summary": {
            "total": metrics.total,
            "hit_at_1": metrics.hit_at_1,
            "hit_at_3": metrics.hit_at_3,
            "hit_at_5": metrics.hit_at_5,
            "hit_rate_1": round(metrics.hit_rate_1, 4),
            "hit_rate_3": round(metrics.hit_rate_3, 4),
            "hit_rate_5": round(metrics.hit_rate_5, 4),
            "answer_correct": metrics.answer_correct,
            "answer_accuracy": round(metrics.accuracy, 4),
            "errors": metrics.errors,
            "elapsed_sec": round(metrics.elapsed_sec, 1),
        },
        "details": [
            {
                "idx": r.idx,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "evidence": r.evidence,
                "model_answer": r.model_answer,
                "retrieved_context": r.retrieved_context,
                "retrieved_scores": r.retrieved_scores,
                "hit_at_1": r.hit_at_1,
                "hit_at_3": r.hit_at_3,
                "hit_at_5": r.hit_at_5,
                "answer_correct": r.answer_correct,
                "error": r.error,
            }
            for r in metrics.results
        ],
    }
    Path(output_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Detailed results saved to: {output_path}")


# ====================================================================
# Markdown Report (auto-generated per run)
# ====================================================================

def generate_report_md(metrics: EvalMetrics, *, top_k: int, tag: str) -> str:
    """Generate a Markdown evaluation report and save to data/eval_reports/."""
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"eval_{date_str}_{tag}.md"
    report_dir = Path("data/eval_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename

    lines = []
    lines.append(f"# Evaluation Report: {tag}")
    lines.append("")
    lines.append(f"- **Date**: {time_str}")
    lines.append(f"- **Tag**: {tag}")
    lines.append(f"- **Dataset**: eval_qa_dataset.json ({metrics.total} samples)")
    lines.append(f"- **Top-K**: {top_k}")
    lines.append(f"- **Time**: {metrics.elapsed_sec:.1f}s ({metrics.elapsed_sec / max(metrics.total, 1):.1f}s/sample)")
    lines.append(f"- **Errors**: {metrics.errors}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Hit@1 | {metrics.hit_at_1}/{metrics.total} ({metrics.hit_rate_1:.1%}) |")
    lines.append(f"| Hit@3 | {metrics.hit_at_3}/{metrics.total} ({metrics.hit_rate_3:.1%}) |")
    lines.append(f"| Hit@5 | {metrics.hit_at_5}/{metrics.total} ({metrics.hit_rate_5:.1%}) |")
    lines.append(f"| Answer Accuracy | {metrics.answer_correct}/{metrics.total} ({metrics.accuracy:.1%}) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-sample Results")
    lines.append("")
    lines.append("| # | Hit@1 | Hit@3 | Hit@5 | Ans | Question |")
    lines.append("|---|-------|-------|-------|-----|----------|")

    for r in metrics.results:
        if r.error:
            lines.append(f"| {r.idx+1} | ERR | ERR | ERR | ERR | {r.question[:50]} |")
        else:
            h1 = "Y" if r.hit_at_1 else "-"
            h3 = "Y" if r.hit_at_3 else "-"
            h5 = "Y" if r.hit_at_5 else "-"
            ac = "Y" if r.answer_correct else "-"
            lines.append(f"| {r.idx+1} | {h1} | {h3} | {h5} | {ac} | {r.question[:50]} |")

    # Retrieval misses
    retrieval_fails = [r for r in metrics.results if not r.hit_at_5 and not r.error]
    if retrieval_fails:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"## Retrieval Misses ({len(retrieval_fails)} samples)")
        lines.append("")
        for r in retrieval_fails[:15]:
            lines.append(f"**[{r.idx+1}]** {r.question}")
            lines.append(f"- Evidence: `{r.evidence[:100]}`")
            if r.retrieved_context:
                lines.append(f"- Top-1 retrieved: `{r.retrieved_context[0][:100]}`")
            else:
                lines.append(f"- (no chunks retrieved)")
            lines.append("")

    # Answer misses
    answer_fails = [r for r in metrics.results if not r.answer_correct and r.hit_at_5 and not r.error]
    if answer_fails:
        lines.append("---")
        lines.append("")
        lines.append(f"## Answer Misses (retrieved but wrong answer, {len(answer_fails)} samples)")
        lines.append("")
        for r in answer_fails[:15]:
            lines.append(f"**[{r.idx+1}]** {r.question}")
            lines.append(f"- Expected: `{r.ground_truth}`")
            lines.append(f"- Got: `{r.model_answer[:120]}`")
            lines.append("")

    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    print(f"Evaluation report saved to: {report_path}")
    return str(report_path)


def append_to_changelog(metrics: EvalMetrics, *, tag: str, report_path: str) -> None:
    """Append a brief evaluation record to CHANGELOG.md."""
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    changelog = Path("CHANGELOG.md")
    if not changelog.is_file():
        return

    entry = (
        f"\n---\n\n"
        f"## [{date_str}] Eval: {tag}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Hit@1 | {metrics.hit_at_1}/{metrics.total} ({metrics.hit_rate_1:.1%}) |\n"
        f"| Hit@3 | {metrics.hit_at_3}/{metrics.total} ({metrics.hit_rate_3:.1%}) |\n"
        f"| Hit@5 | {metrics.hit_at_5}/{metrics.total} ({metrics.hit_rate_5:.1%}) |\n"
        f"| Answer Accuracy | {metrics.answer_correct}/{metrics.total} ({metrics.accuracy:.1%}) |\n\n"
        f"Full report: [{Path(report_path).name}]({report_path})\n"
    )

    text = changelog.read_text(encoding="utf-8")
    # Insert after the first "---" separator (after the header)
    first_sep = text.find("\n---\n")
    if first_sep != -1:
        text = text[:first_sep] + entry + text[first_sep:]
    else:
        text += entry

    changelog.write_text(text, encoding="utf-8")
    print(f"Evaluation record appended to: CHANGELOG.md")


# ====================================================================
# CLI
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to QA dataset JSON.")
    parser.add_argument("--api", default=DEFAULT_API_BASE, help="Agent API base URL.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-K for retrieval.")
    parser.add_argument("--output", default="data/eval_results.json", help="Output results JSON path.")
    parser.add_argument("--tag", default="run", help="Short tag for this eval run (e.g. baseline, chunk300, bge-zh).")
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(dataset)} QA pairs from {dataset_path}")
    print(f"API: {args.api}/eval/query | Top-K: {args.top_k} | Tag: {args.tag}")
    print()

    # Run
    metrics = run_evaluation(dataset, api_base=args.api, top_k=args.top_k)

    # Console report
    print_report(metrics, top_k=args.top_k)

    # Save JSON
    save_results(metrics, args.output)

    # Save MD report + append to CHANGELOG
    report_path = generate_report_md(metrics, top_k=args.top_k, tag=args.tag)
    append_to_changelog(metrics, tag=args.tag, report_path=report_path)


if __name__ == "__main__":
    main()
