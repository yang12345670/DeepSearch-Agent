# -*- coding: utf-8 -*-
"""
Offline Answer-Layer Evaluation Pipeline for DeepSearch Agent.

Reads pre-computed model predictions from JSONL and scores them against
the answer-layer benchmark — no running server required.

Usage:
  python scripts/eval_answer_offline.py \
    --benchmark data/answer_eval_dataset.jsonl \
    --predictions data/model_predictions.jsonl \
    --output data/eval_answer_offline_results.json \
    --tag baseline

  # With manual annotation overrides:
  python scripts/eval_answer_offline.py \
    --benchmark data/answer_eval_dataset.jsonl \
    --predictions data/model_predictions.jsonl \
    --annotations data/manual_annotations.jsonl \
    --tag baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import Rubric from scoring_rubric.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_rubric import Rubric, THRESHOLDS, WEIGHTS


# ====================================================================
# Loading
# ====================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file (one JSON object per line)."""
    records = []
    raw = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  WARNING: {path} line {lineno}: {e}")
    return records


def load_benchmark(path: str) -> Dict[str, Dict[str, Any]]:
    """Load benchmark samples, return {id: sample}."""
    p = Path(path)
    if p.suffix == ".jsonl":
        records = load_jsonl(path)
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        records = data.get("samples", data) if isinstance(data, dict) else data

    out = {}
    for r in records:
        sid = r.get("id")
        if not sid:
            print(f"  WARNING: benchmark record missing 'id', skipped")
            continue
        out[sid] = r
    return out


def load_predictions(path: str) -> Dict[str, Dict[str, Any]]:
    """Load model predictions, return {id: prediction}."""
    records = load_jsonl(path)
    out = {}
    for r in records:
        sid = r.get("id")
        if not sid:
            print(f"  WARNING: prediction record missing 'id', skipped")
            continue
        if "answer" not in r:
            print(f"  WARNING: prediction {sid} missing 'answer', skipped")
            continue
        out[sid] = r
    return out


def load_annotations(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load optional manual annotations, return {id: overrides}."""
    if not path or not Path(path).is_file():
        return {}
    records = load_jsonl(path)
    return {r["id"]: r for r in records if "id" in r}


# ====================================================================
# Matching
# ====================================================================

def match_samples(
    benchmark: Dict[str, Dict],
    predictions: Dict[str, Dict],
) -> Tuple[List[Tuple[Dict, Dict]], List[str], List[str]]:
    """Join benchmark and predictions by id."""
    matched = []
    b_ids = set(benchmark.keys())
    p_ids = set(predictions.keys())

    for sid in sorted(b_ids & p_ids):
        matched.append((benchmark[sid], predictions[sid]))

    unmatched_bench = sorted(b_ids - p_ids)
    unmatched_pred = sorted(p_ids - b_ids)
    return matched, unmatched_bench, unmatched_pred


# ====================================================================
# Scoring
# ====================================================================

def score_all(
    matched: List[Tuple[Dict, Dict]],
    annotations: Dict[str, Dict],
    rubric: Rubric,
) -> List[Dict[str, Any]]:
    """Score all matched pairs. Returns list of per-sample result dicts."""
    results = []

    for i, (sample, prediction) in enumerate(matched):
        sid = sample["id"]
        answer = prediction["answer"]

        # Build retrieved_context from prediction if available
        # The offline pipeline doesn't have actual chunk texts, so we pass
        # gold_evidence_texts + noise_texts as a proxy for what was in context
        retrieved_context = sample.get("gold_evidence_texts", [])[:]
        retrieved_context.extend(sample.get("noise_texts", []))

        # Score via rubric
        rr = rubric.score(
            sample,
            answer,
            retrieved_context=retrieved_context,
        )

        result = rr.to_dict()
        result["difficulty"] = sample.get("difficulty", "unknown")
        result["tags"] = sample.get("tags", [])
        result["prediction_meta"] = prediction.get("meta", {})
        result["used_evidence_ids"] = prediction.get("used_evidence_ids", [])
        result["final_context_ids"] = prediction.get("final_context_ids", [])
        result["manual_override"] = False

        # Apply manual annotation overrides
        ann = annotations.get(sid)
        if ann:
            if "m2_override" in ann:
                result["m2_groundedness"]["score"] = ann["m2_override"]
                result["m2_groundedness"]["method"] = "manual"
                result["manual_override"] = True
            if "m3_override" in ann:
                result["m3_unsupported"]["score"] = ann["m3_override"]
                result["m3_unsupported"]["method"] = "manual"
                result["manual_override"] = True
            if "notes" in ann:
                result["annotation_notes"] = ann["notes"]

            # Recompute composite if overridden
            if result["manual_override"]:
                ct = sample["case_type"]
                w = WEIGHTS.get(ct, WEIGHTS["fully_supported"])
                composite = (
                    w["m1"] * result["m1_accuracy"]["score"]
                    + w["m2"] * result["m2_groundedness"]["score"]
                    + w["m3"] * (1.0 - result["m3_unsupported"]["score"])
                    + w["m4"] * result["m4_refusal"]["score"]
                    + w["m5"] * result["m5_partial"]["score"]
                    + w["m6"] * (1.0 - result["m6_noise"]["score"])
                )
                result["composite_score"] = round(composite, 4)

                # Recheck all_pass
                result["all_pass"] = (
                    (result["m1_accuracy"]["score"] >= THRESHOLDS["m1"] or w["m1"] == 0)
                    and (result["m2_groundedness"]["score"] >= THRESHOLDS["m2"] or w["m2"] == 0)
                    and (result["m3_unsupported"]["score"] <= THRESHOLDS["m3"] or w["m3"] == 0)
                    and (result["m4_refusal"]["score"] >= THRESHOLDS["m4"] or w["m4"] == 0)
                    and (result["m5_partial"]["score"] >= THRESHOLDS["m5"] or w["m5"] == 0)
                    and (result["m6_noise"]["score"] <= THRESHOLDS["m6"] or w["m6"] == 0)
                )

        # Tag errors for this sample
        result["error_tags"] = tag_errors(result, sample, prediction)

        n_tags = len(result["error_tags"])
        tag_str = f"  errors={n_tags}" if n_tags else ""
        print(f"  [{i+1:02d}/{len(matched)}] {sid:10s}  composite={result['composite_score']:.2f}  "
              f"{'PASS' if result['all_pass'] else 'FAIL'}"
              f"{'  (manual)' if result['manual_override'] else ''}{tag_str}")

        results.append(result)

    return results


# ====================================================================
# Aggregation
# ====================================================================

def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _pass_rate(values: List[float], threshold: float, lower_is_better: bool = False) -> float:
    if not values:
        return 0.0
    if lower_is_better:
        passed = sum(1 for v in values if v <= threshold)
    else:
        passed = sum(1 for v in values if v >= threshold)
    return round(passed / len(values), 4)


def aggregate_overall(results: List[Dict]) -> Dict[str, Any]:
    """Compute overall metrics across all samples."""
    m1s = [r["m1_accuracy"]["score"] for r in results]
    m2s = [r["m2_groundedness"]["score"] for r in results]
    m3s = [r["m3_unsupported"]["score"] for r in results]
    m4s = [r["m4_refusal"]["score"] for r in results]
    m5s = [r["m5_partial"]["score"] for r in results]
    m6s = [r["m6_noise"]["score"] for r in results]
    composites = [r["composite_score"] for r in results]

    return {
        "avg_composite": _avg(composites),
        "avg_m1_accuracy": _avg(m1s),
        "avg_m2_groundedness": _avg(m2s),
        "avg_m3_unsupported": _avg(m3s),
        "avg_m4_refusal": _avg(m4s),
        "avg_m5_partial": _avg(m5s),
        "avg_m6_noise": _avg(m6s),
        "pass_rates": {
            "m1": _pass_rate(m1s, THRESHOLDS["m1"]),
            "m2": _pass_rate(m2s, THRESHOLDS["m2"]),
            "m3": _pass_rate(m3s, THRESHOLDS["m3"], lower_is_better=True),
            "m4": _pass_rate(m4s, THRESHOLDS["m4"]),
            "m5": _pass_rate(m5s, THRESHOLDS["m5"]),
            "m6": _pass_rate(m6s, THRESHOLDS["m6"], lower_is_better=True),
        },
        "all_pass_rate": _avg([1.0 if r["all_pass"] else 0.0 for r in results]),
    }


def aggregate_by_group(results: List[Dict], group_key: str) -> Dict[str, Any]:
    """Group results by a key (case_type or difficulty) and compute per-group metrics."""
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        g = r.get(group_key, "unknown")
        groups.setdefault(g, []).append(r)

    out = {}
    for g, items in sorted(groups.items()):
        m1s = [r["m1_accuracy"]["score"] for r in items]
        m2s = [r["m2_groundedness"]["score"] for r in items]
        composites = [r["composite_score"] for r in items]
        out[g] = {
            "count": len(items),
            "avg_composite": _avg(composites),
            "avg_m1": _avg(m1s),
            "avg_m2": _avg(m2s),
            "pass_rate_m1": _pass_rate(m1s, THRESHOLDS["m1"]),
            "all_pass_rate": _avg([1.0 if r["all_pass"] else 0.0 for r in items]),
        }
    return out


def analyze_weak_cases(results: List[Dict], bottom_n: int = 5) -> Dict[str, Any]:
    """Identify weakest samples and failure patterns."""
    # Bottom-N by composite
    sorted_by_composite = sorted(results, key=lambda r: r["composite_score"])
    lowest = []
    for r in sorted_by_composite[:bottom_n]:
        failed = _get_failed_metrics(r)
        lowest.append({
            "id": r["id"],
            "composite": r["composite_score"],
            "case_type": r["case_type"],
            "failed_metrics": failed,
        })

    # All that failed all_pass
    failed_all = []
    for r in results:
        if not r["all_pass"]:
            failed_all.append({
                "id": r["id"],
                "composite": r["composite_score"],
                "case_type": r["case_type"],
                "failed_metrics": _get_failed_metrics(r),
            })

    return {
        "lowest_composite": lowest,
        "failed_all_pass": failed_all,
    }


def _get_failed_metrics(r: Dict) -> List[str]:
    """Determine which metrics failed for a sample."""
    ct = r["case_type"]
    w = WEIGHTS.get(ct, WEIGHTS["fully_supported"])
    failed = []
    if w["m1"] > 0 and r["m1_accuracy"]["score"] < THRESHOLDS["m1"]:
        failed.append("m1")
    if w["m2"] > 0 and r["m2_groundedness"]["score"] < THRESHOLDS["m2"]:
        failed.append("m2")
    if w["m3"] > 0 and r["m3_unsupported"]["score"] > THRESHOLDS["m3"]:
        failed.append("m3")
    if w["m4"] > 0 and r["m4_refusal"]["score"] < THRESHOLDS["m4"]:
        failed.append("m4")
    if w["m5"] > 0 and r["m5_partial"]["score"] < THRESHOLDS["m5"]:
        failed.append("m5")
    if w["m6"] > 0 and r["m6_noise"]["score"] > THRESHOLDS["m6"]:
        failed.append("m6")
    return failed


# ====================================================================
# Error Tagging (per-sample)
# ====================================================================
#
# Seven interpretable error categories, each with a rule-based check.
# Every sample gets a list of error_tags (empty = clean).
#
# Tag                          | Trigger
# -----------------------------|------------------------------------------
# evidence_not_used            | Gold evidence IDs in context but KP recall < 0.8
# overclaim                    | M3 unsupported claim rate > 0.3
# incorrect_refusal            | Refused when should have answered
# missing_refusal              | Answered when should have refused
# partial_no_caveat            | Answered partial but didn't flag the gap
# noise_distraction            | Noise terms leaked into answer
# missing_key_points           | Specific key_points missed (with list)
# contradiction                | Answer contains claim that contradicts gold evidence
# ====================================================================

def _norm_for_tag(text: str) -> str:
    return " ".join(text.lower().split())


def _check_contradiction(answer: str, gold_texts: List[str]) -> Optional[str]:
    """Detect simple negation contradictions against gold evidence.

    Looks for patterns where the answer negates a fact stated in evidence.
    Returns the contradicting snippet or None.
    """
    ans = _norm_for_tag(answer)

    # Negation patterns (Chinese + English)
    neg_markers = ["不是", "并非", "没有", "无法", "不能", "不支持",
                   "is not", "does not", "cannot", "never", "no "]

    for gold in gold_texts:
        g = _norm_for_tag(gold)
        # Extract key 6-char windows from gold
        gold_fragments = []
        for i in range(0, len(g) - 5):
            gold_fragments.append(g[i:i + 6])

        for neg in neg_markers:
            neg_n = _norm_for_tag(neg)
            # Find negation in answer
            idx = ans.find(neg_n)
            while idx != -1:
                # Check if the text AFTER the negation overlaps with gold
                after = ans[idx:idx + 40]
                for frag in gold_fragments:
                    if frag in after:
                        snippet = answer[max(0, idx - 5):idx + 40].strip()
                        return snippet
                idx = ans.find(neg_n, idx + 1)
    return None


def tag_errors(
    result: Dict[str, Any],
    sample: Dict[str, Any],
    prediction: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Assign interpretable error tags to a single scored sample.

    Returns a list of tag dicts: [{"tag": "...", "detail": "..."}, ...]
    """
    tags: List[Dict[str, Any]] = []
    ct = sample["case_type"]
    kps = sample.get("key_points", [])
    gold_texts = sample.get("gold_evidence_texts", [])
    gold_ids = set(sample.get("gold_evidence_ids", []))
    context_ids = set(prediction.get("final_context_ids", []))
    answer = prediction.get("answer", "")

    # 1. Evidence retrieved but not used
    #    Gold evidence was in the context, yet key points are missing
    if gold_ids and gold_ids.issubset(context_ids):
        if result["m1_accuracy"]["score"] < THRESHOLDS["m1"]:
            missed = [kp for kp, hit in zip(kps, result["m1_accuracy"].get("hits", []))
                      if not hit]
            tags.append({
                "tag": "evidence_not_used",
                "detail": f"Gold evidence in context but answer missed: {missed}",
            })

    # 2. Overclaim beyond evidence
    #    High unsupported claim rate = answer went beyond what evidence says
    #    Skip for unsupported questions (no evidence expected, so all claims are "unsupported")
    w = WEIGHTS.get(ct, WEIGHTS["fully_supported"])
    if w["m3"] > 0 and result["m3_unsupported"]["score"] > THRESHOLDS["m3"]:
        unsup = result["m3_unsupported"].get("unsupported", [])
        tags.append({
            "tag": "overclaim",
            "detail": f"{len(unsup)} unsupported claims: {[c[:50] for c in unsup[:3]]}",
        })

    # 3. Incorrect refusal
    #    Model refused when evidence was available to answer
    if ct in ("fully_supported", "noisy_context") and result["m4_refusal"]["score"] < 1.0:
        tags.append({
            "tag": "incorrect_refusal",
            "detail": "Model refused but evidence was available",
        })

    # 4. Missing refusal
    #    Model answered confidently when it should have refused
    if ct == "unsupported" and result["m4_refusal"]["score"] < 1.0:
        tags.append({
            "tag": "missing_refusal",
            "detail": "Model answered but no evidence supports this question",
        })

    # 5. Partial answer without caveat
    #    Answered the supported part but didn't acknowledge the gap
    if ct == "partially_supported":
        reason = result["m5_partial"].get("reason", "")
        if reason == "correct_but_overconfident":
            unsup_q = sample.get("unsupported_subquestion", "unknown")
            tags.append({
                "tag": "partial_no_caveat",
                "detail": f"Answered correctly but didn't flag unsupported part: '{unsup_q}'",
            })
        elif reason == "flagged_but_wrong_content":
            tags.append({
                "tag": "partial_no_caveat",
                "detail": "Flagged a gap but the supported content was wrong",
            })
        elif reason == "neither_answered_nor_flagged":
            tags.append({
                "tag": "partial_no_caveat",
                "detail": "Neither answered the supported part nor flagged the gap",
            })

    # 6. Noise distraction
    #    Answer incorporated content from distractor chunks
    if ct == "noisy_context":
        leaked = result["m6_noise"].get("leaked_terms", [])
        if leaked:
            tags.append({
                "tag": "noise_distraction",
                "detail": f"Noise terms leaked into answer: {leaked[:5]}",
            })

    # 7. Missing key comparison/enumeration points
    #    Specific key_points that were missed (useful for enumeration questions)
    hits = result["m1_accuracy"].get("hits", [])
    if kps and hits:
        missed = [kp for kp, hit in zip(kps, hits) if not hit]
        if missed and "evidence_not_used" not in [t["tag"] for t in tags]:
            tags.append({
                "tag": "missing_key_points",
                "detail": f"Missed: {missed}",
            })

    # 8. Contradiction against evidence
    if gold_texts and ct != "unsupported":
        contradiction = _check_contradiction(answer, gold_texts)
        if contradiction:
            tags.append({
                "tag": "contradiction",
                "detail": f"Answer may contradict evidence near: '{contradiction[:60]}'",
            })

    return tags


def categorize_errors(results: List[Dict]) -> Dict[str, List[str]]:
    """Legacy error categories — derived from error_tags for backward compat."""
    # Map new tag names to old category names
    tag_to_cat = {
        "missing_key_points": "correctness_failures",
        "evidence_not_used": "correctness_failures",
        "overclaim": "groundedness_failures",
        "incorrect_refusal": "false_refusals",
        "missing_refusal": "missing_refusals",
        "partial_no_caveat": "partial_overconfident",
        "noise_distraction": "noise_leaks",
    }
    cats: Dict[str, List[str]] = {
        "correctness_failures": [],
        "groundedness_failures": [],
        "false_refusals": [],
        "missing_refusals": [],
        "partial_overconfident": [],
        "noise_leaks": [],
    }
    for r in results:
        sid = r["id"]
        for et in r.get("error_tags", []):
            cat = tag_to_cat.get(et["tag"])
            if cat and sid not in cats[cat]:
                cats[cat].append(sid)
    return cats


def build_error_distribution(results: List[Dict]) -> Dict[str, Any]:
    """Build error tag distribution from tagged results."""
    tag_counts: Dict[str, int] = {}
    tag_samples: Dict[str, List[str]] = {}

    for r in results:
        for et in r.get("error_tags", []):
            tag = et["tag"]
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            tag_samples.setdefault(tag, []).append(r["id"])

    # Sort by count descending
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])

    total = len(results)
    clean = sum(1 for r in results if not r.get("error_tags"))

    return {
        "total_samples": total,
        "clean_samples": clean,
        "clean_rate": round(clean / total, 4) if total else 0.0,
        "error_counts": dict(sorted_tags),
        "error_sample_ids": tag_samples,
    }


# ====================================================================
# Output
# ====================================================================

def save_report(
    meta: Dict,
    overall: Dict,
    by_case: Dict,
    by_difficulty: Dict,
    per_sample: List[Dict],
    weak: Dict,
    errors: Dict,
    error_dist: Dict,
    path: str,
) -> None:
    """Write the full JSON report."""
    report = {
        "meta": meta,
        "overall": overall,
        "by_case_type": by_case,
        "by_difficulty": by_difficulty,
        "error_distribution": error_dist,
        "per_sample": per_sample,
        "weak_cases": weak,
        "error_categories": errors,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {path}")


def print_summary(
    overall: Dict,
    by_case: Dict,
    weak: Dict,
    errors: Dict,
) -> None:
    """Print a compact console summary."""
    w = 65
    print()
    print("=" * w)
    print("  OFFLINE ANSWER-LAYER EVALUATION REPORT")
    print("=" * w)

    print(f"  Avg Composite:       {overall['avg_composite']:.2f}")
    print(f"  All-Pass Rate:       {overall['all_pass_rate']:.1%}")
    print("-" * w)

    print("  METRIC AVERAGES (higher=better except M3,M6)")
    print(f"    M1 Accuracy:       {overall['avg_m1_accuracy']:.2f}  pass={overall['pass_rates']['m1']:.0%}")
    print(f"    M2 Groundedness:   {overall['avg_m2_groundedness']:.2f}  pass={overall['pass_rates']['m2']:.0%}")
    print(f"    M3 Unsupported:    {overall['avg_m3_unsupported']:.2f}  pass={overall['pass_rates']['m3']:.0%}  (lower=better)")
    print(f"    M4 Refusal:        {overall['avg_m4_refusal']:.2f}  pass={overall['pass_rates']['m4']:.0%}")
    print(f"    M5 Partial:        {overall['avg_m5_partial']:.2f}  pass={overall['pass_rates']['m5']:.0%}")
    print(f"    M6 Noise:          {overall['avg_m6_noise']:.2f}  pass={overall['pass_rates']['m6']:.0%}  (lower=better)")
    print("-" * w)

    print("  BY CASE TYPE")
    for ct, data in by_case.items():
        print(f"    {ct:25s}  n={data['count']:2d}  composite={data['avg_composite']:.2f}  "
              f"all_pass={data['all_pass_rate']:.0%}")
    print("-" * w)

    # Weak cases
    failed = weak.get("failed_all_pass", [])
    if failed:
        print(f"  FAILURES ({len(failed)} samples)")
        for f in failed[:10]:
            print(f"    {f['id']:10s}  composite={f['composite']:.2f}  "
                  f"failed: {', '.join(f['failed_metrics'])}")
    else:
        print("  ALL SAMPLES PASSED")

    # Error categories
    has_errors = any(v for v in errors.values())
    if has_errors:
        print("-" * w)
        print("  ERROR CATEGORIES")
        for cat, ids in errors.items():
            if ids:
                print(f"    {cat}: {ids}")

    print("=" * w)


# ====================================================================
# Markdown Report
# ====================================================================

def generate_markdown_report(
    meta: Dict,
    overall: Dict,
    by_case: Dict,
    by_difficulty: Dict,
    results: List[Dict],
    weak: Dict,
    error_dist: Dict,
    *,
    tag: str,
) -> str:
    """Generate a human-readable markdown report and save to data/eval_reports/."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"eval_answer_offline_{date_str}_{tag}.md"
    report_dir = Path("data/eval_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename

    L: List[str] = []  # lines

    # --- Header ---
    L.append(f"# Answer-Layer Evaluation Report: {tag}")
    L.append("")
    L.append(f"- **Date**: {time_str}")
    L.append(f"- **Benchmark**: `{meta['benchmark_file']}` ({meta['total_benchmark']} samples)")
    L.append(f"- **Predictions**: `{meta['predictions_file']}` ({meta['total_predictions']} predictions)")
    L.append(f"- **Matched**: {meta['matched_samples']}")
    if meta.get("annotations_file"):
        L.append(f"- **Annotations**: `{meta['annotations_file']}`")
    L.append("")

    # --- Overall ---
    L.append("## Overall Metrics")
    L.append("")
    L.append("| Metric | Avg Score | Pass Rate |")
    L.append("|--------|-----------|-----------|")
    L.append(f"| **Composite** | **{overall['avg_composite']:.2f}** | {overall['all_pass_rate']:.0%} |")
    L.append(f"| M1 Accuracy | {overall['avg_m1_accuracy']:.2f} | {overall['pass_rates']['m1']:.0%} |")
    L.append(f"| M2 Groundedness | {overall['avg_m2_groundedness']:.2f} | {overall['pass_rates']['m2']:.0%} |")
    L.append(f"| M3 Unsupported (lower=better) | {overall['avg_m3_unsupported']:.2f} | {overall['pass_rates']['m3']:.0%} |")
    L.append(f"| M4 Refusal | {overall['avg_m4_refusal']:.2f} | {overall['pass_rates']['m4']:.0%} |")
    L.append(f"| M5 Partial | {overall['avg_m5_partial']:.2f} | {overall['pass_rates']['m5']:.0%} |")
    L.append(f"| M6 Noise (lower=better) | {overall['avg_m6_noise']:.2f} | {overall['pass_rates']['m6']:.0%} |")
    L.append("")

    # --- By Case Type ---
    L.append("## By Case Type")
    L.append("")
    L.append("| Case Type | n | Composite | M1 | M2 | All-Pass |")
    L.append("|-----------|---|-----------|----|----|----------|")
    for ct, d in by_case.items():
        L.append(f"| {ct} | {d['count']} | {d['avg_composite']:.2f} | "
                 f"{d['avg_m1']:.2f} | {d['avg_m2']:.2f} | {d['all_pass_rate']:.0%} |")
    L.append("")

    # --- By Difficulty ---
    L.append("## By Difficulty")
    L.append("")
    L.append("| Difficulty | n | Composite | All-Pass |")
    L.append("|-----------|---|-----------|----------|")
    for diff, d in by_difficulty.items():
        L.append(f"| {diff} | {d['count']} | {d['avg_composite']:.2f} | {d['all_pass_rate']:.0%} |")
    L.append("")

    # --- Error Distribution ---
    L.append("## Error Distribution")
    L.append("")
    L.append(f"- **Clean samples**: {error_dist['clean_samples']}/{error_dist['total_samples']} "
             f"({error_dist['clean_rate']:.0%})")
    L.append("")
    if error_dist["error_counts"]:
        L.append("| Error Tag | Count | Sample IDs |")
        L.append("|-----------|-------|------------|")
        for tag_name, count in error_dist["error_counts"].items():
            ids = error_dist["error_sample_ids"].get(tag_name, [])
            ids_str = ", ".join(ids[:5])
            if len(ids) > 5:
                ids_str += f" (+{len(ids) - 5} more)"
            L.append(f"| `{tag_name}` | {count} | {ids_str} |")
        L.append("")
    else:
        L.append("No errors detected.")
        L.append("")

    # --- Per-Sample Results ---
    L.append("## Per-Sample Results")
    L.append("")
    L.append("| ID | Type | Comp | M1 | M2 | M3 | M4 | M5 | M6 | Pass | Errors |")
    L.append("|----|------|------|----|----|----|----|----|----|----- |--------|")
    for r in results:
        err_tags = [et["tag"] for et in r.get("error_tags", [])]
        err_str = ", ".join(err_tags) if err_tags else "-"
        L.append(
            f"| {r['id']} | {r['case_type'][:8]} | {r['composite_score']:.2f} "
            f"| {r['m1_accuracy']['score']:.1f} "
            f"| {r['m2_groundedness']['score']:.1f} "
            f"| {r['m3_unsupported']['score']:.1f} "
            f"| {r['m4_refusal']['score']:.0f} "
            f"| {r['m5_partial']['score']:.1f} "
            f"| {r['m6_noise']['score']:.1f} "
            f"| {'Y' if r['all_pass'] else 'N'} "
            f"| {err_str} |"
        )
    L.append("")

    # --- Weak-Case Deep Dive ---
    failed = weak.get("failed_all_pass", [])
    if failed:
        L.append("## Weak-Case Analysis")
        L.append("")
        for f_case in failed:
            fid = f_case["id"]
            # Find the full result
            full = next((r for r in results if r["id"] == fid), None)
            if not full:
                continue
            L.append(f"### {fid} ({f_case['case_type']}) — composite {f_case['composite']:.2f}")
            L.append("")
            L.append(f"**Question**: {full['question']}")
            L.append("")
            L.append(f"**Answer** (truncated): {full['model_answer'][:150]}...")
            L.append("")
            L.append(f"**Failed metrics**: {', '.join(f_case['failed_metrics'])}")
            L.append("")

            err_tags = full.get("error_tags", [])
            if err_tags:
                L.append("**Error tags**:")
                L.append("")
                for et in err_tags:
                    L.append(f"- `{et['tag']}`: {et['detail']}")
                L.append("")

            # Show ungrounded claims if any
            ungrounded = full["m2_groundedness"].get("ungrounded", [])
            if ungrounded:
                L.append("**Ungrounded claims**:")
                L.append("")
                for ug in ungrounded[:5]:
                    L.append(f"- {ug[:80]}")
                L.append("")

            L.append("---")
            L.append("")

    # --- Footer ---
    L.append(f"*Generated by eval_answer_offline.py at {time_str}*")
    L.append("")

    content = "\n".join(L)
    report_path.write_text(content, encoding="utf-8")
    print(f"Markdown report saved to: {report_path}")
    return str(report_path)


# ====================================================================
# CLI
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline answer-layer evaluation — reads predictions from JSONL."
    )
    parser.add_argument("--benchmark", default="data/answer_eval_dataset.jsonl",
                        help="Benchmark JSONL/JSON path.")
    parser.add_argument("--predictions", required=True,
                        help="Model predictions JSONL path.")
    parser.add_argument("--annotations", default=None,
                        help="Optional manual annotations JSONL path.")
    parser.add_argument("--output", default="data/eval_answer_offline_results.json",
                        help="Output report JSON path.")
    parser.add_argument("--tag", default="run",
                        help="Tag for this eval run.")
    args = parser.parse_args()

    # 1. Load (with file existence checks)
    if not Path(args.benchmark).is_file():
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)
    if not Path(args.predictions).is_file():
        print(f"ERROR: Predictions file not found: {args.predictions}")
        sys.exit(1)

    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)
    print(f"  {len(benchmark)} samples")

    print(f"Loading predictions: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"  {len(predictions)} predictions")

    annotations = load_annotations(args.annotations)
    if annotations:
        print(f"Loaded {len(annotations)} manual annotations")

    # 2. Match
    matched, unmatched_b, unmatched_p = match_samples(benchmark, predictions)
    print(f"\nMatched: {len(matched)}  |  "
          f"Benchmark-only: {len(unmatched_b)}  |  "
          f"Prediction-only: {len(unmatched_p)}")

    if unmatched_b:
        print(f"  Benchmark IDs without predictions: {unmatched_b}")
    if unmatched_p:
        print(f"  Prediction IDs without benchmark: {unmatched_p}")

    if not matched:
        print("ERROR: No matched samples. Check that IDs align.")
        sys.exit(1)

    # 3. Score
    print(f"\nScoring {len(matched)} samples...")
    rubric = Rubric()
    results = score_all(matched, annotations, rubric)

    # 4. Aggregate
    overall = aggregate_overall(results)
    by_case = aggregate_by_group(results, "case_type")
    by_difficulty = aggregate_by_group(results, "difficulty")

    # 5. Analyze
    weak = analyze_weak_cases(results)
    errors = categorize_errors(results)
    error_dist = build_error_distribution(results)

    # 6. Output
    meta = {
        "benchmark_file": args.benchmark,
        "predictions_file": args.predictions,
        "annotations_file": args.annotations,
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_benchmark": len(benchmark),
        "total_predictions": len(predictions),
        "matched_samples": len(matched),
        "unmatched_benchmark_ids": unmatched_b,
        "unmatched_prediction_ids": unmatched_p,
    }

    save_report(meta, overall, by_case, by_difficulty, results, weak, errors, error_dist, args.output)
    print_summary(overall, by_case, weak, errors)
    generate_markdown_report(
        meta, overall, by_case, by_difficulty, results, weak, error_dist, tag=args.tag,
    )


if __name__ == "__main__":
    main()
