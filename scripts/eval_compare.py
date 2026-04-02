# -*- coding: utf-8 -*-
"""
Multi-Setting Comparison for Answer-Layer Evaluation.

Reads multiple eval result JSONs (produced by eval_answer_offline.py)
and generates a side-by-side comparison report.

Workflow:
  1. Run eval_answer_offline.py once per retrieval setting
  2. Run this script to compare all results

Example:
  # Step 1: Generate predictions for each setting
  python scripts/prediction_adapter.py live --benchmark data/answer_eval_dataset.jsonl \
    --output data/predictions/bm25.jsonl
  python scripts/prediction_adapter.py live --benchmark data/answer_eval_dataset.jsonl \
    --output data/predictions/dense.jsonl
  python scripts/prediction_adapter.py live --benchmark data/answer_eval_dataset.jsonl \
    --output data/predictions/hybrid.jsonl
  python scripts/prediction_adapter.py live --benchmark data/answer_eval_dataset.jsonl \
    --output data/predictions/hybrid_reranker.jsonl

  # Step 2: Evaluate each
  python scripts/eval_answer_offline.py --predictions data/predictions/bm25.jsonl \
    --output data/eval_results/bm25.json --tag bm25
  python scripts/eval_answer_offline.py --predictions data/predictions/dense.jsonl \
    --output data/eval_results/dense.json --tag dense
  python scripts/eval_answer_offline.py --predictions data/predictions/hybrid.jsonl \
    --output data/eval_results/hybrid.json --tag hybrid
  python scripts/eval_answer_offline.py --predictions data/predictions/hybrid_reranker.jsonl \
    --output data/eval_results/hybrid_reranker.json --tag hybrid_reranker

  # Step 3: Compare
  python scripts/eval_compare.py \
    --results data/eval_results/bm25.json data/eval_results/dense.json \
              data/eval_results/hybrid.json data/eval_results/hybrid_reranker.json \
    --output data/eval_comparison.json

  # Or use glob:
  python scripts/eval_compare.py --results data/eval_results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ====================================================================
# File Naming Convention
# ====================================================================
#
# Predictions:   data/predictions/{setting}.jsonl
# Eval results:  data/eval_results/{setting}.json
# Comparison:    data/eval_comparison.json
#                data/eval_reports/comparison_YYYY-MM-DD_{tag}.md
#
# The "setting" name is extracted from each result file's meta.tag field.
# If meta.tag is missing, the filename stem is used.
# ====================================================================


# ====================================================================
# Loading
# ====================================================================

def load_result(path: str) -> Dict[str, Any]:
    """Load one eval result JSON and extract its tag + metrics."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Resolve setting name: meta.tag > filename stem
    tag = data.get("meta", {}).get("tag", "")
    if not tag or tag == "run":
        tag = Path(path).stem

    return {
        "tag": tag,
        "path": path,
        "overall": data.get("overall", {}),
        "by_case_type": data.get("by_case_type", {}),
        "by_difficulty": data.get("by_difficulty", {}),
        "error_distribution": data.get("error_distribution", {}),
        "per_sample": data.get("per_sample", []),
        "meta": data.get("meta", {}),
    }


def load_all_results(paths: List[str]) -> List[Dict[str, Any]]:
    """Load multiple eval result JSONs."""
    results = []
    for p in paths:
        if not Path(p).is_file():
            print(f"  WARNING: {p} not found, skipped")
            continue
        try:
            results.append(load_result(p))
            print(f"  Loaded: {p}")
        except Exception as e:
            print(f"  ERROR loading {p}: {e}")
    return results


# ====================================================================
# Comparison Logic
# ====================================================================

# Metrics to compare, with direction (higher/lower = better)
METRICS = [
    ("avg_composite",       "Composite",              "higher"),
    ("avg_m1_accuracy",     "M1 Accuracy",            "higher"),
    ("avg_m2_groundedness", "M2 Groundedness",        "higher"),
    ("avg_m3_unsupported",  "M3 Unsupported Claims",  "lower"),
    ("avg_m4_refusal",      "M4 Refusal",             "higher"),
    ("avg_m5_partial",      "M5 Partial Compliance",  "higher"),
    ("avg_m6_noise",        "M6 Noise Ratio",         "lower"),
    ("all_pass_rate",       "All-Pass Rate",           "higher"),
]


def build_comparison_table(settings: List[Dict]) -> Dict[str, Any]:
    """Build the core metric comparison across settings.

    Returns:
        {
            "metrics": [...],   # row definitions
            "settings": [...],  # column headers (tag names)
            "rows": [           # one per metric
                {
                    "key": "avg_composite",
                    "label": "Composite",
                    "direction": "higher",
                    "values": {"bm25": 0.78, "dense": 0.82, ...},
                    "best": "hybrid_reranker",
                    "worst": "bm25",
                    "delta": 0.12,  # best - worst
                }
            ]
        }
    """
    tags = [s["tag"] for s in settings]
    rows = []

    for key, label, direction in METRICS:
        values = {}
        for s in settings:
            values[s["tag"]] = s["overall"].get(key, 0.0)

        if direction == "higher":
            best_tag = max(values, key=values.get)
            worst_tag = min(values, key=values.get)
        else:
            best_tag = min(values, key=values.get)
            worst_tag = max(values, key=values.get)

        delta = abs(values[best_tag] - values[worst_tag])

        rows.append({
            "key": key,
            "label": label,
            "direction": direction,
            "values": values,
            "best": best_tag,
            "worst": worst_tag,
            "delta": round(delta, 4),
        })

    return {
        "settings": tags,
        "rows": rows,
    }


def build_case_type_comparison(settings: List[Dict]) -> Dict[str, Any]:
    """Compare composite scores per case_type across settings."""
    case_types = set()
    for s in settings:
        case_types.update(s["by_case_type"].keys())

    table = {}
    for ct in sorted(case_types):
        row = {}
        for s in settings:
            ct_data = s["by_case_type"].get(ct, {})
            row[s["tag"]] = ct_data.get("avg_composite", 0.0)

        best = max(row, key=row.get) if row else ""
        table[ct] = {"values": row, "best": best}

    return table


def build_per_sample_comparison(settings: List[Dict]) -> List[Dict[str, Any]]:
    """Compare composite scores per sample across settings.

    Returns list of samples where settings disagree most (largest spread).
    """
    # Collect all sample IDs
    all_ids = set()
    for s in settings:
        for r in s["per_sample"]:
            all_ids.add(r["id"])

    # Build per-sample comparison
    samples = []
    for sid in sorted(all_ids):
        row = {"id": sid}
        scores = {}
        for s in settings:
            match = next((r for r in s["per_sample"] if r["id"] == sid), None)
            if match:
                scores[s["tag"]] = match["composite_score"]
                if "case_type" not in row:
                    row["case_type"] = match.get("case_type", "")
                    row["question"] = match.get("question", "")[:60]
        row["scores"] = scores

        vals = list(scores.values())
        row["spread"] = round(max(vals) - min(vals), 4) if vals else 0.0
        row["best"] = max(scores, key=scores.get) if scores else ""
        row["worst"] = min(scores, key=scores.get) if scores else ""
        samples.append(row)

    # Sort by spread descending (most disagreement first)
    samples.sort(key=lambda x: -x["spread"])
    return samples


def build_improvement_summary(table: Dict[str, Any]) -> Dict[str, Any]:
    """Determine which setting wins each metric category."""
    winners: Dict[str, str] = {}
    for row in table["rows"]:
        winners[row["label"]] = row["best"]

    # Count wins per setting
    win_counts: Dict[str, int] = {}
    for tag in table["settings"]:
        win_counts[tag] = sum(1 for w in winners.values() if w == tag)

    overall_best = max(win_counts, key=win_counts.get) if win_counts else ""

    return {
        "metric_winners": winners,
        "win_counts": win_counts,
        "overall_best": overall_best,
    }


def build_error_comparison(settings: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Compare error tag counts across settings."""
    all_tags = set()
    for s in settings:
        ed = s.get("error_distribution", {})
        all_tags.update(ed.get("error_counts", {}).keys())

    table = {}
    for tag_name in sorted(all_tags):
        row = {}
        for s in settings:
            ed = s.get("error_distribution", {})
            row[s["tag"]] = ed.get("error_counts", {}).get(tag_name, 0)
        table[tag_name] = row

    return table


# ====================================================================
# Output
# ====================================================================

def save_comparison_json(
    meta: Dict,
    table: Dict,
    case_table: Dict,
    per_sample: List[Dict],
    improvement: Dict,
    error_table: Dict,
    path: str,
) -> None:
    """Save comparison report as JSON."""
    report = {
        "meta": meta,
        "comparison_table": table,
        "by_case_type": case_table,
        "per_sample_spread": per_sample[:20],  # Top 20 most divergent
        "improvement_summary": improvement,
        "error_comparison": error_table,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nComparison JSON saved to: {path}")


def generate_comparison_markdown(
    meta: Dict,
    table: Dict,
    case_table: Dict,
    per_sample: List[Dict],
    improvement: Dict,
    error_table: Dict,
    settings: List[Dict],
    *,
    tag: str,
) -> str:
    """Generate a human-readable markdown comparison report."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"comparison_{date_str}_{tag}.md"
    report_dir = Path("data/eval_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / filename

    tags = table["settings"]
    L: List[str] = []

    # --- Header ---
    L.append(f"# Retrieval Setting Comparison: {tag}")
    L.append("")
    L.append(f"- **Date**: {time_str}")
    L.append(f"- **Settings compared**: {', '.join(tags)}")
    L.append(f"- **Benchmark samples**: {meta.get('sample_count', '?')}")
    L.append("")

    # --- Main Comparison Table ---
    L.append("## Overall Metrics")
    L.append("")

    # Header row
    header = "| Metric | " + " | ".join(tags) + " | Best | Delta |"
    sep = "|--------|" + "|".join(["------"] * len(tags)) + "|------|-------|"
    L.append(header)
    L.append(sep)

    for row in table["rows"]:
        cells = []
        for t in tags:
            val = row["values"].get(t, 0.0)
            # Bold the best value
            if t == row["best"]:
                cells.append(f"**{val:.2f}**")
            else:
                cells.append(f"{val:.2f}")
        arrow = "^" if row["direction"] == "higher" else "v"
        label = row["label"]
        if row["direction"] == "lower":
            label += " (v)"

        L.append(f"| {label} | {' | '.join(cells)} | {row['best']} | {row['delta']:.2f} |")
    L.append("")

    # --- Winner Summary ---
    L.append("## Winner Summary")
    L.append("")
    L.append(f"**Overall best setting: `{improvement['overall_best']}`** "
             f"({improvement['win_counts'].get(improvement['overall_best'], 0)}/{len(table['rows'])} metrics won)")
    L.append("")
    L.append("| Metric | Winner |")
    L.append("|--------|--------|")
    for metric, winner in improvement["metric_winners"].items():
        L.append(f"| {metric} | `{winner}` |")
    L.append("")

    # Win count bar
    L.append("**Win counts:**")
    L.append("")
    for t in tags:
        count = improvement["win_counts"].get(t, 0)
        bar = "#" * count
        L.append(f"- `{t}`: {bar} ({count})")
    L.append("")

    # --- By Case Type ---
    L.append("## By Case Type (Composite)")
    L.append("")
    header = "| Case Type | " + " | ".join(tags) + " | Best |"
    sep = "|-----------|" + "|".join(["------"] * len(tags)) + "|------|"
    L.append(header)
    L.append(sep)
    for ct, data in sorted(case_table.items()):
        cells = []
        for t in tags:
            val = data["values"].get(t, 0.0)
            if t == data["best"]:
                cells.append(f"**{val:.2f}**")
            else:
                cells.append(f"{val:.2f}")
        L.append(f"| {ct} | {' | '.join(cells)} | {data['best']} |")
    L.append("")

    # --- Error Comparison ---
    if error_table:
        L.append("## Error Tag Comparison")
        L.append("")
        header = "| Error Tag | " + " | ".join(tags) + " |"
        sep = "|-----------|" + "|".join(["------"] * len(tags)) + "|"
        L.append(header)
        L.append(sep)
        for err_tag, counts in sorted(error_table.items()):
            cells = []
            vals = list(counts.values())
            min_val = min(vals) if vals else 0
            for t in tags:
                v = counts.get(t, 0)
                if v == min_val and v < max(vals):
                    cells.append(f"**{v}**")
                else:
                    cells.append(str(v))
            L.append(f"| `{err_tag}` | {' | '.join(cells)} |")
        L.append("")

    # --- Most Divergent Samples ---
    divergent = [s for s in per_sample if s["spread"] > 0]
    if divergent:
        L.append("## Most Divergent Samples")
        L.append("")
        L.append(f"Samples where retrieval settings disagree the most (top {min(10, len(divergent))}):")
        L.append("")
        header = "| ID | Type | " + " | ".join(tags) + " | Spread | Best | Worst |"
        sep = "|----|------|" + "|".join(["------"] * len(tags)) + "|--------|------|-------|"
        L.append(header)
        L.append(sep)
        for s in divergent[:10]:
            cells = []
            for t in tags:
                val = s["scores"].get(t, 0.0)
                cells.append(f"{val:.2f}")
            ct_short = s.get("case_type", "")[:8]
            L.append(f"| {s['id']} | {ct_short} | {' | '.join(cells)} | "
                     f"{s['spread']:.2f} | {s['best']} | {s['worst']} |")
        L.append("")

    # --- Footer ---
    L.append(f"*Generated by eval_compare.py at {time_str}*")
    L.append("")

    content = "\n".join(L)
    report_path.write_text(content, encoding="utf-8")
    print(f"Markdown comparison saved to: {report_path}")
    return str(report_path)


def print_comparison_summary(
    table: Dict,
    improvement: Dict,
) -> None:
    """Print compact console comparison."""
    tags = table["settings"]
    w = 15 + 10 * len(tags)
    print()
    print("=" * w)
    print("  RETRIEVAL SETTING COMPARISON")
    print("=" * w)

    # Header
    header = f"  {'Metric':<25s}"
    for t in tags:
        header += f" {t:>9s}"
    header += f" {'Best':>9s}"
    print(header)
    print("-" * w)

    for row in table["rows"]:
        line = f"  {row['label']:<25s}"
        for t in tags:
            val = row["values"].get(t, 0.0)
            marker = " *" if t == row["best"] else "  "
            line += f" {val:>7.2f}{marker}"
        line += f" {row['best']:>9s}"
        print(line)

    print("-" * w)
    best = improvement["overall_best"]
    wins = improvement["win_counts"].get(best, 0)
    total = len(table["rows"])
    print(f"  Overall best: {best}  ({wins}/{total} metrics)")
    print("=" * w)


# ====================================================================
# CLI
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple retrieval settings using eval result JSONs."
    )
    parser.add_argument("--results", nargs="+", required=True,
                        help="Eval result JSON files (from eval_answer_offline.py).")
    parser.add_argument("--output", default="data/eval_comparison.json",
                        help="Output comparison JSON path.")
    parser.add_argument("--tag", default="compare",
                        help="Tag for this comparison run.")
    args = parser.parse_args()

    # Load
    print("Loading eval results...")
    settings = load_all_results(args.results)

    if len(settings) < 2:
        print(f"ERROR: Need at least 2 result files to compare, got {len(settings)}.")
        sys.exit(1)

    print(f"\nComparing {len(settings)} settings: {[s['tag'] for s in settings]}")

    # Compare
    table = build_comparison_table(settings)
    case_table = build_case_type_comparison(settings)
    per_sample = build_per_sample_comparison(settings)
    improvement = build_improvement_summary(table)
    error_table = build_error_comparison(settings)

    # Meta
    sample_count = max(len(s["per_sample"]) for s in settings) if settings else 0
    meta = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": [s["tag"] for s in settings],
        "result_files": args.results,
        "sample_count": sample_count,
    }

    # Output
    save_comparison_json(meta, table, case_table, per_sample, improvement, error_table, args.output)
    print_comparison_summary(table, improvement)
    generate_comparison_markdown(
        meta, table, case_table, per_sample, improvement, error_table, settings, tag=args.tag,
    )


if __name__ == "__main__":
    main()
