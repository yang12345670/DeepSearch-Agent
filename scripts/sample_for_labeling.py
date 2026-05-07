"""Stratified sample of chunks for labeling (Phase 1).

Reads data/labeling/raw_chunks.jsonl, assigns each chunk a heuristic
pre-label (text/code/formula/table), then samples per class so labelers
see a balanced mix instead of 95% prose.

The pre-label is ONLY used for sampling. Humans assign the ground truth.

Usage:
    python scripts/sample_for_labeling.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "labeling" / "raw_chunks.jsonl"
OUTPUT = ROOT / "data" / "labeling" / "to_label.jsonl"

# Target sample counts per heuristic class. Real distribution will differ
# after human labeling (humans correct heuristic errors), but this gives
# enough examples in rare classes for the model to learn from.
TARGETS = {
    "code": 400,
    "formula": 250,
    "table": 150,
    "text": 700,
}

SEED = 42

# ------------------------------------------------------------------
# Heuristic pre-classifiers (loose — over-recall, humans correct)
# ------------------------------------------------------------------

_CODE_KEYWORDS = re.compile(
    r"\b(def|class|return|import|from|function|const|let|var|public|private|"
    r"async|await|lambda|elif|=>|->|::|println|System\.out|console\.log)\b"
)
_OPERATORS = re.compile(r"[{};=<>+\-*/%&|^!]=?|==|!=|<=|>=|&&|\|\|")


def is_codey(text: str) -> bool:
    """Loose code detection: indentation OR keywords OR operator density."""
    lines = text.split("\n")
    indented = sum(1 for ln in lines if re.match(r"^( {2,}|\t)", ln))
    if len(lines) >= 2 and indented / max(len(lines), 1) >= 0.3:
        return True
    if len(_CODE_KEYWORDS.findall(text)) >= 2:
        return True
    if len(_OPERATORS.findall(text)) >= 5 and len(text) >= 60:
        return True
    return False


def is_formula(text: str) -> bool:
    if "$$" in text or re.search(r"\$[^$\n]{2,}\$", text):
        return True
    if re.search(r"\\(frac|sum|int|sqrt|alpha|beta|gamma|Delta|nabla|partial|prod|lim)\b", text):
        return True
    if re.search(r"\b(P|Q)\([^)]*\|[^)]*\)\s*[=≈]", text):  # P(x|y) =
        return True
    return False


def is_table(text: str) -> bool:
    pipe_lines = [ln for ln in text.split("\n") if ln.count("|") >= 2]
    if len(pipe_lines) < 2:
        return False
    has_separator = any(re.match(r"^\s*\|?[\s\-:|]+$", ln) and "-" in ln for ln in pipe_lines)
    return has_separator or len(pipe_lines) >= 3


def heuristic_label(text: str) -> str:
    # Order matters: formula and table are more distinctive, check before code
    if is_formula(text):
        return "formula"
    if is_table(text):
        return "table"
    if is_codey(text):
        return "code"
    return "text"


def main() -> None:
    if not INPUT.exists():
        print(f"Missing {INPUT}. Run dump_chunks_for_labeling.py first.")
        return

    buckets: Dict[str, List[dict]] = {k: [] for k in TARGETS}
    total = 0
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cls = heuristic_label(rec["text"])
            rec["heuristic"] = cls
            buckets[cls].append(rec)
            total += 1

    print(f"Heuristic pre-distribution over {total} chunks:")
    for cls, recs in buckets.items():
        print(f"  {cls:8s}: {len(recs):5d}")

    rng = random.Random(SEED)
    sampled: List[dict] = []
    for cls, target in TARGETS.items():
        pool = buckets[cls]
        take = min(target, len(pool))
        sampled.extend(rng.sample(pool, take))
        print(f"Sampled {take}/{target} from class={cls}")

    rng.shuffle(sampled)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sampled)} chunks to {OUTPUT}")


if __name__ == "__main__":
    main()
