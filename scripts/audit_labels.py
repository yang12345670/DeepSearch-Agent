"""Blind human audit of LLM labels on the held-out test split.

Goal: produce a defensible "LLM-label vs human-label agreement rate" number for
the resume / eval report, BEFORE comparing rule baseline vs DistilBERT.

Design:
- Blind: shows chunk text only, NOT the LLM label. You judge first, reveal after.
- Small classes get FULL coverage (table=7, formula=13 in test.jsonl). Code/text
  are stratified-sampled to fill the remaining budget.
- Resumable: re-running skips already-audited chunk_ids.
- Independent file: writes to data/labeling/test_audit.jsonl. Never modifies
  test.jsonl — the LLM labels remain the gold for downstream eval (you only
  use this audit to *report* labeling quality, not to retrain the eval set).

Usage:
    python scripts/audit_labels.py            # default budget=50
    python scripts/audit_labels.py --budget 80
    python scripts/audit_labels.py --report   # skip audit, just print report

Keys during audit:
    1=text  2=code  3=formula  4=table  |  s=skip  b=back  q=quit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "labeling" / "test.jsonl"
AUDIT_PATH = ROOT / "data" / "labeling" / "test_audit.jsonl"

LABELS = ["text", "code", "formula", "table"]
LABEL_MAP = {"1": "text", "2": "code", "3": "formula", "4": "table"}


def load_test() -> List[dict]:
    if not TEST_PATH.exists():
        print(f"Missing {TEST_PATH}. Run split_labels.py first.")
        sys.exit(1)
    with TEST_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_audited() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not AUDIT_PATH.exists():
        return out
    with AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["chunk_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def append_audit(rec: dict, human_label: str) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "chunk_id": rec["chunk_id"],
        "source": rec["source"],
        "llm_label": rec["label"],
        "human_label": human_label,
        "agree": rec["label"] == human_label,
        "heuristic": rec.get("heuristic"),
        "text": rec["text"],
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rewrite_audit(records: List[dict]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def select_audit_pool(test: List[dict], budget: int, seed: int = 42) -> List[dict]:
    """Full coverage of small classes (table, formula); stratified random for big ones."""
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for r in test:
        by_class[r["label"]].append(r)

    rng = random.Random(seed)
    chosen: List[dict] = []

    # Small classes: take all
    for c in ("table", "formula"):
        chosen.extend(by_class.get(c, []))

    remaining = max(0, budget - len(chosen))
    big_pool = by_class.get("code", []) + by_class.get("text", [])
    n_code, n_text = len(by_class.get("code", [])), len(by_class.get("text", []))
    if remaining and (n_code + n_text):
        # Stratify proportional to original ratio
        code_target = round(remaining * n_code / (n_code + n_text))
        text_target = remaining - code_target
        code_sample = rng.sample(by_class.get("code", []), min(code_target, n_code))
        text_sample = rng.sample(by_class.get("text", []), min(text_target, n_text))
        chosen.extend(code_sample)
        chosen.extend(text_sample)

    rng.shuffle(chosen)  # randomize presentation order
    return chosen


def render_chunk(rec: dict, done_count: int, total: int) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"[{done_count + 1} / {total}]  source={rec['source']}  heuristic={rec.get('heuristic')}")
    print(f"{sep}")
    print(rec["text"])
    print(sep)
    print("Your call → 1=text  2=code  3=formula  4=table  |  s=skip  b=back  q=quit")


def reveal(rec: dict, human_label: str) -> None:
    llm = rec["label"]
    if human_label == llm:
        print(f"  ✓ AGREE     LLM said: {llm}")
    else:
        print(f"  ✗ DISAGREE  LLM said: {llm}  |  You said: {human_label}")


def print_report() -> None:
    audited = load_audited()
    n = len(audited)
    if n == 0:
        print("No audit data yet.")
        return

    n_agree = sum(1 for r in audited.values() if r["agree"])
    overall = n_agree / n

    per_class_total: Counter = Counter()
    per_class_agree: Counter = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)  # rows=human (true-ish), cols=llm
    for r in audited.values():
        h, l = r["human_label"], r["llm_label"]
        per_class_total[h] += 1
        if h == l:
            per_class_agree[h] += 1
        confusion[h][l] += 1

    print(f"\n=== LLM label vs human audit ===")
    print(f"Audited:    {n}")
    print(f"Agreement:  {n_agree}/{n}  =  {overall*100:.1f}%")
    print()
    print(f"{'class':10s} {'agreed':>8s} {'total':>8s} {'rate':>8s}")
    for c in LABELS:
        t = per_class_total.get(c, 0)
        a = per_class_agree.get(c, 0)
        rate = (a / t * 100) if t else 0.0
        print(f"{c:10s} {a:>8d} {t:>8d} {rate:>7.1f}%")

    print("\nConfusion (rows=human truth, cols=LLM said):")
    print(" " * 10 + "".join(f"{c:>10s}" for c in LABELS))
    for h in LABELS:
        row = "".join(f"{confusion[h].get(c, 0):>10d}" for c in LABELS)
        print(f"{h:10s}{row}")

    disagreements = [r for r in audited.values() if not r["agree"]]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for r in disagreements[:20]:
            preview = r["text"].replace("\n", " ")[:70]
            print(f"  human={r['human_label']:8s} llm={r['llm_label']:8s}  | {preview}...")
        if len(disagreements) > 20:
            print(f"  ... and {len(disagreements) - 20} more (see {AUDIT_PATH})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="Skip audit; just print agreement report from existing audit file")
    args = ap.parse_args()

    if args.report:
        print_report()
        return

    test = load_test()
    audited = load_audited()
    pool = select_audit_pool(test, args.budget, seed=args.seed)
    pending = [r for r in pool if r["chunk_id"] not in audited]

    if not pending:
        print(f"All {len(pool)} pool items already audited.")
        print_report()
        return

    print(f"Pool: {len(pool)}  Already audited: {len(audited)}  Pending: {len(pending)}")
    print("Blind mode: you'll judge first, then see what the LLM said.\n")

    # History for 'back' support
    history: List[dict] = []
    if AUDIT_PATH.exists():
        with AUDIT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    i = 0
    while i < len(pending):
        rec = pending[i]
        render_chunk(rec, len(history), len(pool))
        try:
            ans = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSaved. Bye.")
            break

        if ans == "q":
            print("Saved. Bye.")
            break
        if ans == "s":
            i += 1
            continue
        if ans == "b":
            if not history:
                print("(nothing to go back to)")
                continue
            last = history.pop()
            rewrite_audit(history)
            audited = load_audited()
            pending = [r for r in pool if r["chunk_id"] not in audited]
            target = last["chunk_id"]
            i = next((k for k, r in enumerate(pending) if r["chunk_id"] == target), i)
            continue

        human_label: Optional[str] = LABEL_MAP.get(ans)
        if not human_label:
            print(f"Unknown key '{ans}'. Use 1/2/3/4 or s/b/q.")
            continue

        append_audit(rec, human_label)
        reveal(rec, human_label)
        history.append({
            "chunk_id": rec["chunk_id"],
            "source": rec["source"],
            "llm_label": rec["label"],
            "human_label": human_label,
            "agree": rec["label"] == human_label,
            "heuristic": rec.get("heuristic"),
            "text": rec["text"],
        })
        i += 1

    print_report()


if __name__ == "__main__":
    main()
