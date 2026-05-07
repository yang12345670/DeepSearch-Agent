"""Interactive labeling CLI for chunk block-type classifier (Phase 1).

Reads chunks from data/labeling/to_label.jsonl, shows each one, and asks
the labeler to press 1/2/3/4 for text/code/formula/table.

Resumable: already-labeled chunk_ids in data/labeling/labeled.jsonl are
skipped. Each label is fsync'd immediately so Ctrl+C never loses work.

Keys:
  1 = text     2 = code     3 = formula     4 = table
  s = skip     b = back (re-label previous)     q = save & quit
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Windows consoles default to GBK; force UTF-8 so Chinese / emoji / math symbols print
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
TO_LABEL = ROOT / "data" / "labeling" / "to_label.jsonl"
LABELED = ROOT / "data" / "labeling" / "labeled.jsonl"

LABEL_MAP = {"1": "text", "2": "code", "3": "formula", "4": "table"}


def load_done() -> Dict[str, str]:
    done: Dict[str, str] = {}
    if not LABELED.exists():
        return done
    with LABELED.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["chunk_id"]] = rec["label"]
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_label(rec: dict, label: str) -> None:
    LABELED.parent.mkdir(parents=True, exist_ok=True)
    with LABELED.open("a", encoding="utf-8") as f:
        out = {
            "chunk_id": rec["chunk_id"],
            "source": rec["source"],
            "label": label,
            "heuristic": rec.get("heuristic"),
            "text": rec["text"],
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rewrite_labeled(records: List[dict]) -> None:
    """Rewrite the entire labeled file (used for 'back' to undo last)."""
    LABELED.parent.mkdir(parents=True, exist_ok=True)
    with LABELED.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def render_chunk(rec: dict, idx: int, total: int, done_count: int) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"[{done_count + 1} / {total}]  source={rec['source']}  heuristic={rec.get('heuristic')}")
    print(f"{sep}")
    print(rec["text"])
    print(sep)
    print("1=text  2=code  3=formula  4=table  |  s=skip  b=back  q=quit")


def main() -> None:
    if not TO_LABEL.exists():
        print(f"Missing {TO_LABEL}. Run sample_for_labeling.py first.")
        return

    with TO_LABEL.open("r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    done = load_done()
    pending = [r for r in all_records if r["chunk_id"] not in done]

    if not pending:
        print(f"All {len(all_records)} chunks already labeled. Done.")
        _print_distribution(done)
        return

    print(f"Total: {len(all_records)}  Already labeled: {len(done)}  Pending: {len(pending)}")

    # Track labeled records in this session for 'back' support
    history: List[dict] = []  # list of full label records (matching labeled.jsonl format)
    if LABELED.exists():
        with LABELED.open("r", encoding="utf-8") as f:
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
        render_chunk(rec, i, len(all_records), len(history))
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
            rewrite_labeled(history)
            # find the corresponding pending record and put it back at current position
            # easiest: rewind one step in pending if previous was in pending list
            #   (it should be — we just labeled it)
            # We need to re-locate: the popped record's chunk_id should match pending[i-1]
            # But pending was filtered to exclude already-done. So we need to splice it back.
            # Simpler approach: rebuild pending from scratch.
            done = load_done()
            pending = [r for r in all_records if r["chunk_id"] not in done]
            # find new index for the un-labeled chunk
            target_id = last["chunk_id"]
            new_i = next((k for k, r in enumerate(pending) if r["chunk_id"] == target_id), i)
            i = new_i
            continue

        label = LABEL_MAP.get(ans)
        if not label:
            print(f"Unknown key '{ans}'. Use 1/2/3/4 or s/b/q.")
            continue

        append_label(rec, label)
        history.append({
            "chunk_id": rec["chunk_id"],
            "source": rec["source"],
            "label": label,
            "heuristic": rec.get("heuristic"),
            "text": rec["text"],
        })
        i += 1

    _print_distribution(load_done())


def _print_distribution(done: Dict[str, str]) -> None:
    if not done:
        return
    counts: Dict[str, int] = {}
    for lbl in done.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    print("\nCurrent label distribution:")
    for lbl in ("text", "code", "formula", "table"):
        print(f"  {lbl:8s}: {counts.get(lbl, 0)}")
    print(f"  total   : {sum(counts.values())}")


if __name__ == "__main__":
    main()
