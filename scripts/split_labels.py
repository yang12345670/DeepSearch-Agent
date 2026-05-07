"""Split labeled.jsonl into train/val/test (stratified by label).

Default split: 80/10/10. Seed fixed for reproducibility.

Usage:
    python scripts/split_labels.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

LABELED = ROOT / "data" / "labeling" / "labeled.jsonl"
TRAIN = ROOT / "data" / "labeling" / "train.jsonl"
VAL = ROOT / "data" / "labeling" / "val.jsonl"
TEST = ROOT / "data" / "labeling" / "test.jsonl"

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remainder = test


def main() -> None:
    if not LABELED.exists():
        print(f"Missing {LABELED}.")
        sys.exit(1)

    by_label: Dict[str, List[dict]] = defaultdict(list)
    with LABELED.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_label[rec["label"]].append(rec)

    rng = random.Random(SEED)
    train, val, test = [], [], []
    for label, records in by_label.items():
        rng.shuffle(records)
        n = len(records)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train.extend(records[:n_train])
        val.extend(records[n_train: n_train + n_val])
        test.extend(records[n_train + n_val:])
        print(f"  {label:8s}: {n} -> train={n_train}, val={n_val}, test={n - n_train - n_val}")

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    for path, recs in [(TRAIN, train), (VAL, val), (TEST, test)]:
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(recs)} -> {path}")


if __name__ == "__main__":
    main()
