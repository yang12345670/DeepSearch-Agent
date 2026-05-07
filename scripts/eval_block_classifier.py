"""Evaluate a block-type classifier on labeled chunks.

Default backend: rule (the heuristic in app/rag/chunker.py:classify_block_rule).
After training, pass --backend distilbert to evaluate the trained model.

Usage:
    python scripts/eval_block_classifier.py --backend rule
    python scripts/eval_block_classifier.py --backend distilbert --model models/block_classifier
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.rag.chunker import classify_block_rule

LABELS = ["text", "code", "formula", "table"]
LABELED_PATH = ROOT / "data" / "labeling" / "labeled.jsonl"
TEST_SPLIT_PATH = ROOT / "data" / "labeling" / "test.jsonl"


def load_test_set() -> List[dict]:
    """Prefer the held-out test split; fall back to labeled.jsonl if not split yet."""
    path = TEST_SPLIT_PATH if TEST_SPLIT_PATH.exists() else LABELED_PATH
    if not path.exists():
        raise FileNotFoundError(f"No labels found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_predictor(backend: str, model_path: str | None) -> Callable[[str], str]:
    if backend == "rule":
        return classify_block_rule
    if backend == "distilbert":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        id2label = model.config.id2label

        def predict(text: str) -> str:
            with torch.no_grad():
                enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
                out = model(**enc)
                idx = int(out.logits.argmax(-1).item())
            return id2label[idx]
        return predict
    raise ValueError(f"unknown backend {backend}")


def confusion_and_metrics(pairs: List[Tuple[str, str]]) -> Dict:
    cm: Dict[str, Dict[str, int]] = {a: defaultdict(int) for a in LABELS}
    for gold, pred in pairs:
        if gold not in cm:
            cm[gold] = defaultdict(int)
        cm[gold][pred] += 1

    metrics: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []
    for label in LABELS:
        tp = cm[label].get(label, 0)
        fp = sum(cm[other].get(label, 0) for other in LABELS if other != label)
        fn = sum(cm[label].get(other, 0) for other in LABELS if other != label)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        metrics[label] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
        f1_list.append(f1)

    correct = sum(1 for g, p in pairs if g == p)
    acc = correct / len(pairs) if pairs else 0.0
    macro_f1 = sum(f1_list) / len(f1_list)
    return {"per_class": metrics, "accuracy": acc, "macro_f1": macro_f1, "confusion": cm}


def print_report(report: Dict, backend: str, n: int) -> None:
    print(f"\n=== Block classifier eval: backend={backend}, n={n} ===")
    print(f"Accuracy : {report['accuracy']:.4f}")
    print(f"Macro F1 : {report['macro_f1']:.4f}")
    print()
    print(f"{'class':10s} {'P':>8s} {'R':>8s} {'F1':>8s} {'support':>8s}")
    for label in LABELS:
        m = report["per_class"][label]
        print(f"{label:10s} {m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f} {m['support']:8d}")

    print("\nConfusion (rows=gold, cols=pred):")
    print(" " * 10 + "".join(f"{l:>10s}" for l in LABELS))
    cm = report["confusion"]
    for gold in LABELS:
        row = "".join(f"{cm[gold].get(p, 0):>10d}" for p in LABELS)
        print(f"{gold:10s}{row}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rule", "distilbert"], default="rule")
    ap.add_argument("--model", default=str(ROOT / "models" / "block_classifier"),
                    help="model dir (only used for backend=distilbert)")
    ap.add_argument("--output", default=None,
                    help="optional path to save JSON report")
    args = ap.parse_args()

    test = load_test_set()
    predict = make_predictor(args.backend, args.model)

    pairs: List[Tuple[str, str]] = []
    for rec in test:
        gold = rec["label"]
        pred = predict(rec["text"])
        pairs.append((gold, pred))

    report = confusion_and_metrics(pairs)
    print_report(report, args.backend, len(pairs))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # confusion uses defaultdict — flatten for json
        report["confusion"] = {k: dict(v) for k, v in report["confusion"].items()}
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
