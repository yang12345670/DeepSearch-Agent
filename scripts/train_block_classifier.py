"""Train DistilBERT 4-class block-type classifier (Phase 3).

Inputs : data/labeling/train.jsonl, val.jsonl  (run split_labels.py first)
Output : models/block_classifier/  (HF Transformers format)

Defaults are tuned for ~1.2k train examples on CPU (5-30 min) or GPU (~3 min).

Usage:
    python scripts/train_block_classifier.py
    python scripts/train_block_classifier.py --epochs 8 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LABELS = ["text", "code", "formula", "table"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

TRAIN = ROOT / "data" / "labeling" / "train.jsonl"
VAL = ROOT / "data" / "labeling" / "val.jsonl"
OUTPUT_DIR = ROOT / "models" / "block_classifier"


def load_split(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}. Run scripts/split_labels.py first.")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    train_recs = load_split(TRAIN)
    val_recs = load_split(VAL)
    print(f"Loaded train={len(train_recs)}, val={len(val_recs)}")

    train_dist = Counter(r["label"] for r in train_recs)
    print("Train distribution:", dict(train_dist))

    # Class weights inversely proportional to frequency (balanced).
    counts = np.array([train_dist.get(l, 0) for l in LABELS], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(LABELS) * counts)
    print("Class weights:", {l: round(float(w), 3) for l, w in zip(LABELS, weights)})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class ChunkDataset(Dataset):
        def __init__(self, recs: List[Dict]):
            self.recs = recs

        def __len__(self): return len(self.recs)

        def __getitem__(self, idx):
            r = self.recs[idx]
            enc = tokenizer(
                r["text"],
                truncation=True,
                max_length=args.max_len,
                padding=False,
            )
            enc["labels"] = LABEL2ID[r["label"]]
            return enc

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)

    train_ds = ChunkDataset(train_recs)
    val_ds = ChunkDataset(val_recs)

    weights_t = torch.tensor(weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = nn.CrossEntropyLoss(weight=weights_t.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.argmax(-1)
        labels = eval_pred.label_ids
        f1s = []
        for cls_idx in range(len(LABELS)):
            tp = int(((preds == cls_idx) & (labels == cls_idx)).sum())
            fp = int(((preds == cls_idx) & (labels != cls_idx)).sum())
            fn = int(((preds != cls_idx) & (labels == cls_idx)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        acc = float((preds == labels).mean())
        return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=0,  # Windows safety
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    final_metrics = trainer.evaluate()
    print("\nFinal val metrics:", final_metrics)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
