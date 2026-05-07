"""Offline retrieval evaluation — no server required.

Loads the persisted index (chunks.json + faiss.index + bm25 in-memory) via
KnowledgeBase and runs the eval_qa_dataset against it directly. Reuses the
matching helpers from scripts/eval_retrieval.py so numbers are comparable.

Usage:
    python scripts/eval_retrieval_offline.py --tag rule_baseline
    python scripts/eval_retrieval_offline.py --tag distilbert --output data/eval_reports/retrieval_distilbert.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Reuse judgment helpers from the online eval script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import (
    normalize,
    chunk_contains_answer,
    chunk_is_relevant,
    precision_at_k,
)

from app.rag.knowledge_base import KnowledgeBase

DEFAULT_DATASET = ROOT / "data" / "eval_qa_dataset.json"


def load_samples(path: Path) -> List[Dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw.get("samples", raw.get("data", [])) or []
    return raw


def first_hit_rank(answer: str, contexts: List[str]) -> int:
    """1-based rank of first chunk that contains the answer; 0 if no hit."""
    ans = normalize(answer)
    if not ans:
        return 0
    for i, ctx in enumerate(contexts, 1):
        if ans in normalize(ctx):
            return i
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--tag", default="run", help="report tag (used in output filename)")
    ap.add_argument("--output", default=None,
                    help="write detailed JSON report to this path")
    args = ap.parse_args()

    print(f"Loading KnowledgeBase from persisted index...")
    kb = KnowledgeBase.from_persisted_index()
    if kb is None:
        print("ERROR: no persisted index found. Run scripts/ingest_docs.py first.")
        sys.exit(1)
    print(f"Loaded {len(kb.chunks)} chunks")

    samples = load_samples(Path(args.dataset))
    print(f"Eval samples: {len(samples)}")

    # Suppress reranker per-query INFO logging — we already have the data
    import logging
    logging.getLogger("app.rag.knowledge_base").setLevel(logging.WARNING)

    n = len(samples)
    n_hit1 = 0
    n_hit5 = 0
    sum_rr = 0.0
    sum_p5 = 0.0

    # Per-block-type breakdown: bucket each query by the block_type of its
    # FIRST hit chunk (proxy for "what type of evidence answers this query")
    by_block: Dict[str, Dict[str, int]] = {}

    per_sample: List[Dict] = []
    start = time.time()
    for i, s in enumerate(samples, 1):
        question = s.get("question", "")
        gold_answer = s.get("answer", "")
        evidence = s.get("evidence", "")
        gold_chunks = s.get("gold_chunks", []) or []
        gold_strings = [g for g in [gold_answer, evidence, *gold_chunks] if g]

        retrieved = kb.retrieve(question, top_n=args.top_k)
        contexts = [c.text for c, _ in retrieved]
        block_types = [c.metadata.get("block_type", "?") for c, _ in retrieved]

        rank = first_hit_rank(gold_answer, contexts)
        hit1 = rank == 1
        hit5 = 0 < rank <= args.top_k
        rr = 1.0 / rank if rank else 0.0
        p5 = precision_at_k(contexts, gold_strings, args.top_k)

        n_hit1 += int(hit1)
        n_hit5 += int(hit5)
        sum_rr += rr
        sum_p5 += p5

        # Bucket by first-hit block_type
        first_hit_bt = block_types[rank - 1] if rank else "MISS"
        bucket = by_block.setdefault(first_hit_bt, {"total": 0, "hit5": 0})
        bucket["total"] += 1
        if hit5:
            bucket["hit5"] += 1

        per_sample.append({
            "id": s.get("id"),
            "question": question,
            "answer": gold_answer,
            "rank": rank,
            "hit1": hit1,
            "hit5": hit5,
            "p5": p5,
            "first_hit_block_type": first_hit_bt,
            "retrieved_block_types": block_types,
        })

        if i % 10 == 0 or i == n:
            elapsed = time.time() - start
            print(f"  [{i}/{n}]  hit5={n_hit5/i:.3f}  mrr={sum_rr/i:.3f}  "
                  f"p5={sum_p5/i:.3f}  ({elapsed:.0f}s)")

    summary = {
        "tag": args.tag,
        "samples": n,
        "top_k": args.top_k,
        "hit1": n_hit1 / n,
        "hit5": n_hit5 / n,
        "mrr": sum_rr / n,
        "precision_at_5": sum_p5 / n,
        "by_block_type": {
            bt: {
                "total": v["total"],
                "hit5": v["hit5"],
                "hit5_rate": v["hit5"] / v["total"] if v["total"] else 0.0,
            } for bt, v in by_block.items()
        },
        # Distribution of block_types across all retrieved chunks (top_k * n)
        "retrieved_block_dist": dict(Counter(
            bt for r in per_sample for bt in r["retrieved_block_types"]
        )),
    }

    print(f"\n=== Retrieval eval [{args.tag}] ===")
    print(f"Hit@1          : {summary['hit1']:.4f}")
    print(f"Hit@{args.top_k}          : {summary['hit5']:.4f}")
    print(f"MRR            : {summary['mrr']:.4f}")
    print(f"Precision@{args.top_k}    : {summary['precision_at_5']:.4f}")
    print(f"\nBy first-hit block_type (where the gold evidence lives):")
    for bt in ("text", "code", "formula", "table", "MISS"):
        v = summary["by_block_type"].get(bt)
        if not v:
            continue
        print(f"  {bt:6s}  total={v['total']:3d}  hit@{args.top_k}={v['hit5']:3d}  "
              f"rate={v['hit5_rate']:.3f}")
    print(f"\nRetrieved-chunk block_type distribution (top_{args.top_k} × {n} queries):")
    total_retrieved = sum(summary["retrieved_block_dist"].values())
    for bt in ("text", "code", "formula", "table"):
        n_ret = summary["retrieved_block_dist"].get(bt, 0)
        pct = n_ret / total_retrieved * 100 if total_retrieved else 0.0
        print(f"  {bt:8s} {n_ret:5d}  ({pct:5.1f}%)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "summary": summary,
            "per_sample": per_sample,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
