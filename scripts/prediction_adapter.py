# -*- coding: utf-8 -*-
"""
Prediction Adapter — converts real DeepSearch Agent output into the
standard prediction format required by eval_answer_offline.py.

Supports three input modes:
  1. Live mode:  Hit the running agent API, collect and convert responses
  2. Log mode:   Read a raw agent output JSONL (e.g., from a batch run)
  3. Inline:     Call adapt_one() from Python for programmatic use

Output format (one per line in JSONL):
  {
    "id": "fs-101",
    "answer": "...",
    "used_evidence_ids": ["doc0_chunk3"],
    "final_context_ids": ["doc0_chunk3", "doc1_chunk0", ...],
    "meta": {"model": "gpt-4o-mini", "latency_ms": 1200, ...}
  }

Usage:
  # Live mode: query agent for each benchmark sample
  python scripts/prediction_adapter.py live \
    --benchmark data/answer_eval_dataset.jsonl \
    --api http://127.0.0.1:8000 \
    --output data/model_predictions.jsonl

  # Log mode: convert raw agent logs to prediction format
  python scripts/prediction_adapter.py convert \
    --input data/raw_agent_output.jsonl \
    --output data/model_predictions.jsonl

  # Then evaluate:
  python scripts/eval_answer_offline.py \
    --predictions data/model_predictions.jsonl --tag v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ====================================================================
# Field Mapping Rules
# ====================================================================
#
# Real Agent Field          →  Prediction Field       Fallback
# ────────────────────────  ─  ──────────────────     ──────────
# question                  →  (used for ID gen)      required
# final_answer / answer     →  answer                 ""
# retrieved_chunks[].id     →  final_context_ids      []
# reranked_chunks[].id      →  final_context_ids      (fallback to retrieved)
# final_context[].id        →  final_context_ids      (fallback to reranked)
# evidence_used             →  used_evidence_ids      []
# debug_trace               →  meta.debug_trace       null
# tool_trace                →  meta.tool_trace        null
# metadata.*                →  meta.*                 {}
# ====================================================================

# Fields that can hold the answer, in priority order
_ANSWER_FIELDS = ["final_answer", "answer", "response", "output", "reply"]

# Fields that can hold chunk lists, in priority order (most filtered → least)
_CONTEXT_FIELDS = ["final_context", "reranked_chunks", "retrieved_chunks", "retrieved_context"]


def _extract_answer(raw: Dict[str, Any]) -> str:
    """Extract the answer string from raw agent output.

    Tries multiple field names in priority order.
    Falls back to empty string.
    """
    for field in _ANSWER_FIELDS:
        val = raw.get(field)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_chunk_ids(chunks: Any) -> List[str]:
    """Extract chunk IDs from a chunk list.

    Handles multiple formats:
      - List[str]                    → use as-is
      - List[{"id": ...}]           → extract id field
      - List[{"chunk_id": ...}]     → extract chunk_id field
      - List[{"text": ...}]         → generate hash-based ID
    """
    if not chunks or not isinstance(chunks, list):
        return []

    ids = []
    for item in chunks:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            cid = (item.get("id")
                   or item.get("chunk_id")
                   or item.get("doc_id"))
            if cid:
                ids.append(str(cid))
            elif item.get("text"):
                # Generate a stable hash ID from text content
                h = hashlib.md5(item["text"].encode("utf-8")).hexdigest()[:8]
                ids.append(f"chunk_{h}")
        # Skip unknown types silently
    return ids


def _extract_context_ids(raw: Dict[str, Any]) -> List[str]:
    """Extract final_context_ids from raw output, trying multiple fields."""
    for field in _CONTEXT_FIELDS:
        val = raw.get(field)
        ids = _extract_chunk_ids(val)
        if ids:
            return ids
    return []


def _extract_used_evidence_ids(raw: Dict[str, Any]) -> List[str]:
    """Extract used_evidence_ids from raw output.

    Handles:
      - evidence_used: List[str]           → text snippets, generate hash IDs
      - used_evidence_ids: List[str]       → direct IDs
      - cited_chunks: List[dict]           → extract IDs
    """
    # Direct IDs
    direct = raw.get("used_evidence_ids")
    if direct and isinstance(direct, list):
        return _extract_chunk_ids(direct)

    # Cited chunks
    cited = raw.get("cited_chunks")
    if cited:
        return _extract_chunk_ids(cited)

    # evidence_used (text snippets from RAGResult) — generate hash IDs
    ev = raw.get("evidence_used")
    if ev and isinstance(ev, list):
        ids = []
        for item in ev:
            if isinstance(item, str) and len(item) > 20:
                h = hashlib.md5(item.encode("utf-8")).hexdigest()[:8]
                ids.append(f"ev_{h}")
            elif isinstance(item, str):
                ids.append(item)
        return ids

    return []


def _extract_meta(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata for the prediction record.

    Preserves debug information for later analysis.
    """
    meta: Dict[str, Any] = {}

    # Copy explicit metadata block
    if isinstance(raw.get("metadata"), dict):
        meta.update(raw["metadata"])
    if isinstance(raw.get("meta"), dict):
        meta.update(raw["meta"])

    # Model info
    for f in ["model", "model_name", "llm_model"]:
        if raw.get(f):
            meta["model"] = raw[f]
            break

    # Latency
    for f in ["latency_ms", "elapsed_ms", "duration_ms"]:
        if raw.get(f) is not None:
            meta["latency_ms"] = raw[f]
            break

    # Debug trace (truncated for storage)
    trace = raw.get("debug_trace") or raw.get("tool_trace")
    if trace:
        if isinstance(trace, str):
            meta["debug_trace"] = trace[:500]
        elif isinstance(trace, list):
            meta["tool_trace"] = trace[:10]  # Keep up to 10 tool calls

    # Retrieved scores
    scores = raw.get("retrieved_scores") or raw.get("reranked_scores")
    if scores and isinstance(scores, list):
        meta["retrieved_scores"] = [round(float(s), 4) for s in scores[:10]]

    # Session / user
    for f in ["session_id", "user_id"]:
        if raw.get(f):
            meta[f] = raw[f]

    return meta


def _generate_id(question: str, index: int) -> str:
    """Generate a stable ID from question text + index."""
    h = hashlib.md5(question.encode("utf-8")).hexdigest()[:6]
    return f"pred_{index}_{h}"


# ====================================================================
# Core Adapter
# ====================================================================

def adapt_one(
    raw: Dict[str, Any],
    *,
    sample_id: Optional[str] = None,
    index: int = 0,
) -> Dict[str, Any]:
    """Convert one raw agent output record to the standard prediction format.

    Args:
        raw: Raw agent output dict (any combination of fields).
        sample_id: If provided, use as the prediction ID.
                   If None, tries raw["id"] then generates one.
        index: Sequence number (used for ID generation fallback).

    Returns:
        Prediction dict matching the evaluator's expected format.
    """
    # ID resolution: explicit > raw > generated
    pid = sample_id or raw.get("id")
    if not pid:
        question = raw.get("question", "")
        pid = _generate_id(question, index)

    return {
        "id": pid,
        "answer": _extract_answer(raw),
        "used_evidence_ids": _extract_used_evidence_ids(raw),
        "final_context_ids": _extract_context_ids(raw),
        "meta": _extract_meta(raw),
    }


def adapt_batch(
    records: List[Dict[str, Any]],
    *,
    id_source: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Convert a batch of raw agent outputs.

    Args:
        records: List of raw agent output dicts.
        id_source: Optional {index: sample_id} mapping for ID override.

    Returns:
        List of prediction dicts.
    """
    results = []
    for i, raw in enumerate(records):
        sid = id_source.get(i) if id_source else None
        results.append(adapt_one(raw, sample_id=sid, index=i))
    return results


# ====================================================================
# Live Mode: Hit agent API for each benchmark sample
# ====================================================================

def run_live(
    benchmark_path: str,
    api_base: str,
    output_path: str,
    *,
    top_k: int = 5,
) -> None:
    """Query the live agent for each benchmark sample, adapt, and save."""
    # Load benchmark
    raw_text = Path(benchmark_path).read_text(encoding="utf-8")
    if benchmark_path.endswith(".jsonl"):
        samples = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    else:
        data = json.loads(raw_text)
        samples = data.get("samples", data) if isinstance(data, dict) else data

    print(f"Loaded {len(samples)} benchmark samples")
    predictions = []

    for i, sample in enumerate(samples):
        sid = sample["id"]
        question = sample["question"]
        print(f"  [{i+1:02d}/{len(samples)}] {sid}: {question[:50]}...", end=" ", flush=True)

        start = time.time()
        try:
            resp = requests.post(
                f"{api_base}/eval/query",
                json={"question": question, "top_k": top_k},
                timeout=120,
            )
            resp.raise_for_status()
            agent_out = resp.json()
            latency = int((time.time() - start) * 1000)

            # Merge question + response for adapter
            agent_out["question"] = question
            agent_out["latency_ms"] = latency

            pred = adapt_one(agent_out, sample_id=sid, index=i)
            predictions.append(pred)
            print(f"OK ({latency}ms)")

        except Exception as e:
            print(f"ERROR: {e}")
            # Write a stub prediction so the sample isn't silently dropped
            predictions.append({
                "id": sid,
                "answer": "",
                "used_evidence_ids": [],
                "final_context_ids": [],
                "meta": {"error": str(e)},
            })

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(predictions)} predictions to: {output_path}")


# ====================================================================
# Convert Mode: Raw agent logs → prediction format
# ====================================================================

def run_convert(
    input_path: str,
    output_path: str,
) -> None:
    """Read raw agent output JSONL and convert to prediction format."""
    raw_text = Path(input_path).read_text(encoding="utf-8")
    records = []
    for lineno, line in enumerate(raw_text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  WARNING: line {lineno}: {e}")

    print(f"Loaded {len(records)} raw records from {input_path}")

    predictions = adapt_batch(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to: {output_path}")


# ====================================================================
# CLI
# ====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapter: convert agent output to evaluator prediction format."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Live mode
    live_p = sub.add_parser("live", help="Query live agent API for each benchmark sample.")
    live_p.add_argument("--benchmark", required=True, help="Benchmark JSONL/JSON path.")
    live_p.add_argument("--api", default="http://127.0.0.1:8000", help="Agent API base URL.")
    live_p.add_argument("--top_k", type=int, default=5, help="Top-K for retrieval.")
    live_p.add_argument("--output", default="data/model_predictions.jsonl", help="Output JSONL.")

    # Convert mode
    conv_p = sub.add_parser("convert", help="Convert raw agent output JSONL to prediction format.")
    conv_p.add_argument("--input", required=True, help="Raw agent output JSONL path.")
    conv_p.add_argument("--output", default="data/model_predictions.jsonl", help="Output JSONL.")

    args = parser.parse_args()

    if args.mode == "live":
        run_live(args.benchmark, args.api, args.output, top_k=args.top_k)
    elif args.mode == "convert":
        run_convert(args.input, args.output)


if __name__ == "__main__":
    main()
