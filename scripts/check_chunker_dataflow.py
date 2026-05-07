"""Phase 4 data-flow sanity check for the new block-classifier-aware chunker.

Re-runs split_documents over data/docs/, then dumps:
  - total chunk count
  - block_type distribution
  - chunk length distribution (min / p50 / p95 / max)
  - count + samples of chunks longer than chunk_size (oversized blocks now
    pass through whole instead of getting char-split — this is the new behavior)
  - longest-chunk implications for downstream MiniLM embedder (max_seq_length=256
    → tokens get truncated; flagged when chunk len exceeds that comfortably)

Run with two configs to A/B compare:
  CHUNKER_CLASSIFIER=rule       python scripts/check_chunker_dataflow.py
  CHUNKER_CLASSIFIER=distilbert python scripts/check_chunker_dataflow.py
"""

from __future__ import annotations

import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import settings
from app.rag.chunker import split_documents

DOCS_DIR = ROOT / "data" / "docs"
CHUNK_SIZE = 256


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[k]


def main() -> None:
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(text)

    print(f"Backend           : {settings.chunker_classifier}")
    print(f"Documents         : {len(docs)}")
    print(f"chunk_size        : {CHUNK_SIZE}")

    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, overlap=64, min_chunk_size=20)

    print(f"\nTotal chunks      : {len(chunks)}")

    btypes = Counter(c.metadata.get("block_type", "?") for c in chunks)
    print("\nBlock type distribution:")
    for bt in ("text", "code", "formula", "table", "?"):
        n = btypes.get(bt, 0)
        if n == 0 and bt == "?":
            continue
        pct = n / len(chunks) * 100 if chunks else 0
        print(f"  {bt:8s} {n:5d}  ({pct:5.1f}%)")

    lens = [len(c.text) for c in chunks]
    print("\nChunk length stats:")
    print(f"  min   = {min(lens) if lens else 0}")
    print(f"  p50   = {percentile(lens, 50)}")
    print(f"  p95   = {percentile(lens, 95)}")
    print(f"  max   = {max(lens) if lens else 0}")
    print(f"  mean  = {statistics.mean(lens):.1f}" if lens else "  mean  = 0")

    oversized = [c for c in chunks if len(c.text) > CHUNK_SIZE]
    print(f"\nChunks exceeding chunk_size ({CHUNK_SIZE}): {len(oversized)}  "
          f"({len(oversized) / max(len(chunks), 1) * 100:.1f}%)")
    if oversized:
        print("  By block_type:")
        for bt, n in Counter(c.metadata.get("block_type") for c in oversized).items():
            print(f"    {bt:8s} {n}")
        print("  Top 5 longest:")
        for c in sorted(oversized, key=lambda x: -len(x.text))[:5]:
            preview = c.text.replace("\n", " ")[:70]
            print(f"    len={len(c.text):5d}  bt={c.metadata.get('block_type'):8s}  "
                  f"id={c.chunk_id} | {preview}...")

    very_long = [c for c in chunks if len(c.text) > 1024]
    if very_long:
        print(f"\n⚠  {len(very_long)} chunks > 1024 chars — MiniLM embedder will silently "
              f"truncate (max_seq_length=256 tokens ≈ ~700-1000 chars in zh+en).")


if __name__ == "__main__":
    main()
