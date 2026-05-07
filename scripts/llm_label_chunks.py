"""LLM-based block-type labeling (replaces manual labeling).

Calls gpt-4o-mini (or whatever LLM_MODEL_NAME is set) on each chunk in
data/labeling/to_label.jsonl and writes results to data/labeling/labeled.jsonl
in the same format as the manual CLI. Resumable.

Concurrency 8, ~3-5 min for 1.4k chunks, ~$0.10 cost on gpt-4o-mini.

Usage:
    python scripts/llm_label_chunks.py
    python scripts/llm_label_chunks.py --concurrency 16 --model gpt-4o-mini
    python scripts/llm_label_chunks.py --limit 50  # for a smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import settings

TO_LABEL = ROOT / "data" / "labeling" / "to_label.jsonl"
LABELED = ROOT / "data" / "labeling" / "labeled.jsonl"

VALID = {"text", "code", "formula", "table"}

SYSTEM_PROMPT = """You classify a single text chunk from a markdown technical book into ONE of four block types.

Labels:
- "code": multi-line source code (Python, JS, Go, GDScript, shell, etc.). Heuristics: contains def/func/class/return/import/if/for, indented blocks, code-style identifiers. A passage of prose with a single inline `code` reference is NOT code.
- "formula": dominated by mathematical notation. Heuristics: $$...$$ blocks, $...$ inline math, LaTeX commands like \\frac \\sum \\int, or probability/equation-style expressions like P(x|y) = ...
- "table": markdown table with pipe separators AND a separator row containing dashes (| --- | --- |).
- "text": natural-language prose (Chinese or English), bullet/numbered lists, headings, descriptions. This is the default for anything that is not clearly code/formula/table.

Decision rule: pick the label that describes the DOMINANT content of the chunk. If a chunk is mostly prose with a small code or formula snippet inside, label it "text".

Output STRICTLY a JSON object: {"label": "<one of text|code|formula|table>"}. No markdown, no explanation, no extra keys."""

USER_TEMPLATE = """Chunk to classify:

```
{text}
```

Respond with JSON only."""


_io_lock = threading.Lock()


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


def append_label(rec: dict, label: str, *, fh) -> None:
    out = {
        "chunk_id": rec["chunk_id"],
        "source": rec["source"],
        "label": label,
        "heuristic": rec.get("heuristic"),
        "text": rec["text"],
    }
    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    fh.flush()


def classify_one(client, model: str, rec: dict, max_retries: int = 2) -> Optional[str]:
    user_msg = USER_TEMPLATE.format(text=rec["text"])
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content)
            label = parsed.get("label", "").strip().lower()
            if label in VALID:
                return label
            return None
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    if last_err:
        print(f"[err] {rec['chunk_id']}: {type(last_err).__name__}: {last_err}", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.llm_model_name)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only label N pending chunks (smoke test)")
    args = ap.parse_args()

    if settings.llm_provider == "local" or not settings.llm_api_key:
        print("ERROR: LLM_PROVIDER is local or LLM_API_KEY is empty. "
              "Configure provider/api key in .env first.")
        sys.exit(1)

    if not TO_LABEL.exists():
        print(f"Missing {TO_LABEL}. Run sample_for_labeling.py first.")
        sys.exit(1)

    with TO_LABEL.open("r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    done = load_done()
    pending = [r for r in all_records if r["chunk_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]

    if not pending:
        print(f"All {len(all_records)} chunks already labeled. Done.")
        return

    print(f"Total: {len(all_records)}  Done: {len(done)}  Pending: {len(pending)}")
    print(f"Model: {args.model}, concurrency: {args.concurrency}")

    from openai import OpenAI
    base_url = settings.llm_base_url or "https://api.openai.com/v1"
    client = OpenAI(api_key=settings.llm_api_key, base_url=base_url)

    LABELED.parent.mkdir(parents=True, exist_ok=True)
    fh = LABELED.open("a", encoding="utf-8")

    n_ok = 0
    n_fail = 0
    start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            future_to_rec = {
                pool.submit(classify_one, client, args.model, rec): rec
                for rec in pending
            }
            for i, fut in enumerate(as_completed(future_to_rec), 1):
                rec = future_to_rec[fut]
                label = fut.result()
                if label is None:
                    n_fail += 1
                    continue
                with _io_lock:
                    append_label(rec, label, fh=fh)
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                n_ok += 1

                if i % 50 == 0 or i == len(pending):
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(pending)}]  ok={n_ok} fail={n_fail}  "
                          f"rate={rate:.1f}/s  eta={eta:.0f}s",
                          flush=True)
    finally:
        fh.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. ok={n_ok}, fail={n_fail}")

    # Distribution summary
    done = load_done()
    counts: Dict[str, int] = {}
    for lbl in done.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    print("\nLabel distribution:")
    for lbl in ("text", "code", "formula", "table"):
        print(f"  {lbl:8s}: {counts.get(lbl, 0)}")
    print(f"  total   : {sum(counts.values())}")


if __name__ == "__main__":
    main()
