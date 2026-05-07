"""Dump all chunks to JSONL for human labeling (Phase 0 of block-type classifier).

Reads .md/.txt files under data/docs/, runs the production chunker, and writes
one JSON object per chunk to data/labeling/raw_chunks.jsonl.

NOTE: this script intentionally does NOT use app.utils.helpers.load_text_and_md_files
because that helper collapses all whitespace (including newlines), which destroys
code-block / table / formula structure. Newlines are essential for the classifier
to learn block-type features, so we use a local loader that preserves them.

Usage:
    python scripts/dump_chunks_for_labeling.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.rag.chunker import split_documents

OUTPUT_PATH = Path(ROOT) / "data" / "labeling" / "raw_chunks.jsonl"


def _strip_html_keep_newlines(text: str) -> str:
    """Remove HTML/XML tags and entities, but PRESERVE newlines and indentation.

    Differs from app.utils.helpers.strip_html_tags which collapses \\s+ to a single
    space — that destroys code blocks, tables, and formula structure.
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # collapse only spaces/tabs, not newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_docs(directory: str) -> List[Tuple[str, str]]:
    root = Path(directory)
    out: List[Tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".md", ".txt"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        out.append((str(p.relative_to(root)), _strip_html_keep_newlines(text)))
    return out


def main() -> None:
    files = _load_docs(settings.docs_dir)
    if not files:
        print(f"No .md/.txt files under {settings.docs_dir}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for source, content in files:
            chunks = split_documents([content], chunk_size=256, overlap=64)
            for i, c in enumerate(chunks):
                rec = {
                    "chunk_id": f"{source}::{i}",
                    "source": source,
                    "chunk_index": i,
                    "text": c.text,
                    "char_len": len(c.text),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {len(files)} files -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
