"""Ingest docs from `data/docs` into a local FAISS index + chunks JSON.

Steps:
1) Read .txt/.md files under data/docs (recursive)
2) Chunk texts (chunk_size=500, overlap=100)
3) Embed each chunk using local embeddings (sentence-transformers or hash fallback)
4) Write FAISS index to `data/index/faiss.index`
5) Write chunk_id/text/metadata to `data/index/chunks.json`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for `python scripts/ingest_docs.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.llm.embeddings import get_embedding_model
from app.rag.chunker import DocumentChunk
from app.rag.index_store import save_chunks_json
from app.utils.helpers import load_text_and_md_files


def _safe_id(s: str) -> str:
    import re

    base = s.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return base or "doc"


def chunk_from_source(text: str, source: str) -> list[DocumentChunk]:
    """Chunk one document and attach metadata including source path."""
    from app.rag.chunker import split_documents

    # Use split_documents on a single-item list, then enrich metadata.
    chunks = split_documents([text], chunk_size=256, overlap=64)
    out: list[DocumentChunk] = []
    for i, c in enumerate(chunks):
        out.append(
            DocumentChunk(
                chunk_id=f"{_safe_id(source)}_{i}",
                text=c.text,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    **(c.metadata or {}),
                },
            )
        )
    return out


def main() -> None:
    settings.ensure_data_dirs()

    files = load_text_and_md_files(settings.docs_dir)
    if not files:
        print(f"No .txt/.md files under {settings.docs_dir}")
        return

    chunks: list[DocumentChunk] = []
    for rel_path, content in files:
        chunks.extend(chunk_from_source(content, rel_path))

    if not chunks:
        print("No chunks produced (all docs empty?)")
        return

    force_hash = os.environ.get("DEEPSEARCH_EMBED_FORCE_HASH", "").lower() in ("1", "true", "yes")
    embedder = get_embedding_model(settings.embedding_model_name, force_hash=force_hash, dim=384)
    print(f"Embedding backend: {embedder.backend} (dim={embedder.dim})")

    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts).astype(np.float32)
    # normalize for cosine via inner product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)

    import faiss

    index = faiss.IndexFlatIP(embedder.dim)
    index.add(np.ascontiguousarray(vecs, dtype=np.float32))
    faiss.write_index(index, settings.faiss_index_path)

    save_chunks_json(
        settings.chunks_json_path,
        chunks,
        embedding_model=settings.embedding_model_name,
        embedding_backend=embedder.backend,
        embedding_dim=embedder.dim,
    )

    print(f"Ingested {len(chunks)} chunks.")
    print(f"FAISS index -> {settings.faiss_index_path}")
    print(f"Chunks JSON -> {settings.chunks_json_path}")


if __name__ == "__main__":
    main()

