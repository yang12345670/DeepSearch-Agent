"""Index persistence helpers (chunks JSON).

We store chunk texts + metadata in JSON, and store vectors in a FAISS index.
This module is intentionally lightweight so ingestion scripts can import it
without pulling in reranker/agent modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.rag.chunker import DocumentChunk


def save_chunks_json(
    path: str,
    chunks: List[DocumentChunk],
    *,
    embedding_model: str,
    embedding_backend: str,
    embedding_dim: int,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": 1,
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_dim": embedding_dim,
        "chunks": [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": dict(c.metadata)} for c in chunks
        ],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks_json(path: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    chunks: List[DocumentChunk] = []
    for item in data.get("chunks", []):
        chunks.append(
            DocumentChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item.get("metadata") or {},
            )
        )
    meta = {
        "embedding_model": data.get("embedding_model"),
        "embedding_backend": data.get("embedding_backend"),
        "embedding_dim": data.get("embedding_dim", 384),
    }
    return chunks, meta

