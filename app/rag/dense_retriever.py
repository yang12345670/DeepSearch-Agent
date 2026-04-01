"""Dense retriever.

- If a FAISS index exists (built by `scripts/ingest_docs.py`), use it for recall.
- Otherwise, fall back to in-memory cosine similarity so the project is still runnable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import os

import numpy as np

from app.config import settings
from app.llm.embeddings import get_embedding_model
from app.rag.chunker import DocumentChunk


class DenseRetriever:
    """Semantic retriever using dense embeddings + FAISS (optional)."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        model_name: str,
        *,
        embedding_dim: int = 384,
        embedding_backend: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
    ) -> None:
        self.chunks: List[DocumentChunk] = list(chunks)
        force_hash = embedding_backend == "hash_fallback" or os.environ.get("DEEPSEARCH_EMBED_FORCE_HASH", "").lower() in (
            "1",
            "true",
            "yes",
        )
        embedder = get_embedding_model(model_name, force_hash=force_hash, dim=embedding_dim)
        self._embedder = embedder
        self._index = None
        self._matrix = None

        # If no chunks, keep trivial state.
        if not self.chunks:
            self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
            return

        # Try to use FAISS if index file exists.
        index_path = faiss_index_path or settings.faiss_index_path
        try:
            from pathlib import Path

            p = Path(index_path)
            if p.is_file():
                import faiss

                index = faiss.read_index(str(p))
                if index.d != embedder.dim:
                    raise ValueError(f"FAISS dim {index.d} != embedder dim {embedder.dim}")
                if index.ntotal != len(self.chunks):
                    raise ValueError(
                        f"FAISS vectors {index.ntotal} != chunks {len(self.chunks)}. Rebuild index."
                    )
                self._index = index
        except Exception:
            self._index = None

        # Fallback: embed all chunks in memory.
        if self._index is None:
            texts = [c.text for c in self.chunks]
            vecs = embedder.encode(texts).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)
            self._matrix = vecs

    def search(self, query: str, top_k: int = 8) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, dense_score) sorted by score desc."""
        if not self.chunks:
            return []
        if self._index is not None:
            q = self._embedder.encode_query(query).astype(np.float32).reshape(1, -1)
            import faiss

            faiss.normalize_L2(q)
            scores, indices = self._index.search(q, min(top_k, len(self.chunks)))
            out: List[Tuple[DocumentChunk, float]] = []
            for j in range(indices.shape[1]):
                idx = int(indices[0, j])
                if idx < 0:
                    continue
                out.append((self.chunks[idx], float(scores[0, j])))
            return out

        assert self._matrix is not None
        q = self._embedder.encode_query(query).astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q), 1e-8)
        scores = (self._matrix @ q).astype(np.float64)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in order]

