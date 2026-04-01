"""Hybrid fusion retriever.

Implements:
- BM25 Top-K retrieval
- Dense Top-K retrieval
- Dedupe by chunk_id
- Min-max normalization per branch
- Weighted fusion with alpha
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from app.rag.bm25_retriever import BM25Retriever
from app.rag.chunker import DocumentChunk
from app.rag.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


@dataclass
class HybridCandidate:
    chunk: DocumentChunk
    fused_score: float

    bm25_norm: float
    dense_norm: float


def _min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    v_min = float(arr.min())
    v_max = float(arr.max())
    if np.isclose(v_min, v_max):
        return [1.0 for _ in values]
    return ((arr - v_min) / (v_max - v_min)).tolist()


class HybridRetriever:
    """Fuse sparse + dense retrieval results."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        alpha: float = 0.5,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha

    def search(
        self,
        query: str,
        *,
        top_k_bm25: int = 8,
        top_k_dense: int = 8,
        top_k: int = 8,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, fused_score) for downstream reranker."""
        bm25_hits = self.bm25.search(query, top_k=top_k_bm25)
        dense_hits = self.dense.search(query, top_k=top_k_dense)

        # --- Dedup by chunk_id (union) ---
        bm25_raw = {c.chunk_id: float(s) for c, s in bm25_hits}
        dense_raw = {c.chunk_id: float(s) for c, s in dense_hits}
        chunk_by_id: Dict[str, DocumentChunk] = {}
        for c, _ in bm25_hits:
            chunk_by_id[c.chunk_id] = c
        for c, _ in dense_hits:
            chunk_by_id[c.chunk_id] = c

        bm25_norm_map: Dict[str, float] = {}
        if bm25_hits:
            norms = _min_max_normalize([s for _, s in bm25_hits])
            for (c, _), n in zip(bm25_hits, norms):
                bm25_norm_map[c.chunk_id] = n

        dense_norm_map: Dict[str, float] = {}
        if dense_hits:
            norms = _min_max_normalize([s for _, s in dense_hits])
            for (c, _), n in zip(dense_hits, norms):
                dense_norm_map[c.chunk_id] = n

        union_ids = list(dict.fromkeys(list(bm25_raw.keys()) + list(dense_raw.keys())))
        fused: List[Tuple[DocumentChunk, float]] = []
        a = float(self.alpha)
        for cid in union_ids:
            bn = bm25_norm_map.get(cid, 0.0)
            dn = dense_norm_map.get(cid, 0.0)
            score = a * bn + (1.0 - a) * dn
            fused.append((chunk_by_id[cid], score))

        fused.sort(key=lambda x: x[1], reverse=True)
        result = fused[:top_k]

        # --- Debug log ---
        logger.info(
            "\n===== HYBRID RETRIEVAL DEBUG =====\n"
            "Query: %s\n"
            "alpha: %.2f  (formula: %.2f * bm25_norm + %.2f * dense_norm)\n"
            "BM25 hits: %d  (requested top_k_bm25=%d)\n%s\n"
            "Dense hits: %d  (requested top_k_dense=%d)\n%s\n"
            "After dedup (union by chunk_id): %d unique chunks\n"
            "After fusion top_k=%d: %d chunks\n%s\n"
            "==================================",
            query,
            a, a, 1.0 - a,
            len(bm25_hits), top_k_bm25,
            "\n".join(
                "  [BM25] %s  raw=%.4f  norm=%.4f  text=%.50s" % (
                    c.chunk_id, s, bm25_norm_map.get(c.chunk_id, 0.0), c.text[:50]
                ) for c, s in bm25_hits
            ) if bm25_hits else "  (none)",
            len(dense_hits), top_k_dense,
            "\n".join(
                "  [Dense] %s  raw=%.4f  norm=%.4f  text=%.50s" % (
                    c.chunk_id, s, dense_norm_map.get(c.chunk_id, 0.0), c.text[:50]
                ) for c, s in dense_hits
            ) if dense_hits else "  (none)",
            len(union_ids),
            top_k, len(result),
            "\n".join(
                "  [Fused] %s  score=%.4f  bm25_n=%.4f  dense_n=%.4f  text=%.50s" % (
                    c.chunk_id, s,
                    bm25_norm_map.get(c.chunk_id, 0.0),
                    dense_norm_map.get(c.chunk_id, 0.0),
                    c.text[:50],
                ) for c, s in result
            ) if result else "  (none)",
        )

        return result

