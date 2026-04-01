"""Cross-Encoder reranker (sentence-transformers).

This module provides:
- a clear functional API: `rerank(query, candidates, top_n, model_name=...)`
- a class wrapper: `CrossEncoderReranker`

Input candidates are typically the output of hybrid retrieval:
`List[Tuple[DocumentChunk, fused_score]]`
The cross-encoder ignores fused_score and re-scores (query, chunk.text) pairs.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import os

from app.rag.chunker import DocumentChunk


def rerank(
    query: str,
    candidates: Sequence[Tuple[DocumentChunk, float]],
    top_n: int,
    *,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Tuple[DocumentChunk, float]]:
    """Functional rerank API.

    Args:
        query: user query
        candidates: hybrid candidates `(chunk, fused_score)`
        top_n: keep at most this many after reranking
        model_name: sentence-transformers cross-encoder model id

    Returns:
        List of `(chunk, rerank_score)` sorted by rerank_score desc.
        If `DEEPSEARCH_SIMPLE_RERANKER=1`, returns the original order truncated.
    """
    if not candidates or top_n <= 0:
        return []
    if os.environ.get("DEEPSEARCH_SIMPLE_RERANKER", "").lower() in ("1", "true", "yes"):
        return list(candidates)[:top_n]

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [(query, c.text) for c, _ in candidates]
    scores = model.predict(pairs)
    reranked = [(c, float(s)) for (c, _), s in zip(candidates, scores)]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_n]


def contexts_from_reranked(reranked: Sequence[Tuple[DocumentChunk, float]]) -> List[str]:
    """Return final context texts in rerank order."""
    return [c.text for c, _ in reranked]


class CrossEncoderReranker:
    """Cross-encoder reranker."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            if os.environ.get("DEEPSEARCH_SIMPLE_RERANKER", "").lower() in ("1", "true", "yes"):
                self._model = None
            else:
                self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[DocumentChunk, float]],
        top_n: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Rerank candidates, return (chunk, rerank_score) sorted by score desc."""
        if not candidates or top_n <= 0:
            return []

        if self._model is None:
            # Fallback: keep original fused order and just truncate.
            return list(candidates)[:top_n]

        pairs = [(query, c.text) for c, _ in candidates]
        scores = self._model.predict(pairs)
        reranked = [(c, float(s)) for (c, _), s in zip(candidates, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_n]

