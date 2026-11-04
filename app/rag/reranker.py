"""Cross-Encoder reranker (sentence-transformers).

Input candidates are typically the output of hybrid retrieval:
`List[Tuple[DocumentChunk, fused_score]]`
The cross-encoder ignores fused_score and re-scores (query, chunk.text) pairs.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import os

from app.rag.chunker import DocumentChunk


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
