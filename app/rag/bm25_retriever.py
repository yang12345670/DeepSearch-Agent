"""BM25 retriever (rank-bm25) with Chinese tokenization support.

Uses jieba for Chinese text segmentation, falls back to whitespace split
for pure ASCII text. This dramatically improves BM25 recall for Chinese queries.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.chunker import DocumentChunk

# Lazy-load jieba to avoid import overhead when not needed
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(jieba.logging.WARNING)
        _jieba = jieba
    return _jieba


def _has_chinese(text: str) -> bool:
    """Check if text contains any Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def tokenize(text: str) -> List[str]:
    """Tokenize text: jieba for Chinese, whitespace split for English."""
    text = text.lower().strip()
    if not text:
        return []
    if _has_chinese(text):
        jieba = _get_jieba()
        return [w for w in jieba.lcut(text) if w.strip()]
    return text.split()


class BM25Retriever:
    """Sparse lexical retriever with Chinese support."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks: List[DocumentChunk] = list(chunks)
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 8) -> List[Tuple[DocumentChunk, float]]:
        """Return (chunk, bm25_score) sorted by score desc."""
        if not self.chunks or self._bm25 is None:
            return []

        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in top_indices]
