# -*- coding: utf-8 -*-
"""Memory deduplicator based on embedding cosine similarity.

Changed from text-based similarity (SequenceMatcher + Jaccard) to
embedding-based cosine similarity with threshold 0.9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeduplicateResult:
    is_duplicate: bool
    best_similarity: float


class MemoryDeduplicator:
    """Deduplicate memory candidates using embedding cosine similarity.

    If embedder is not provided, falls back to a simple text overlap check.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        embedder=None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder

    def check_duplicate(
        self,
        *,
        candidate: str,
        existing_records: List[Any],
        existing_vectors: Optional[np.ndarray] = None,
    ) -> DeduplicateResult:
        """Check if candidate is a duplicate of any existing record.

        Args:
            candidate: text content of the new candidate.
            existing_records: list of MemoryRecord objects.
            existing_vectors: (N, dim) numpy array of existing record embeddings.
                If provided with self.embedder, uses cosine similarity.

        Returns:
            DeduplicateResult with is_duplicate flag and best_similarity score.
        """
        if not existing_records:
            return DeduplicateResult(is_duplicate=False, best_similarity=0.0)

        # --- Embedding-based dedup (preferred) ---
        if self.embedder is not None and existing_vectors is not None and existing_vectors.shape[0] > 0:
            return self._embedding_dedup(candidate, existing_vectors)

        # --- Text fallback ---
        return self._text_fallback(candidate, existing_records)

    def _embedding_dedup(
        self,
        candidate: str,
        existing_vectors: np.ndarray,
    ) -> DeduplicateResult:
        """Cosine similarity between candidate embedding and all existing vectors."""
        q = self.embedder.encode([candidate]).astype(np.float32)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        ev = existing_vectors.astype(np.float32)
        ev_norm = ev / (np.linalg.norm(ev, axis=1, keepdims=True) + 1e-12)

        sims = (ev_norm @ q_norm[0]).astype(np.float64)
        best = float(sims.max()) if sims.size > 0 else 0.0
        is_dup = best >= self.similarity_threshold

        if is_dup:
            logger.debug("Embedding dedup: sim=%.4f >= %.2f, DUPLICATE: %s",
                         best, self.similarity_threshold, candidate[:50])
        return DeduplicateResult(is_duplicate=is_dup, best_similarity=best)

    def _text_fallback(
        self,
        candidate: str,
        existing_records: List[Any],
    ) -> DeduplicateResult:
        """Simple text overlap fallback when embedder is not available."""
        best = 0.0
        cand_tokens = set(candidate.lower().split())
        if not cand_tokens:
            return DeduplicateResult(is_duplicate=False, best_similarity=0.0)

        for rec in existing_records:
            content = str(getattr(rec, "content", ""))
            rec_tokens = set(content.lower().split())
            if not rec_tokens:
                continue
            jaccard = len(cand_tokens & rec_tokens) / len(cand_tokens | rec_tokens)
            if jaccard > best:
                best = jaccard

        is_dup = best >= self.similarity_threshold
        return DeduplicateResult(is_duplicate=is_dup, best_similarity=best)
