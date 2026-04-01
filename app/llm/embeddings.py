"""Unified local embedding interface with hash fallback."""

from __future__ import annotations

import hashlib
from typing import Literal, Optional

import numpy as np

EmbeddingBackend = Literal["sentence_transformers", "hash_fallback"]


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    data = text.encode("utf-8", errors="ignore")
    if not data:
        return vec
    h = hashlib.sha256(data).digest()
    for i, b in enumerate(data):
        vec[i % dim] += float(b) / 255.0
    for j in range(len(h)):
        vec[j % dim] += float(h[j]) / 255.0
    n = float(np.linalg.norm(vec))
    if n > 1e-8:
        vec /= n
    return vec.astype(np.float32)


class EmbeddingModel:
    """Encodes text to dense vectors; never calls remote APIs."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        *,
        force_hash: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.backend: EmbeddingBackend = "hash_fallback"
        self._model = None
        if force_hash:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self.backend = "sentence_transformers"
        except Exception:
            self._model = None
            self.backend = "hash_fallback"

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            emb = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            return np.asarray(emb, dtype=np.float32)
        out = np.stack([_hash_embedding(t, self.dim) for t in texts], axis=0)
        return out.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedding_model(
    model_name: Optional[str] = None,
    *,
    force_hash: bool = False,
    dim: int = 384,
) -> EmbeddingModel:
    name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    return EmbeddingModel(model_name=name, dim=dim, force_hash=force_hash)

