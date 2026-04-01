"""Knowledge base: maintains chunks + metadata and orchestrates RAG search.

Supports loading from persisted index (`data/index/chunks.json` + `faiss.index`)
or building from in-memory documents (for quick demos).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import logging

from app.config import settings
from app.rag.bm25_retriever import BM25Retriever
from app.rag.chunker import DocumentChunk, split_documents
from app.rag.dense_retriever import DenseRetriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.index_store import load_chunks_json
from app.rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Minimal in-memory KB for runnable skeleton."""

    def __init__(
        self,
        *,
        chunks: List[DocumentChunk],
        embedding_model_name: str,
        embedding_backend: Optional[str] = None,
        embedding_dim: int = 384,
        faiss_index_path: Optional[str] = None,
    ) -> None:
        # Maintain chunk list and metadata
        self.chunks: List[DocumentChunk] = list(chunks)

        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(
            self.chunks,
            model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            embedding_backend=embedding_backend,
            faiss_index_path=faiss_index_path,
        )
        self.hybrid = HybridRetriever(
            self.bm25,
            self.dense,
            alpha=settings.hybrid_alpha,
        )
        self.reranker = CrossEncoderReranker(settings.reranker_model_name)

    @classmethod
    def from_persisted_index(cls) -> Optional["KnowledgeBase"]:
        """Load chunks + FAISS index built by ingest."""
        settings.ensure_data_dirs()
        cpath = Path(settings.chunks_json_path)
        fpath = Path(settings.faiss_index_path)
        if not cpath.is_file() or not fpath.is_file():
            return None
        chunks, meta = load_chunks_json(str(cpath))
        if not chunks:
            return None
        model_name = str(meta.get("embedding_model") or settings.embedding_model_name)
        backend = meta.get("embedding_backend")
        dim = int(meta.get("embedding_dim") or 384)
        return cls(
            chunks=chunks,
            embedding_model_name=model_name,
            embedding_backend=backend,
            embedding_dim=dim,
            faiss_index_path=str(fpath),
        )

    @classmethod
    def from_documents(cls, documents: Iterable[str]) -> "KnowledgeBase":
        """Build from raw documents without persistence (for demo)."""
        chunks = split_documents(documents, chunk_size=500, overlap=100)
        # DenseRetriever will require FAISS; so this path is only valid if you
        # have ingested and loaded persisted index. Keep for API completeness.
        return cls(
            chunks=chunks,
            embedding_model_name=settings.embedding_model_name,
            embedding_backend=None,
            embedding_dim=384,
            faiss_index_path=None,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k_bm25: Optional[int] = None,
        top_k_dense: Optional[int] = None,
        top_k_fusion: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Return reranked chunk candidates with scores."""
        actual_bm25_k = max(top_k_bm25 or settings.hybrid_top_k_bm25, 1)
        actual_dense_k = max(top_k_dense or settings.hybrid_top_k_dense, 1)
        actual_fusion_k = max(top_k_fusion or settings.hybrid_fusion_top_k, 1)
        actual_top_n = top_n or settings.rag_top_k

        hybrid_candidates = self.hybrid.search(
            query,
            top_k_bm25=actual_bm25_k,
            top_k_dense=actual_dense_k,
            top_k=actual_fusion_k,
        )
        reranked = self.reranker.rerank(
            query,
            hybrid_candidates,
            top_n=actual_top_n,
        )

        reranker_active = self.reranker._model is not None
        logger.info(
            "\n===== RERANKER + FINAL SELECT =====\n"
            "Query: %s\n"
            "Hybrid input to reranker: %d chunks\n"
            "Reranker model: %s  (active=%s)\n"
            "Reranker top_n: %d\n"
            "After reranker: %d chunks\n%s\n"
            "===================================",
            query,
            len(hybrid_candidates),
            self.reranker.model_name, reranker_active,
            actual_top_n,
            len(reranked),
            "\n".join(
                "  [Reranked] %s  score=%.4f  text=%.60s" % (
                    c.chunk_id, s, c.text[:60]
                ) for c, s in reranked
            ) if reranked else "  (none)",
        )

        return reranked

    def search(
        self,
        query: str,
        *,
        top_k_bm25: Optional[int] = None,
        top_k_dense: Optional[int] = None,
        top_k_fusion: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Return final context texts after hybrid + rerank."""
        reranked = self.retrieve(
            query,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            top_k_fusion=top_k_fusion,
            top_n=top_n,
        )
        return [c.text for c, _ in reranked]

