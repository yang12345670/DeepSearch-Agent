# -*- coding: utf-8 -*-
"""Long-term memory with structured storage + FAISS index.

Storage flow: extract → gate → dedup → structure → embed → FAISS persist
Recall flow: query → embed → FAISS search → filter by type → format

Structured storage fields per record:
  - memory_id, user_id, memory_type, content, metadata, timestamp

Three memory types:
  - user_preference:    用户偏好
  - research_direction: 用户研究方向
  - task_conclusion:    重要任务结论
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from app.config import settings
from app.llm.embeddings import get_embedding_model
from app.memory.deduplicator import MemoryDeduplicator

logger = logging.getLogger(__name__)

MemoryType = Literal["user_preference", "research_direction", "task_conclusion"]


@dataclass
class MemoryRecord:
    """Structured memory record."""
    memory_id: str
    user_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    confidence: float = 0.5     # 0.0~1.0  memory reliability (LLM=0.9, regex=0.5, task=0.7)
    importance: float = 0.33    # 0.0~1.0  derived from gate_score (1→0.33, 2→0.67, 3→1.0)


class LongTermMemory:
    """Long-term memory store persisted to disk and searchable by vector similarity."""

    def __init__(
        self,
        *,
        index_dir: str = "data/memory",
        model_name: Optional[str] = None,
        embedding_dim: int = 384,
        dedup_threshold: float = 0.9,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.index_dir / "records.json"
        self.faiss_path = self.index_dir / "memory.faiss"
        self.vectors_path = self.index_dir / "vectors.npy"

        self.embedder = get_embedding_model(
            model_name or settings.embedding_model_name, dim=embedding_dim
        )
        self.embedding_dim = self.embedder.dim

        # Deduplicator now uses embedding similarity >= 0.9
        self.deduplicator = MemoryDeduplicator(
            similarity_threshold=dedup_threshold,
            embedder=self.embedder,
        )

        self.records: List[MemoryRecord] = []
        self._vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._faiss = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.records_path.exists():
            try:
                raw = json.loads(self.records_path.read_text(encoding="utf-8"))
                self.records = [
                    MemoryRecord(
                        memory_id=str(item["memory_id"]),
                        user_id=str(item.get("user_id", "default")),
                        memory_type=item["memory_type"],
                        content=str(item["content"]),
                        metadata=dict(item.get("metadata", {})),
                        timestamp=float(item.get("timestamp", 0)),
                        confidence=float(item.get("confidence", 0.5)),
                        importance=float(item.get("importance", 0.33)),
                    )
                    for item in raw
                ]
            except Exception:
                self.records = []

        if self.vectors_path.exists():
            try:
                vec = np.load(self.vectors_path)
                self._vectors = np.asarray(vec, dtype=np.float32)
            except Exception:
                self._vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)

        if self._vectors.shape[0] != len(self.records):
            logger.warning("Vector/record count mismatch (%d vs %d), resetting",
                           self._vectors.shape[0], len(self.records))
            self.records = []
            self._vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        self._faiss = None
        if self._vectors.shape[0] == 0:
            return
        try:
            import faiss

            vectors = np.asarray(self._vectors, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            normed = vectors / norms
            index = faiss.IndexFlatIP(normed.shape[1])
            index.add(normed)
            self._faiss = index
        except Exception:
            self._faiss = None

    def _persist(self) -> None:
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "memory_id": r.memory_id,
                "user_id": r.user_id,
                "memory_type": r.memory_type,
                "content": r.content,
                "metadata": r.metadata,
                "timestamp": r.timestamp,
                "confidence": r.confidence,
                "importance": r.importance,
            }
            for r in self.records
        ]
        self.records_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(self.vectors_path, self._vectors)

        if self._faiss is not None:
            try:
                import faiss
                faiss.write_index(self._faiss, str(self.faiss_path))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Storage: dedup → structure → embed → persist
    # ------------------------------------------------------------------

    def add_memories(self, candidates: List[Dict[str, Any]]) -> List[MemoryRecord]:
        """Add gated candidates after dedup + structuring + embedding.

        Candidates should already have passed the memory gate.
        Each candidate dict must have: memory_type, content.
        Optional: metadata (with user_id, session_id, gate_score, etc.)
        """
        valid_types = ("user_preference", "research_direction", "task_conclusion")
        created: List[MemoryRecord] = []
        clean: List[Dict[str, Any]] = []

        for c in candidates:
            memory_type = c.get("memory_type")
            content = str(c.get("content", "")).strip()
            if memory_type not in valid_types or not content:
                continue

            # Embedding-based dedup (cosine similarity >= 0.9)
            dedup = self.deduplicator.check_duplicate(
                candidate=content,
                existing_records=self.records,
                existing_vectors=self._vectors,
            )
            if dedup.is_duplicate:
                logger.info("Dedup rejected (sim=%.3f): %s", dedup.best_similarity, content[:50])
                continue
            clean.append(c)

        if not clean:
            return created

        # Embed
        contents = [str(c["content"]).strip() for c in clean]
        vectors = self.embedder.encode(contents)
        vectors = np.asarray(vectors, dtype=np.float32)

        # Structure + store (with confidence & importance)
        _CONFIDENCE_MAP = {
            "llm_extraction": 0.9,
            "regex_fallback": 0.5,
            "task_result": 0.7,
        }

        for idx, c in enumerate(clean):
            meta = dict(c.get("metadata", {}))

            # confidence: derived from extraction source
            source = str(meta.get("source", "regex_fallback"))
            confidence = _CONFIDENCE_MAP.get(source, 0.5)

            # importance: derived from gate_score (0~3 → 0.0~1.0)
            gate_score = int(meta.get("gate_score", 1))
            importance = round(gate_score / 3.0, 2)

            rec = MemoryRecord(
                memory_id=str(uuid.uuid4()),
                user_id=str(meta.get("user_id", "default")),
                memory_type=c["memory_type"],
                content=contents[idx],
                metadata=meta,
                timestamp=time.time(),
                confidence=confidence,
                importance=importance,
            )
            self.records.append(rec)
            created.append(rec)

        if self._vectors.shape[0] == 0:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors]).astype(np.float32)

        self._rebuild_faiss()
        self._persist()

        logger.info("Added %d long-term memories (total: %d)", len(created), len(self.records))
        for r in created:
            logger.info("  [%s] user=%s content=%s", r.memory_type, r.user_id, r.content[:60])

        return created

    # ------------------------------------------------------------------
    # Recall: query → embed → FAISS search → filter by type
    # ------------------------------------------------------------------

    def recall_memories(
        self,
        query: str,
        top_k: int = 5,
        *,
        filter_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """Recall relevant memories via FAISS vector search + optional filtering.

        Args:
            query: user query to match against.
            top_k: max number of memories to return.
            filter_types: if set, only return these memory types
                          e.g. ["user_preference", "research_direction"]
            user_id: if set, only return memories for this user.

        Returns:
            List of MemoryRecord sorted by relevance.
        """
        if not query.strip() or not self.records:
            return []

        q = self.embedder.encode([query]).astype(np.float32)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        # Retrieve more candidates than needed, then filter
        search_k = min(top_k * 3, len(self.records))

        if self._faiss is not None:
            try:
                scores, indices = self._faiss.search(qn, search_k)
                raw_results = [
                    (self.records[int(idx)], float(scores[0][i]))
                    for i, idx in enumerate(indices[0])
                    if idx >= 0
                ]
            except Exception:
                raw_results = self._fallback_search(qn, search_k)
        else:
            raw_results = self._fallback_search(qn, search_k)

        # Apply filters
        filtered = []
        for rec, score in raw_results:
            if filter_types and rec.memory_type not in filter_types:
                continue
            if user_id and rec.user_id != user_id:
                continue
            filtered.append((rec, score))

        # Composite ranking: similarity × (w1*confidence + w2*importance + w3*recency)
        now = time.time()
        W_CONF, W_IMP, W_REC = 0.3, 0.3, 0.4
        ranked = []
        for rec, sim_score in filtered:
            days_old = (now - rec.timestamp) / 86400.0 if rec.timestamp > 0 else 30.0
            recency = 1.0 / (1.0 + days_old)
            quality = W_CONF * rec.confidence + W_IMP * rec.importance + W_REC * recency
            rank_score = sim_score * (0.5 + 0.5 * quality)  # quality modulates similarity
            ranked.append((rec, rank_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        result = [rec for rec, _ in ranked[:top_k]]

        logger.info(
            "Recall: query=%s, raw=%d, after filter=%d, returned=%d (types=%s, user=%s)",
            query[:30], len(raw_results), len(filtered), len(result),
            filter_types, user_id,
        )
        if ranked[:top_k]:
            for rec, rs in ranked[:top_k]:
                logger.debug(
                    "  [%s] rank=%.4f conf=%.2f imp=%.2f ts=%s content=%s",
                    rec.memory_type, rs, rec.confidence, rec.importance,
                    time.strftime("%Y-%m-%d", time.localtime(rec.timestamp)),
                    rec.content[:40],
                )
        return result

    def _fallback_search(
        self, qn: np.ndarray, k: int
    ) -> List[tuple]:
        """In-memory cosine search when FAISS is unavailable."""
        vectors = np.asarray(self._vectors, dtype=np.float32)
        vn = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        sims = (vn @ qn[0]).astype(np.float32)
        order = np.argsort(-sims)[:k]
        return [(self.records[int(i)], float(sims[int(i)])) for i in order]

    # ------------------------------------------------------------------
    # Format for LLM context
    # ------------------------------------------------------------------

    @staticmethod
    def format_memories_for_context(records: List[MemoryRecord]) -> str:
        """Format recalled memories as prompt-ready text, grouped by type."""
        if not records:
            return ""

        groups: Dict[str, List[str]] = {}
        type_labels = {
            "user_preference": "用户偏好",
            "research_direction": "研究方向",
            "task_conclusion": "任务结论",
        }
        for rec in records:
            label = type_labels.get(rec.memory_type, rec.memory_type)
            groups.setdefault(label, []).append(rec.content)

        lines = ["长期记忆参考："]
        for label, items in groups.items():
            lines.append(f"[{label}]")
            for item in items:
                lines.append(f"  - {item}")
        return "\n".join(lines)
