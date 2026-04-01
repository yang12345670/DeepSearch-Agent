# -*- coding: utf-8 -*-
"""Auto-detect new/changed documents and rebuild FAISS index on startup.

Compares a fingerprint (filename + size + mtime) of all docs in data/docs/
against the last recorded fingerprint. If any change is detected, triggers
a full re-ingestion: chunk → embed → FAISS index → persist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.config import settings
from app.llm.embeddings import get_embedding_model
from app.rag.chunker import DocumentChunk
from app.rag.index_store import save_chunks_json
from app.utils.helpers import load_text_and_md_files

logger = logging.getLogger(__name__)

FINGERPRINT_PATH = Path(settings.index_dir) / "docs_fingerprint.json"


def _compute_docs_fingerprint(docs_dir: str) -> Dict[str, Dict]:
    """Build a fingerprint dict: {rel_path: {size, mtime_ns}} for all docs."""
    root = Path(docs_dir)
    if not root.exists():
        return {}
    fp: Dict[str, Dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".txt", ".md"):
            continue
        stat = p.stat()
        fp[str(p.relative_to(root))] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return fp


def _load_saved_fingerprint() -> Dict[str, Dict]:
    """Load the fingerprint from last successful indexing."""
    if not FINGERPRINT_PATH.is_file():
        return {}
    try:
        return json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_fingerprint(fp: Dict[str, Dict]) -> None:
    """Persist current fingerprint after successful indexing."""
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(
        json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _safe_id(s: str) -> str:
    import re
    base = s.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)
    return base or "doc"


def _chunk_from_source(text: str, source: str) -> List[DocumentChunk]:
    from app.rag.chunker import split_documents

    chunks = split_documents([text], chunk_size=256, overlap=64)
    out: List[DocumentChunk] = []
    for i, c in enumerate(chunks):
        out.append(
            DocumentChunk(
                chunk_id=f"{_safe_id(source)}_{i}",
                text=c.text,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    **(c.metadata or {}),
                },
            )
        )
    return out


def docs_changed() -> bool:
    """Check whether docs directory has changed since last indexing."""
    current = _compute_docs_fingerprint(settings.docs_dir)
    saved = _load_saved_fingerprint()
    return current != saved


def rebuild_index() -> int:
    """Re-ingest all docs and rebuild FAISS index.

    Returns:
        Number of chunks indexed, or 0 if no docs found.
    """
    settings.ensure_data_dirs()

    files = load_text_and_md_files(settings.docs_dir)
    if not files:
        logger.warning("auto_index: No .txt/.md files under %s", settings.docs_dir)
        return 0

    chunks: List[DocumentChunk] = []
    for rel_path, content in files:
        chunks.extend(_chunk_from_source(content, rel_path))

    if not chunks:
        logger.warning("auto_index: No chunks produced (all docs empty?)")
        return 0

    force_hash = os.environ.get("DEEPSEARCH_EMBED_FORCE_HASH", "").lower() in (
        "1", "true", "yes",
    )
    embedder = get_embedding_model(
        settings.embedding_model_name, force_hash=force_hash, dim=384
    )
    logger.info("auto_index: Embedding backend: %s (dim=%d)", embedder.backend, embedder.dim)

    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)

    import faiss

    index = faiss.IndexFlatIP(embedder.dim)
    index.add(np.ascontiguousarray(vecs, dtype=np.float32))
    faiss.write_index(index, settings.faiss_index_path)

    save_chunks_json(
        settings.chunks_json_path,
        chunks,
        embedding_model=settings.embedding_model_name,
        embedding_backend=embedder.backend,
        embedding_dim=embedder.dim,
    )

    # Save fingerprint so next startup won't re-index unnecessarily
    fp = _compute_docs_fingerprint(settings.docs_dir)
    _save_fingerprint(fp)

    logger.info(
        "auto_index: Indexed %d chunks from %d files. FAISS -> %s",
        len(chunks), len(files), settings.faiss_index_path,
    )
    return len(chunks)


def auto_index_if_needed() -> None:
    """Check for doc changes and rebuild index if needed. Safe to call on startup."""
    index_exists = (
        Path(settings.faiss_index_path).is_file()
        and Path(settings.chunks_json_path).is_file()
    )

    if not index_exists:
        logger.info("auto_index: No existing index found, building from scratch...")
        count = rebuild_index()
        if count:
            logger.info("auto_index: Initial indexing complete (%d chunks).", count)
        return

    if docs_changed():
        logger.info("auto_index: Document changes detected, rebuilding index...")
        count = rebuild_index()
        logger.info("auto_index: Re-indexing complete (%d chunks).", count)
    else:
        logger.info("auto_index: Documents unchanged, skipping re-index.")
