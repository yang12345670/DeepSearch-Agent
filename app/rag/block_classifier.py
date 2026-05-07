"""Block-type classifier entry point for RAG chunks (text/code/formula/table).

Public API: a single function ``classify_block(text)``. The chunker calls this
to tag every emitted chunk with ``metadata.block_type`` and to decide whether
oversized code/formula/table blocks should bypass character-level splitting.

Backend selection is driven by ``settings.chunker_classifier``:
  - "rule"       — keyword heuristic (chunker.classify_block_rule). Always available.
  - "distilbert" — fine-tuned model in ``models/block_classifier/``. Raises if unloadable.
  - "auto"       — distilbert when the model dir loads cleanly, otherwise rule.

The DistilBERT path is:
  - lazy-loaded on first call (~600MB cost not paid by callers that only need rule)
  - process-global singleton, guarded by a lock against double-init
  - LRU-cached on input text (re-ingest of the same corpus skips re-inference)
  - wrapped in try/except — any inference error degrades to rule, never crashes ingest
"""

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path
from typing import Literal, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

BlockType = Literal["text", "code", "formula", "table"]
LABELS = ("text", "code", "formula", "table")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "block_classifier"

_LOAD_LOCK = threading.Lock()
_MODEL_BUNDLE: Optional[Tuple] = None  # (tokenizer, model, device, id2label)
_LOAD_FAILED: bool = False


def _load_model_bundle() -> Optional[Tuple]:
    """Load tokenizer + model once. Returns None if anything fails."""
    global _MODEL_BUNDLE, _LOAD_FAILED

    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE
    if _LOAD_FAILED:
        return None

    with _LOAD_LOCK:
        if _MODEL_BUNDLE is not None:
            return _MODEL_BUNDLE
        if _LOAD_FAILED:
            return None

        if not MODEL_DIR.exists():
            logger.warning(
                "DistilBERT block classifier requested but model dir missing: %s. "
                "Falling back to rule classifier.", MODEL_DIR,
            )
            _LOAD_FAILED = True
            return None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
            model.eval()
            model.to(device)
            id2label = model.config.id2label
            _MODEL_BUNDLE = (tokenizer, model, device, id2label)
            logger.info("Loaded DistilBERT block classifier on %s from %s", device, MODEL_DIR)
            return _MODEL_BUNDLE
        except Exception as e:
            logger.warning(
                "Failed to load DistilBERT block classifier (%s); falling back to rule.", e,
            )
            _LOAD_FAILED = True
            return None


def _classify_distilbert(text: str) -> BlockType:
    """Single-text DistilBERT inference. Caller handles exceptions."""
    bundle = _load_model_bundle()
    if bundle is None:
        raise RuntimeError("DistilBERT classifier not available")

    import torch
    tokenizer, model, device, id2label = bundle
    enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    idx = int(logits.argmax(-1).item())
    label = id2label[idx]
    if label not in LABELS:
        raise ValueError(f"Unexpected label from model: {label!r}")
    return label  # type: ignore[return-value]


def _rule_classify(text: str) -> BlockType:
    """Local proxy to chunker.classify_block_rule — function-level import avoids cycle."""
    from app.rag.chunker import classify_block_rule
    return classify_block_rule(text)


@functools.lru_cache(maxsize=4096)
def _classify_cached(text: str, backend: str) -> BlockType:
    """Inner cached classifier. Backend in cache key so flipping config is clean."""
    if backend == "rule":
        return _rule_classify(text)

    if backend == "distilbert":
        try:
            return _classify_distilbert(text)
        except Exception as e:
            logger.warning("DistilBERT inference failed (%s); falling back to rule.", e)
            return _rule_classify(text)

    if backend == "auto":
        if _load_model_bundle() is None:
            return _rule_classify(text)
        try:
            return _classify_distilbert(text)
        except Exception as e:
            logger.warning("DistilBERT inference failed (%s); falling back to rule.", e)
            return _rule_classify(text)

    raise ValueError(f"Unknown chunker_classifier backend: {backend!r}")


def classify_block(text: str) -> BlockType:
    """Classify a chunk into one of: text, code, formula, table.

    Backend chosen by ``settings.chunker_classifier`` (env CHUNKER_CLASSIFIER).
    Empty / whitespace-only input returns ``"text"`` without invoking any backend.
    """
    if not text or not text.strip():
        return "text"
    backend = (settings.chunker_classifier or "auto").strip().lower()
    return _classify_cached(text, backend)
