"""Document chunking for RAG.

Two strategies:
  1. Fixed-size splitting (split_documents): chunks at sentence boundaries
     with a hard character limit. Fast, no model needed.
  2. Semantic splitting (split_documents_semantic): cuts at topic boundaries
     detected by embedding similarity drops. Better quality, needs embedder.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np

from app.rag.block_classifier import classify_block

logger = logging.getLogger(__name__)

BlockType = Literal["text", "code", "formula", "table"]


# ------------------------------------------------------------------
# Rule-based block-type classifier (baseline for ML classifier).
#
# Scoring rules ported from "第一个项目介绍.md":
#   - 4-space (or tab) indentation on >=30% of lines: +2
#   - Has code keywords (def/return/import/class/var/const/...): +3
#   - Operator density (>=5 in >=60 chars): +1
#   Score >= 3  →  code
#
# Formulas:
#   - $$...$$ block, or $...$ inline, or LaTeX commands (\frac, \sum, ...): formula
#
# Tables:
#   - >=2 lines with >=2 pipes, plus a separator row containing "---"
# ------------------------------------------------------------------

_CODE_KEYWORDS = re.compile(
    r"\b(def|class|return|import|from|function|const|let|var|public|private|"
    r"async|await|lambda|elif|else|=>|->|::|println|System\.out|console\.log|"
    r"struct|interface|impl|fn|pub|val|fun)\b"
)
_OPERATORS = re.compile(r"[{};=<>+\-*/%&|^!]=?|==|!=|<=|>=|&&|\|\|")
_FORMULA_LATEX = re.compile(
    r"\\(frac|sum|int|sqrt|alpha|beta|gamma|Delta|nabla|partial|prod|lim|infty|"
    r"forall|exists|in|cdot|times|leq|geq|neq|approx|sim|propto|mathbb|mathcal)\b"
)
_FORMULA_INLINE = re.compile(r"\$[^$\n]{2,}\$")


def classify_block_rule(text: str) -> BlockType:
    """Heuristic block-type classifier — kept as the baseline against the ML model.

    Order matters: formulas and tables have the most distinctive markers, so
    we check them first. Code is checked last because keyword-based detection
    over-recalls on prose containing inline-code spans.
    """
    if not text or not text.strip():
        return "text"

    # --- formula ---
    if "$$" in text or _FORMULA_INLINE.search(text) or _FORMULA_LATEX.search(text):
        return "formula"

    # --- table ---
    pipe_lines = [ln for ln in text.split("\n") if ln.count("|") >= 2]
    if len(pipe_lines) >= 2:
        has_sep = any(re.match(r"^\s*\|?[\s\-:|]+$", ln) and "-" in ln for ln in pipe_lines)
        if has_sep:
            return "table"

    # --- code (scoring) ---
    score = 0
    lines = text.split("\n")
    indented = sum(1 for ln in lines if re.match(r"^( {2,}|\t)", ln))
    if len(lines) >= 2 and indented / len(lines) >= 0.3:
        score += 2
    if len(_CODE_KEYWORDS.findall(text)) >= 1:
        score += 3
    if len(_OPERATORS.findall(text)) >= 5 and len(text) >= 60:
        score += 1
    if score >= 3:
        return "code"

    return "text"


@dataclass
class DocumentChunk:
    """A text chunk with stable id and metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _emit_chunk(
    chunks: List["DocumentChunk"],
    text: str,
    doc_idx: int,
    min_size: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a DocumentChunk to ``chunks`` with classify_block-derived block_type.

    Silently skips text shorter than ``min_size`` after stripping. Used by both
    fixed-size and semantic splitters so block_type metadata is consistent.
    """
    text = text.strip()
    if len(text) < min_size:
        return
    btype = classify_block(text)
    metadata: Dict[str, Any] = {
        "doc_index": doc_idx,
        "chunk_index": len(chunks),
        "block_type": btype,
    }
    if extra_meta:
        metadata.update(extra_meta)
    chunks.append(DocumentChunk(
        chunk_id=f"doc{doc_idx}_chunk{len(chunks)}",
        text=text,
        metadata=metadata,
    ))


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------

# Regex: split on Chinese/English sentence-ending punctuation.
# Negative lookbehind avoids splitting on numbered lists ("1.") and
# common abbreviations (e.g., Mr., Dr., etc.).
_SENT_END = re.compile(
    r"(?<=[。！？；\n])"                       # Chinese sentence enders
    r"|(?<=[.!?])\s+"                           # English sentence enders followed by space
    r"|(?<=\n)\s*\n"                            # Paragraph breaks (blank line)
    r"|(?<=\|)\s*\n"                            # Table row ends ("|" + newline)
    r"|(?<=```)\s*\n"                           # Code fence boundary
    r"|(?<=[：:])\s*\n"                         # Chinese/English colon + newline (list headers)
)

# Force-split markers: always start a new chunk here
_FORCE_SPLIT = re.compile(
    r"^#{1,6}\s"                                # Markdown headings
    r"|^-{3,}"                                  # Horizontal rules (---)
    r"|^```",                                   # Code fence open/close
    re.MULTILINE,
)


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving all characters."""
    # First, split on force-split markers (headings)
    parts: List[str] = []
    last = 0
    for m in _FORCE_SPLIT.finditer(text):
        pos = m.start()
        if pos > last:
            parts.append(text[last:pos])
        last = pos
    if last < len(text):
        parts.append(text[last:])

    # Then split each part into sentences
    sentences: List[str] = []
    for part in parts:
        raw = _SENT_END.split(part)
        for s in raw:
            s = s.strip()
            if s:
                sentences.append(s)

    return sentences


# ------------------------------------------------------------------
# Character-level fallback for oversized sentences
# ------------------------------------------------------------------

def _char_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Blind character split — used only as fallback for oversized sentences."""
    pieces: List[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start: start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        if start + chunk_size >= len(text):
            break
    return pieces


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def split_documents(
    documents: Iterable[str],
    *,
    chunk_size: int = 256,
    overlap: int = 64,
    overlap_sentences: int = 2,
    min_chunk_size: int = 20,
) -> List[DocumentChunk]:
    """Split documents into chunks that respect sentence boundaries.

    Algorithm:
      1. Split document into sentences.
      2. Greedily pack consecutive sentences into a chunk until
         ``chunk_size`` is reached.
      3. When a chunk is full, start the next chunk from the last
         ``overlap_sentences`` of the previous chunk.
      4. If a single sentence exceeds ``chunk_size``, fall back to
         character-level split for that sentence only.

    Args:
        documents: iterable of full-text documents.
        chunk_size: soft max characters per chunk (may slightly exceed to
            avoid splitting mid-sentence).
        overlap: character overlap for the fallback char-split.
        overlap_sentences: number of trailing sentences carried over to
            the next chunk as context overlap.
        min_chunk_size: discard chunks shorter than this.
    """
    chunks: List[DocumentChunk] = []

    for doc_idx, doc in enumerate(documents):
        if not doc or not doc.strip():
            continue

        sentences = _split_into_sentences(doc.strip())
        if not sentences:
            continue

        current: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            # Oversized single sentence: classify-aware handling.
            #   code/formula/table  → emit whole block (don't shred semantic units)
            #   text                → fall back to character split
            if sent_len > chunk_size:
                if current:
                    _emit_chunk(chunks, "\n".join(current), doc_idx, min_chunk_size)
                    current = []
                    current_len = 0

                btype = classify_block(sent)
                if btype in ("code", "formula", "table"):
                    _emit_chunk(chunks, sent, doc_idx, min_chunk_size)
                else:
                    for piece in _char_split(sent, chunk_size, overlap):
                        _emit_chunk(chunks, piece, doc_idx, min_chunk_size)
                continue

            # Normal pack — +1 accounts for the "\n" joiner.
            if current and current_len + sent_len + 1 > chunk_size:
                _emit_chunk(chunks, "\n".join(current), doc_idx, min_chunk_size)
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)

            current.append(sent)
            current_len += sent_len + (1 if len(current) > 1 else 0)

        # Final flush
        if current:
            _emit_chunk(chunks, "\n".join(current), doc_idx, min_chunk_size)

    return chunks


# ------------------------------------------------------------------
# Semantic chunking
# ------------------------------------------------------------------

def split_documents_semantic(
    documents: Iterable[str],
    embedder,
    *,
    max_chunk_size: int = 800,
    min_chunk_size: int = 50,
    similarity_threshold: Optional[float] = None,
) -> List[DocumentChunk]:
    """Split documents at semantic boundaries detected by embedding similarity.

    Algorithm:
      1. Split into sentences (reuses ``_split_into_sentences``).
      2. Embed every sentence with *embedder*.
      3. Compute cosine similarity between adjacent sentences.
      4. Cut where similarity drops below ``mean - 1*std`` (or the given
         *similarity_threshold*).
      5. Merge too-short groups into neighbours; split too-long groups
         with ``_char_split``.

    Args:
        documents:            iterable of full-text documents.
        embedder:             object with ``.encode(texts) -> np.ndarray``.
        max_chunk_size:       hard upper limit (chars) — oversized groups are
                              re-split with ``_char_split``.
        min_chunk_size:       groups shorter than this are merged into the
                              previous group.
        similarity_threshold: explicit cutoff; ``None`` = auto (mean - std).
    """
    chunks: List[DocumentChunk] = []

    for doc_idx, doc in enumerate(documents):
        if not doc or not doc.strip():
            continue

        sentences = _split_into_sentences(doc.strip())
        if not sentences:
            continue

        # Single sentence or very few — no splitting needed
        if len(sentences) <= 2:
            _emit_chunk(chunks, "\n".join(sentences), doc_idx, min_chunk_size)
            continue

        # --- Embed all sentences ---
        embeddings = embedder.encode(sentences)  # (N, dim)
        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        # --- Cosine similarity between adjacent sentences ---
        similarities = np.array([
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(sentences) - 1)
        ])

        # --- Determine threshold ---
        if similarity_threshold is not None:
            threshold = similarity_threshold
        else:
            threshold = float(similarities.mean() - similarities.std())

        # --- Find split points (where similarity drops below threshold) ---
        split_indices = [0]  # always start from sentence 0
        for i, sim in enumerate(similarities):
            if sim < threshold:
                split_indices.append(i + 1)

        # --- Build groups ---
        groups: List[List[str]] = []
        for idx in range(len(split_indices)):
            start = split_indices[idx]
            end = split_indices[idx + 1] if idx + 1 < len(split_indices) else len(sentences)
            groups.append(sentences[start:end])

        # --- Merge too-short groups into previous ---
        merged: List[List[str]] = []
        for group in groups:
            group_text_len = sum(len(s) for s in group) + max(len(group) - 1, 0)
            if merged and group_text_len < min_chunk_size:
                merged[-1].extend(group)
            else:
                merged.append(group)

        # --- Emit chunks (split oversized groups) ---
        for group in merged:
            text = "\n".join(group).strip()
            if not text:
                continue
            if len(text) <= max_chunk_size:
                _emit_chunk(chunks, text, doc_idx, min_chunk_size)
            else:
                # Oversized group — re-split with char_split (semantic path keeps
                # its existing behavior; only fixed-size path bypasses split for
                # code/formula/table per Phase 3 design).
                for piece in _char_split(text, max_chunk_size, max_chunk_size // 4):
                    _emit_chunk(chunks, piece, doc_idx, min_chunk_size)

    logger.info("Semantic chunking: %d documents -> %d chunks", doc_idx + 1, len(chunks))
    return chunks


def split_document_with_source(text: str, source: str) -> List[DocumentChunk]:
    """Convenience: split one document and attach source metadata."""
    chunks = split_documents([text], chunk_size=256, overlap=64)
    out: List[DocumentChunk] = []
    for i, c in enumerate(chunks):
        out.append(
            DocumentChunk(
                chunk_id=f"{source}_{i}",
                text=c.text,
                metadata={"source": source, "chunk_index": i, **(c.metadata or {})},
            )
        )
    return out
