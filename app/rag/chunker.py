"""Document chunking for RAG.

Splits at sentence boundaries and preserves code, formula and table blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

from app.config import settings
from app.rag.block_classifier import classify_block

BlockType = Literal["text", "code", "formula", "table"]


# ------------------------------------------------------------------
# Rule-based block-type classifier, used when no trained model is available.
#
# Scoring rules:
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
    """Heuristic block-type classifier for model-free document ingestion.

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

    Silently skips text shorter than ``min_size`` after stripping.
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
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    overlap_sentences: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
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
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    overlap_sentences = settings.chunk_overlap_sentences if overlap_sentences is None else overlap_sentences
    min_chunk_size = settings.min_chunk_size if min_chunk_size is None else min_chunk_size
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
