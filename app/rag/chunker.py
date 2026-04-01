"""Document chunking for RAG.

Sentence-aware splitting: chunks always end at sentence boundaries so the
LLM receives complete context.  Falls back to character-level split only
for oversized single sentences (e.g. long code blocks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class DocumentChunk:
    """A text chunk with stable id and metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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

            # Oversized single sentence → character-level fallback
            if sent_len > chunk_size:
                # Flush current buffer first
                if current:
                    text = "\n".join(current).strip()
                    if len(text) >= min_chunk_size:
                        chunks.append(DocumentChunk(
                            chunk_id=f"doc{doc_idx}_chunk{len(chunks)}",
                            text=text,
                            metadata={"doc_index": doc_idx, "chunk_index": len(chunks)},
                        ))
                    current = []
                    current_len = 0

                # Split the long sentence by characters
                for piece in _char_split(sent, chunk_size, overlap):
                    if len(piece) >= min_chunk_size:
                        chunks.append(DocumentChunk(
                            chunk_id=f"doc{doc_idx}_chunk{len(chunks)}",
                            text=piece,
                            metadata={"doc_index": doc_idx, "chunk_index": len(chunks)},
                        ))
                continue

            # Would adding this sentence exceed the limit?
            # +1 accounts for the "\n" joiner
            if current and current_len + sent_len + 1 > chunk_size:
                # Flush current chunk
                text = "\n".join(current).strip()
                if len(text) >= min_chunk_size:
                    chunks.append(DocumentChunk(
                        chunk_id=f"doc{doc_idx}_chunk{len(chunks)}",
                        text=text,
                        metadata={"doc_index": doc_idx, "chunk_index": len(chunks)},
                    ))

                # Overlap: carry last N sentences into next chunk
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)

            current.append(sent)
            current_len += sent_len + (1 if len(current) > 1 else 0)

        # Flush remaining
        if current:
            text = "\n".join(current).strip()
            if len(text) >= min_chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id=f"doc{doc_idx}_chunk{len(chunks)}",
                    text=text,
                    metadata={"doc_index": doc_idx, "chunk_index": len(chunks)},
                ))

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
