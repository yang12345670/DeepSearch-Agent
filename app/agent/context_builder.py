# -*- coding: utf-8 -*-
"""Clean context builder for final LLM input assembly.

Single-pass flow: retrieve -> filter -> build context -> generate -> parse.

Token Budget system:
  - P0: System prompt (fixed, incompressible)
  - P1: User question (fixed, incompressible)
  - P2: RAG evidence (60% of remaining budget)
  - P3: Conversation history (25% — recent 2 rounds full, older rounds summarized)
  - P4: Long-term memory (15% — ranked by recall score, truncate low-score)

LLM output protocol (3 blocks):
  [DEBUG_TRACE_START] ... [DEBUG_TRACE_END]
  [FINAL_ANSWER] ... [/FINAL_ANSWER]
  [EVIDENCE_USED] ... [/EVIDENCE_USED]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.config import settings
from app.rag.chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_tokenizer = None


def _get_tokenizer():
    """Lazy-load tiktoken encoder. Fallback to char-based estimation."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        import tiktoken
        _tokenizer = tiktoken.encoding_for_model(settings.tokenizer_model_name)
        logger.info("Token counter: using tiktoken for %s", settings.tokenizer_model_name)
    except Exception:
        _tokenizer = None
        logger.info("Token counter: tiktoken unavailable, using configured character estimation")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text. Uses tiktoken if available, else configured character estimation."""
    if not text:
        return 0
    enc = _get_tokenizer()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, int(len(text) / settings.token_estimate_chars_per_token))


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """One citation linking an answer reference [n] to its source."""
    id: int
    source: str
    text: str


@dataclass
class RAGResult:
    """Structured answer returned by the RAG agent and orchestrator.

    Fields:
        answer:        Clean user-facing text (from FINAL_ANSWER block).
        debug_trace:   Internal reasoning steps (from DEBUG_TRACE block).
        evidence_used: Cited evidence list (from EVIDENCE_USED block).
        citations:     Source-linked citation list for the frontend.
        used_chunk_ids: chunk IDs used in this result (for sequential dedup).
    """
    answer: str
    debug_trace: str = ""
    evidence_used: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    used_chunk_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = settings.rag_system_prompt


# ---------------------------------------------------------------------------
# Token budget allocation
# ---------------------------------------------------------------------------

@dataclass
class TokenBudget:
    """Token budget allocation for context assembly."""
    total: int
    system_prompt: int
    user_question: int
    evidence: int
    history: int
    memory: int


def compute_token_budget(
    query: str,
    *,
    output_reserve: Optional[int] = None,
) -> TokenBudget:
    """Compute token budget allocation based on model context window.

    Priority order: P0 system > P1 query > P2 evidence > P3 history > P4 memory
    """
    output_reserve = settings.token_output_reserve if output_reserve is None else output_reserve
    context_window = settings.context_window_tokens
    system_tokens = count_tokens(settings.rag_system_prompt)
    query_tokens = count_tokens(query) + settings.token_query_overhead

    # Available budget after fixed allocations + output reserve
    available = context_window - output_reserve - system_tokens - query_tokens
    available = max(0, available)

    # Distribute remaining budget by configured ratios
    evidence_budget = int(available * settings.token_budget_evidence_ratio)
    history_budget = int(available * settings.token_budget_history_ratio)
    memory_budget = available - evidence_budget - history_budget  # remainder to memory

    return TokenBudget(
        total=context_window,
        system_prompt=system_tokens,
        user_question=query_tokens,
        evidence=evidence_budget,
        history=history_budget,
        memory=memory_budget,
    )


# ---------------------------------------------------------------------------
# Budget-aware section builders
# ---------------------------------------------------------------------------

def _truncate_evidence_by_budget(
    evidence_texts: List[str],
    budget_tokens: int,
) -> List[str]:
    """Fill evidence into budget, prioritizing higher-ranked (earlier) items.

    Strategy:
      - Top-1 always gets full text
      - Subsequent items added in full if budget allows
      - If an item doesn't fit in full, keep first + last paragraph (head-tail)
      - If even head-tail doesn't fit, skip
    """
    if not evidence_texts or budget_tokens <= 0:
        return []

    result: List[str] = []
    used = 0
    header_overhead = settings.token_evidence_header_overhead

    for i, text in enumerate(evidence_texts):
        item_tokens = count_tokens(text) + header_overhead

        if used + item_tokens <= budget_tokens:
            # Fits in full
            result.append(text)
            used += item_tokens
        else:
            remaining = budget_tokens - used
            if remaining < settings.token_min_evidence_tokens:
                break  # Not enough space for meaningful content

            # Head-tail truncation: keep first and last paragraph
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            if len(paragraphs) >= 3:
                head = paragraphs[0]
                tail = paragraphs[-1]
                compressed = head + "\n...\n" + tail
            else:
                # Single paragraph: character-level truncation
                # Estimate chars from remaining tokens
                char_limit = int(remaining * settings.token_estimate_chars_per_token)
                compressed = text[:char_limit] + "..."

            comp_tokens = count_tokens(compressed) + header_overhead
            if used + comp_tokens <= budget_tokens:
                result.append(compressed)
                used += comp_tokens
            # else: skip this evidence entirely

    logger.info(
        "Evidence budget: %d tokens, used %d tokens, kept %d/%d items",
        budget_tokens, used, len(result), len(evidence_texts),
    )
    return result


def _truncate_history_by_budget(
    history_text: str,
    budget_tokens: int,
) -> str:
    """Compress conversation history to fit budget.

    Strategy:
      - Keep most recent 2 rounds (4 messages) in full
      - Older messages: keep only first line (as summary)
      - If still over budget, drop oldest summaries
    """
    if not history_text.strip() or budget_tokens <= 0:
        return ""

    lines = history_text.strip().split("\n")

    # Find message boundaries (lines starting with "User:" or "Assistant:")
    msg_indices: List[int] = []
    for i, line in enumerate(lines):
        if line.startswith("User:") or line.startswith("Assistant:"):
            msg_indices.append(i)

    if not msg_indices:
        # Not structured as messages, just truncate
        text = history_text.strip()
        if count_tokens(text) <= budget_tokens:
            return text
        char_limit = int(budget_tokens * settings.token_estimate_chars_per_token)
        return text[:char_limit] + "..."

    # Split into messages
    messages: List[str] = []
    for idx, start in enumerate(msg_indices):
        end = msg_indices[idx + 1] if idx + 1 < len(msg_indices) else len(lines)
        messages.append("\n".join(lines[start:end]))

    if not messages:
        return ""

    # Keep the configured recent messages in full, summarize older messages.
    recent_count = min(settings.history_recent_messages, len(messages))
    recent_msgs = messages[-recent_count:] if recent_count else []
    older_msgs = messages[:-recent_count] if recent_count else messages

    # Summarize older messages: keep first line only
    older_summaries = []
    for msg in older_msgs:
        first_line = msg.split("\n")[0]
        # Truncate long first lines
        if len(first_line) > settings.history_summary_chars:
            first_line = first_line[:settings.history_summary_chars] + "..."
        older_summaries.append(first_line)

    # Build result: summaries + recent full messages
    parts: List[str] = []
    if older_summaries:
        parts.append("[早期对话摘要]")
        parts.extend(older_summaries)
        parts.append("")  # blank line separator

    parts.extend(recent_msgs)
    result = "\n".join(parts)

    # Final check: if still over budget, drop oldest summaries one by one
    while count_tokens(result) > budget_tokens and older_summaries:
        older_summaries.pop(0)
        parts = []
        if older_summaries:
            parts.append("[早期对话摘要]")
            parts.extend(older_summaries)
            parts.append("")
        parts.extend(recent_msgs)
        result = "\n".join(parts)

    # Last resort: if recent messages alone exceed budget, truncate
    if count_tokens(result) > budget_tokens:
        char_limit = int(budget_tokens * settings.token_estimate_chars_per_token)
        result = result[:char_limit] + "..."

    return result


def _truncate_memory_by_budget(
    memory_text: str,
    budget_tokens: int,
) -> str:
    """Truncate long-term memory to fit budget.

    Strategy: memory items are already ranked by recall score,
    so just truncate from the end.
    """
    if not memory_text.strip() or budget_tokens <= 0:
        return ""

    if count_tokens(memory_text) <= budget_tokens:
        return memory_text

    # Split by memory items (lines starting with "- ")
    lines = memory_text.strip().split("\n")
    result_lines: List[str] = []
    used = 0

    for line in lines:
        line_tokens = count_tokens(line) + 1  # +1 for newline
        if used + line_tokens <= budget_tokens:
            result_lines.append(line)
            used += line_tokens
        else:
            break

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Evidence filtering
# ---------------------------------------------------------------------------

def filter_retrieved_docs(
    candidates: List[Tuple[DocumentChunk, float]],
    *,
    score_threshold: Optional[float] = None,
    max_docs: Optional[int] = None,
    score_gap_ratio: Optional[float] = None,
) -> List[Tuple[DocumentChunk, float]]:
    """Filter retrieved docs by relevance score and cap count.

    Args:
        candidates: (chunk, score) pairs from knowledge_base.retrieve().
        score_threshold: minimum reranker score to keep a doc.
        max_docs: maximum number of docs to retain.
        score_gap_ratio: drop docs scoring below top_score * this ratio.

    Returns:
        Filtered list sorted by descending score.
    """
    score_threshold = settings.rag_score_threshold if score_threshold is None else score_threshold
    max_docs = settings.rag_max_docs if max_docs is None else max_docs
    score_gap_ratio = settings.rag_score_gap_ratio if score_gap_ratio is None else score_gap_ratio
    filtered = [
        (chunk, score)
        for chunk, score in candidates
        if score >= score_threshold and chunk.text.strip()
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    if filtered:
        top_score = filtered[0][1]
        filtered = [
            (chunk, score) for chunk, score in filtered
            if score >= top_score * score_gap_ratio
        ]
    return filtered[:max_docs]


# ---------------------------------------------------------------------------
# Context assembly (budget-aware)
# ---------------------------------------------------------------------------

def build_final_context(
    *,
    query: str,
    evidence_texts: List[str],
    long_term_memory: str = "",
    short_term_memory: str = "",
) -> str:
    """Assemble the user message for the LLM with token budget management.

    Each section is compressed to fit within its allocated budget.
    Priority: evidence > history > memory.
    """
    budget = compute_token_budget(query)
    logger.info(
        "Token budget: total=%d, system=%d, query=%d, evidence=%d, history=%d, memory=%d",
        budget.total, budget.system_prompt, budget.user_question,
        budget.evidence, budget.history, budget.memory,
    )

    sections: List[str] = []

    # P4: Long-term memory (lowest priority)
    if long_term_memory.strip():
        compressed_memory = _truncate_memory_by_budget(long_term_memory.strip(), budget.memory)
        if compressed_memory:
            sections.append("## 长期记忆\n" + compressed_memory)

    # P3: Conversation history
    if short_term_memory.strip():
        compressed_history = _truncate_history_by_budget(short_term_memory.strip(), budget.history)
        if compressed_history:
            sections.append("## 对话历史\n" + compressed_history)

    # P2: Evidence (highest variable-priority)
    if evidence_texts:
        compressed_evidence = _truncate_evidence_by_budget(evidence_texts, budget.evidence)
        if compressed_evidence:
            evidence_block = "\n\n".join(
                "[证据" + str(i + 1) + "] " + text.strip()
                for i, text in enumerate(compressed_evidence)
            )
            sections.append("## 参考证据\n" + evidence_block)
        else:
            sections.append("## 参考证据\n无相关证据")
    else:
        sections.append("## 参考证据\n无相关证据")

    # P1: User question (fixed)
    sections.append("## 用户问题\n" + query.strip())

    sections.append(settings.rag_answer_instructions)

    final_context = "\n\n".join(sections)

    # Log actual usage
    actual_tokens = count_tokens(final_context)
    logger.info(
        "Context assembled: %d tokens (budget available: %d)",
        actual_tokens, budget.evidence + budget.history + budget.memory,
    )

    return final_context


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_RE_TRACE = re.compile(
    r"\[DEBUG_TRACE_START\]\s*\n?(.*?)\n?\s*\[DEBUG_TRACE_END\]",
    re.DOTALL,
)
_RE_ANSWER = re.compile(
    r"\[FINAL_ANSWER\]\s*\n?(.*?)\n?\s*\[/FINAL_ANSWER\]",
    re.DOTALL,
)
_RE_EVIDENCE = re.compile(
    r"\[EVIDENCE_USED\]\s*\n?(.*?)\n?\s*\[/EVIDENCE_USED\]",
    re.DOTALL,
)


_PARSE_FALLBACK = "系统已生成中间步骤，但最终答案解析失败。"


def parse_llm_response(raw: str) -> Tuple[str, str, List[str]]:
    """Extract (debug_trace, final_answer, evidence_used) from LLM output.

    Guarantees:
      - final_answer is NEVER empty.  Three-tier fallback:
        1. [FINAL_ANSWER] tag content
        2. Raw text after stripping trace/evidence blocks
        3. Hard-coded sentinel: "系统已生成中间步骤，但最终答案解析失败。"
    """
    logger.debug("parse_llm_response raw input (%d chars):\n%s", len(raw), raw)

    trace_match = _RE_TRACE.search(raw)
    answer_match = _RE_ANSWER.search(raw)
    evidence_match = _RE_EVIDENCE.search(raw)

    # --- debug_trace ---
    debug_trace = trace_match.group(1).strip() if trace_match else ""

    # --- final_answer: 3-tier fallback ---
    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        # Tier 2: strip known blocks, use remaining text
        cleaned = raw
        for pattern in (_RE_TRACE, _RE_EVIDENCE):
            cleaned = pattern.sub("", cleaned)
        final_answer = cleaned.strip()

    if not final_answer:
        # Tier 3: guaranteed non-empty sentinel
        final_answer = _PARSE_FALLBACK
        logger.warning(
            "FINAL_ANSWER empty after parsing. debug_trace present=%s. "
            "Raw output (%d chars): %.200s",
            bool(debug_trace), len(raw), raw,
        )

    # --- evidence_used ---
    if evidence_match:
        evidence_used = [
            line.lstrip("- ").strip()
            for line in evidence_match.group(1).strip().splitlines()
            if line.strip() and line.strip() != "-"
        ]
    else:
        evidence_used = []

    logger.debug(
        "parse_llm_response result: trace=%d chars, answer=%d chars, evidence=%d items",
        len(debug_trace), len(final_answer), len(evidence_used),
    )
    return debug_trace, final_answer, evidence_used
