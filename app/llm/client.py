# -*- coding: utf-8 -*-
"""Unified LLM client abstraction.

Supports multiple providers via OpenAI-compatible API format:
  - openai:   OpenAI / Azure / any OpenAI-compatible endpoint
  - deepseek: DeepSeek official API
  - zhipu:    Zhipu GLM series
  - local:    Deterministic fallback (no API call, for testing only)

Provider and credentials are configured via .env / environment variables.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import settings
from app.llm.prompts import build_answer_prompt

logger = logging.getLogger(__name__)

# Default base URLs per provider (used when LLM_BASE_URL is not set)
_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}


class LLMClient:
    """LLM client that dispatches to real API or local fallback."""

    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower().strip()
        self.model = settings.llm_model_name
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = None

        if self.provider != "local":
            self._init_api_client()

    def _init_api_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        api_key = settings.llm_api_key
        if not api_key:
            logger.warning(
                "LLM_API_KEY not set for provider '%s', falling back to local.",
                self.provider,
            )
            self.provider = "local"
            return

        base_url = settings.llm_base_url or _DEFAULT_BASE_URLS.get(self.provider)

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(
                "LLM client initialized: provider=%s, model=%s, base_url=%s",
                self.provider, self.model, base_url,
            )
        except ImportError:
            logger.warning(
                "openai package not installed. Install with: pip install openai. "
                "Falling back to local."
            )
            self.provider = "local"
        except Exception as e:
            logger.warning("Failed to init LLM client: %s. Falling back to local.", e)
            self.provider = "local"

    def generate_with_context(self, *, system_prompt: str, user_message: str) -> str:
        """Main generation method: system prompt + assembled user message.

        If provider is not 'local' and API client is available, calls the real
        LLM API. Otherwise falls back to deterministic local generation.
        """
        if self.provider != "local" and self._client is not None:
            return self._api_generate(system_prompt, user_message)
        return self._local_fallback(system_prompt, user_message)

    def _api_generate(self, system_prompt: str, user_message: str) -> str:
        """Call OpenAI-compatible chat completions API."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.info(
                "LLM API response: model=%s, tokens=%s, length=%d",
                self.model,
                getattr(response.usage, "total_tokens", "?"),
                len(content),
            )
            return content
        except Exception as e:
            logger.error("LLM API call failed: %s. Falling back to local.", e)
            return self._local_fallback(system_prompt, user_message)

    def generate(self, query: str, contexts: List[str], *, recent_context: Optional[str] = None) -> str:
        """Legacy interface -- kept for backward compatibility."""
        _ = build_answer_prompt(query, contexts, recent_context=recent_context)
        if not contexts:
            return "\u6211\u6682\u65f6\u6ca1\u6709\u68c0\u7d22\u5230\u53ef\u7528\u4e0a\u4e0b\u6587\u3002\u4f60\u7684\u95ee\u9898\u662f\uff1a" + query
        context_block = "\n".join("- " + chunk for chunk in contexts[:3])
        return (
            "\u57fa\u4e8e\u68c0\u7d22\u5230\u7684\u6587\u6863\uff0c\u6211\u7684\u56de\u7b54\u5982\u4e0b\uff1a\n"
            + context_block + "\n\n"
            + "\u9488\u5bf9\u4f60\u7684\u95ee\u9898\u300c" + query + "\u300d\uff0c\u8bf7\u5148\u53c2\u8003\u4ee5\u4e0a\u4fe1\u606f\u3002"
        )

    # ------------------------------------------------------------------
    # Local deterministic fallback (no API, for testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _local_fallback(system_prompt: str, user_message: str) -> str:
        """Deterministic local response generator (no LLM API call)."""
        evidence_lines = []
        question_line = ""
        in_question = False
        for line in user_message.splitlines():
            stripped = line.strip()
            if stripped.startswith("[\u8bc1\u636e"):
                evidence_lines.append(stripped)
            if stripped == "## \u7528\u6237\u95ee\u9898":
                in_question = True
                continue
            if in_question and stripped:
                question_line = stripped
                in_question = False

        q = question_line or "unknown"
        has_evidence = bool(evidence_lines)

        if not has_evidence:
            return (
                "[DEBUG_TRACE_START]\n"
                "Step 1: Question Understanding\n"
                "- \u95ee\u9898\uff1a" + q + "\n"
                "- \u7c7b\u578b\uff1a\u65e0\u6cd5\u5224\u5b9a\n\n"
                "Step 2: Evidence Analysis\n"
                "- \u65e0\u76f8\u5173\u8bc1\u636e\n\n"
                "Step 3: Synthesis Strategy\n"
                "- \u8bc1\u636e\u4e0d\u8db3\uff0c\u65e0\u6cd5\u751f\u6210\u56de\u7b54\n\n"
                "Step 4: Answer Construction\n"
                "- \u8fd4\u56de\u6807\u51c6\u4e0d\u8db3\u56de\u590d\n"
                "[DEBUG_TRACE_END]\n\n"
                "[FINAL_ANSWER]\n"
                "\u8bc1\u636e\u4e0d\u8db3\uff0c\u65e0\u6cd5\u56de\u7b54\u8be5\u95ee\u9898\u3002\n"
                "[/FINAL_ANSWER]\n\n"
                "[EVIDENCE_USED]\n"
                "[/EVIDENCE_USED]"
            )

        evidence_summaries = []
        for el in evidence_lines[:5]:
            idx = el.find("] ")
            text = el[idx + 2:].strip() if idx != -1 else el.strip()
            evidence_summaries.append(text)

        trace_evidence = "\n".join("- " + e for e in evidence_summaries)
        cited = "\n".join("- " + e for e in evidence_summaries)
        key_points = "\n".join(
            str(i + 1) + ". " + e for i, e in enumerate(evidence_summaries)
        )
        answer_text = (
            "\u5173\u4e8e\u300c" + q + "\u300d\uff0c"
            "\u6839\u636e\u5df2\u68c0\u7d22\u5230\u7684\u8bc1\u636e\uff0c\u603b\u7ed3\u5982\u4e0b\uff1a\n"
            + key_points
        )

        return (
            "[DEBUG_TRACE_START]\n"
            "Step 1: Question Understanding\n"
            "- \u95ee\u9898\uff1a" + q + "\n"
            "- \u7c7b\u578b\uff1a\u4e8b\u5b9e\u67e5\u8be2\n\n"
            "Step 2: Evidence Analysis\n"
            + trace_evidence + "\n"
            "- \u6bcf\u6761\u8bc1\u636e\u5747\u4e0e\u7528\u6237\u95ee\u9898\u300c" + q + "\u300d\u76f8\u5173\n\n"
            "Step 3: Synthesis Strategy\n"
            "- \u4f7f\u7528 (B) \u7efc\u5408\u6a21\u5f0f\n"
            "- \u539f\u56e0\uff1a\u9700\u7ed3\u5408\u7528\u6237\u95ee\u9898\u4e0e\u591a\u6761\u8bc1\u636e\u7efc\u5408\u56de\u7b54\n\n"
            "Step 4: Answer Construction\n"
            "- \u56f4\u7ed5\u95ee\u9898\u300c" + q + "\u300d\u7efc\u5408\u8bc1\u636e\u751f\u6210\u7ed3\u8bba\n"
            "[DEBUG_TRACE_END]\n\n"
            "[FINAL_ANSWER]\n"
            + answer_text + "\n"
            "[/FINAL_ANSWER]\n\n"
            "[EVIDENCE_USED]\n"
            + cited + "\n"
            "[/EVIDENCE_USED]"
        )
