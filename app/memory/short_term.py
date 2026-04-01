# -*- coding: utf-8 -*-
"""Redis-backed short-term memory.

Sliding window context with filtered output:
  - conversation history (recent N rounds)
  - tool call traces
  - current task state

Only these three types are included in the context sent to LLM.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Session short-term memory backed by Redis."""

    def __init__(self, max_rounds: int = 8) -> None:
        self.max_rounds = max_rounds
        self.max_messages = max_rounds * 2
        self.max_traces = 50
        self._redis = None
        try:
            import redis

            self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self._redis.ping()
        except Exception:
            self._redis = None

    # ------------------------------------------------------------------
    # Key builders
    # ------------------------------------------------------------------

    def _messages_key(self, session_id: str) -> str:
        return f"deepsearch:st:{session_id}:messages"

    def _traces_key(self, session_id: str) -> str:
        return f"deepsearch:st:{session_id}:traces"

    def _task_state_key(self, session_id: str) -> str:
        return f"deepsearch:st:{session_id}:task_state"

    # ------------------------------------------------------------------
    # Conversation history (sliding window)
    # ------------------------------------------------------------------

    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """Save one message and keep only recent N rounds."""
        if self._redis is None:
            return False
        try:
            payload = {
                "role": role,
                "content": content,
                "ts": time.time(),
            }
            key = self._messages_key(session_id)
            self._redis.rpush(key, json.dumps(payload, ensure_ascii=False))
            self._redis.ltrim(key, -self.max_messages, -1)
            return True
        except Exception:
            return False

    def get_recent_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Return recent messages as structured list."""
        if self._redis is None:
            return []
        try:
            raw_items = self._redis.lrange(self._messages_key(session_id), 0, -1)
            messages: List[Dict[str, str]] = []
            for raw in raw_items:
                try:
                    item = json.loads(raw)
                except Exception:
                    continue
                role = str(item.get("role", "user"))
                content = str(item.get("content", "")).strip()
                if content:
                    messages.append({"role": role, "content": content})
            return messages
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Tool call traces
    # ------------------------------------------------------------------

    def save_trace(self, session_id: str, trace_type: str, trace_data: Dict[str, Any]) -> bool:
        """Save one trace record and keep recent trace window."""
        if self._redis is None:
            return False
        try:
            payload = {
                "trace_type": trace_type,
                "trace_data": trace_data,
                "ts": time.time(),
            }
            key = self._traces_key(session_id)
            self._redis.rpush(key, json.dumps(payload, ensure_ascii=False))
            self._redis.ltrim(key, -self.max_traces, -1)
            return True
        except Exception:
            return False

    def get_recent_traces(self, session_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Return recent traces as (trace_type, trace_data)."""
        if self._redis is None:
            return []
        try:
            raw_items = self._redis.lrange(self._traces_key(session_id), 0, -1)
            out: List[Tuple[str, Dict[str, Any]]] = []
            for raw in raw_items:
                try:
                    item = json.loads(raw)
                except Exception:
                    continue
                trace_type = str(item.get("trace_type", "unknown"))
                trace_data = item.get("trace_data") or {}
                if not isinstance(trace_data, dict):
                    trace_data = {"value": trace_data}
                out.append((trace_type, trace_data))
            return out
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Current task state
    # ------------------------------------------------------------------

    def save_task_state(self, session_id: str, state: Dict[str, Any]) -> bool:
        """Save current task state (overwrites previous)."""
        if self._redis is None:
            return False
        try:
            key = self._task_state_key(session_id)
            self._redis.set(key, json.dumps(state, ensure_ascii=False))
            return True
        except Exception:
            return False

    def get_task_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current task state."""
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._task_state_key(session_id))
            if raw:
                return json.loads(raw)
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Filtered context assembly
    # ------------------------------------------------------------------

    def get_recent_context(
        self,
        session_id: str,
        *,
        include_history: bool = True,
        include_traces: bool = True,
        include_task_state: bool = True,
    ) -> str:
        """Return filtered short-term context as prompt-ready text.

        Only includes:
          1. Conversation history (sliding window)
          2. Tool call traces (recent)
          3. Current task state

        Args:
            session_id: session to read from.
            include_history: include conversation history.
            include_traces: include tool call traces.
            include_task_state: include current task state.

        Returns:
            Formatted text ready for injection into LLM prompt.
        """
        sections: List[str] = []

        # 1. Conversation history
        if include_history:
            messages = self.get_recent_messages(session_id)
            if messages:
                lines = []
                for msg in messages:
                    role_name = "User" if msg["role"] == "user" else "Assistant"
                    lines.append(f"{role_name}: {msg['content']}")
                sections.append("对话历史：\n" + "\n".join(lines))

        # 2. Tool call traces
        if include_traces:
            traces = self.get_recent_traces(session_id)
            if traces:
                trace_lines = []
                for ttype, tdata in traces[-5:]:  # Last 5 traces only
                    summary = str(tdata.get("summary", ""))[:100] if tdata.get("summary") else ttype
                    trace_lines.append(f"- [{ttype}] {summary}")
                sections.append("工具调用轨迹：\n" + "\n".join(trace_lines))

        # 3. Current task state
        if include_task_state:
            state = self.get_task_state(session_id)
            if state:
                state_text = json.dumps(state, ensure_ascii=False, indent=0)
                sections.append(f"当前任务状态：\n{state_text}")

        return "\n\n".join(sections)
