# -*- coding: utf-8 -*-
"""Chat history persistence via Supabase.

Provides session CRUD and message storage that survives Redis restarts.
Agent logic is NOT affected — this is a pure storage layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init Supabase client."""
    global _client
    if _client is not None:
        return _client
    url = settings.supabase_url
    key = settings.supabase_key
    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set, chat history disabled.")
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        logger.warning("Failed to init Supabase client: %s", e)
        return None


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------

def create_session(session_id: str, title: str = "New Chat") -> Optional[Dict]:
    sb = _get_client()
    if sb is None:
        return None
    try:
        # Check if session already exists — don't overwrite title/preview
        existing = sb.table("chat_sessions").select("session_id").eq(
            "session_id", session_id
        ).execute()
        if existing.data:
            return existing.data[0]
        r = sb.table("chat_sessions").insert({
            "session_id": session_id,
            "title": title,
            "preview": "",
        }).execute()
        return r.data[0] if r.data else None
    except Exception as e:
        logger.error("create_session failed: %s", e)
        return None


def list_sessions() -> List[Dict]:
    sb = _get_client()
    if sb is None:
        return []
    try:
        r = sb.table("chat_sessions").select("*").order(
            "last_active", desc=True
        ).execute()
        return r.data or []
    except Exception as e:
        logger.error("list_sessions failed: %s", e)
        return []


def update_session(session_id: str, **fields) -> None:
    sb = _get_client()
    if sb is None:
        return
    try:
        fields["last_active"] = datetime.now(timezone.utc).isoformat()
        sb.table("chat_sessions").update(fields).eq(
            "session_id", session_id
        ).execute()
    except Exception as e:
        logger.error("update_session failed: %s", e)


def delete_session(session_id: str) -> bool:
    sb = _get_client()
    if sb is None:
        return False
    try:
        # Messages cascade-deleted via FK
        sb.table("chat_sessions").delete().eq(
            "session_id", session_id
        ).execute()
        return True
    except Exception as e:
        logger.error("delete_session failed: %s", e)
        return False


# ------------------------------------------------------------------
# Messages
# ------------------------------------------------------------------

def save_message(session_id: str, role: str, content: str) -> None:
    sb = _get_client()
    if sb is None:
        return
    try:
        sb.table("chat_messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
        }).execute()
        # Update session preview & last_active
        preview = content[:60] if role == "user" else None
        fields: Dict[str, Any] = {"last_active": datetime.now(timezone.utc).isoformat()}
        if preview:
            fields["preview"] = preview
        sb.table("chat_sessions").update(fields).eq(
            "session_id", session_id
        ).execute()
    except Exception as e:
        logger.error("save_message failed: %s", e)


def get_messages(session_id: str) -> List[Dict]:
    sb = _get_client()
    if sb is None:
        return []
    try:
        r = sb.table("chat_messages").select(
            "role, content, created_at"
        ).eq("session_id", session_id).order("created_at").execute()
        return r.data or []
    except Exception as e:
        logger.error("get_messages failed: %s", e)
        return []
