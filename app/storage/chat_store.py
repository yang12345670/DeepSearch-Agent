# -*- coding: utf-8 -*-
"""Chat history persistence via local SQLite.

Stores sessions and messages in `data/chat.db`. No external service required.
Public API is identical to the previous Supabase-backed implementation, so
nothing in routes.py / orchestrator.py needs to change.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path("data") / "chat.db"
_init_lock = threading.Lock()
_initialized = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    """Open a new connection. SQLite handles its own locking for short writes."""
    _ensure_schema()
    conn = sqlite3.connect(str(_DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_schema() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), timeout=5.0)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id  TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New Chat',
                    preview     TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    last_active TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON chat_messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_sessions_last_active
                    ON chat_sessions(last_active DESC);
                """
            )
            conn.commit()
        finally:
            conn.close()
        _initialized = True


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------

def create_session(session_id: str, title: str = "New Chat") -> Optional[Dict]:
    try:
        conn = _connect()
        try:
            existing = conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing:
                return dict(existing)
            now = _now()
            conn.execute(
                "INSERT INTO chat_sessions (session_id, title, preview, created_at, last_active) "
                "VALUES (?, ?, '', ?, ?)",
                (session_id, title, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("create_session failed: %s", e)
        return None


def list_sessions() -> List[Dict]:
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY last_active DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.error("list_sessions failed: %s", e)
        return []


def update_session(session_id: str, **fields) -> None:
    if not fields:
        return
    try:
        conn = _connect()
        try:
            fields["last_active"] = _now()
            cols = ", ".join(f"{k} = ?" for k in fields)
            params = list(fields.values()) + [session_id]
            conn.execute(
                f"UPDATE chat_sessions SET {cols} WHERE session_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.error("update_session failed: %s", e)


def delete_session(session_id: str) -> bool:
    try:
        conn = _connect()
        try:
            conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.error("delete_session failed: %s", e)
        return False


# ------------------------------------------------------------------
# Messages
# ------------------------------------------------------------------

def save_message(session_id: str, role: str, content: str) -> None:
    try:
        conn = _connect()
        try:
            now = _now()
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            updates: Dict[str, Any] = {"last_active": now}
            if role == "user":
                updates["preview"] = content[:60]
            cols = ", ".join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [session_id]
            conn.execute(
                f"UPDATE chat_sessions SET {cols} WHERE session_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.error("save_message failed: %s", e)


def get_messages(session_id: str) -> List[Dict]:
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT role, content, created_at FROM chat_messages "
                "WHERE session_id = ? ORDER BY created_at ASC, id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.error("get_messages failed: %s", e)
        return []
