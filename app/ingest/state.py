"""Incremental ingestion state for finance data sources.

Persists per-source `last_success_at` to `data/finance_source_state.json`.
A fetch run only updates state if it succeeded — failed runs retry the same
window next time, so nothing is missed or double-fetched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings


def _state_path() -> Path:
    return Path(settings.finance_state_path)


def load_state() -> dict[str, str]:
    p = _state_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, str]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def get_since(source_name: str, fallback: datetime) -> datetime:
    """Return the last_success_at for `source_name`, or `fallback` if unknown."""
    raw = load_state().get(source_name)
    if not raw:
        return fallback
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return fallback


def update_source(source_name: str, ts: Optional[datetime] = None) -> None:
    state = load_state()
    when = ts or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    state[source_name] = when.isoformat()
    save_state(state)
