"""Shared API schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for POST /chat."""

    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id for short-term & long-term memory.",
    )


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    answer: str

