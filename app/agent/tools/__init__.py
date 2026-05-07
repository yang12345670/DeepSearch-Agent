"""Tool abstraction + global registry for the ReAct agent layer.

Usage:
    @register_tool(
        name="generate_pdf",
        description="Generate a PDF report from structured sections.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "sections": {"type": "array", "items": {...}},
            },
            "required": ["title", "sections"],
        },
    )
    def generate_pdf(title: str, sections: list) -> dict:
        ...

The registered tool is exposed to the LLM via `get_openai_tools_schema()`
and dispatched at runtime via `get_tool(name).execute(**args)`.

Tool functions should:
  - Return a JSON-serializable dict (LLM sees this as tool_response)
  - Raise normal exceptions on failure — `execute()` wraps them as
    {"error": "..."} so the LLM can decide to retry or abort
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema (OpenAI tools format)
    func: Callable[..., Dict[str, Any]]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the tool, capturing exceptions as {"error": ...}."""
        try:
            result = self.func(**kwargs)
            if not isinstance(result, dict):
                # Coerce non-dict returns so LLM always gets structured response
                result = {"result": result}
            return result
        except TypeError as e:
            # Most commonly: bad arguments from LLM (missing/extra keys)
            logger.warning("Tool %s argument error: %s (kwargs=%s)", self.name, e, kwargs)
            return {"error": f"Invalid arguments for {self.name}: {e}"}
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool %s execution failed", self.name)
            return {"error": f"{type(e).__name__}: {e}"}

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI Chat Completions `tools` parameter format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOLS_REGISTRY: Dict[str, Tool] = {}


def register_tool(
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register the wrapped function as a tool in the global registry."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in TOOLS_REGISTRY:
            logger.warning("Tool %s already registered, overwriting", name)
        TOOLS_REGISTRY[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            func=fn,
        )
        logger.debug("Registered tool: %s", name)
        return fn

    return decorator


def get_tool(name: str) -> Optional[Tool]:
    return TOOLS_REGISTRY.get(name)


def get_all_tools() -> List[Tool]:
    return list(TOOLS_REGISTRY.values())


def get_openai_tools_schema() -> List[Dict[str, Any]]:
    """Return the full registry as OpenAI `tools` parameter (list of dicts)."""
    return [t.to_openai_schema() for t in TOOLS_REGISTRY.values()]


def load_builtin_tools() -> None:
    """Import all built-in tool modules so their @register_tool decorators run.

    Call this once at application startup. Safe to call multiple times.
    """
    # Lazy imports to avoid circular dependencies
    from app.agent.tools import pdf_export  # noqa: F401
    from app.agent.tools import search_kb  # noqa: F401

    logger.info("Loaded built-in tools: %s", list(TOOLS_REGISTRY.keys()))
