"""Small helper utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple


def strip_html_tags(text: str) -> str:
    """Remove HTML/XML tags and collapse excessive whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)        # strip tags
    text = re.sub(r"&[a-zA-Z]+;", " ", text)    # strip HTML entities like &nbsp;
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text.strip()


def load_text_and_md_files(directory: str) -> List[Tuple[str, str]]:
    """Load all .txt/.md files recursively. Return (rel_path, content).

    HTML tags are stripped from content to improve chunking and retrieval quality.
    """
    root = Path(directory)
    if not root.exists():
        return []
    out: List[Tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".txt", ".md"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        text = strip_html_tags(text)
        out.append((str(p.relative_to(root)), text))
    return out
