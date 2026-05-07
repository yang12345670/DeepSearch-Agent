"""PDF generation tool for the ReAct agent.

Renders structured (title + sections) input into a PDF file under
`data/exports/{timestamp}_{slug}.pdf` using reportlab + the built-in
CID font STSong-Light (no external ttf required, works on Win/Mac/Linux).

Tool input shape (LLM's responsibility to fill correctly):
    {
        "title": "AAPL Q1 风险概览",
        "sections": [
            {
                "heading": "主要风险因素",
                "body": "Apple 在 10-K 中披露的风险包括 ...",
                "citations": ["AAPL 10-K 2024 Item 1A", "..."]
            },
            ...
        ],
        "filename": "aapl_q1_risk"   # optional
    }

Tool output:
    {"file_path": "data/exports/20260506_180000_aapl_q1_risk.pdf",
     "size_bytes": 23456, "sections_count": 4}
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.agent.tools import register_tool

logger = logging.getLogger(__name__)

# Register the built-in CJK CID font once on module import.
# STSong-Light is bundled with reportlab via Adobe-GB1 — no ttf download needed.
_FONT_NAME = "STSong-Light"
try:
    pdfmetrics.registerFont(UnicodeCIDFont(_FONT_NAME))
except Exception as e:  # noqa: BLE001
    logger.warning("Failed to register %s, falling back to Helvetica: %s", _FONT_NAME, e)
    _FONT_NAME = "Helvetica"


_EXPORTS_DIR = Path("data/exports")


def _slugify(s: str) -> str:
    """ASCII-safe slug. Non-ASCII chars (Chinese etc.) collapse to '_'."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    return (s[:50] or "report").lower()


def _styles() -> Dict[str, ParagraphStyle]:
    """Build the four paragraph styles we use, all CJK-safe."""
    base = getSampleStyleSheet()["Normal"]
    return {
        "title": ParagraphStyle(
            "TitleCJK", parent=base,
            fontName=_FONT_NAME, fontSize=20, leading=26,
            spaceAfter=14, textColor=colors.HexColor("#1a1a1a"),
        ),
        "h1": ParagraphStyle(
            "H1CJK", parent=base,
            fontName=_FONT_NAME, fontSize=14, leading=18,
            spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#0a3d8a"),
        ),
        "body": ParagraphStyle(
            "BodyCJK", parent=base,
            fontName=_FONT_NAME, fontSize=11, leading=16, spaceAfter=4,
        ),
        "citation": ParagraphStyle(
            "CitationCJK", parent=base,
            fontName=_FONT_NAME, fontSize=9, leading=12,
            textColor=colors.HexColor("#666666"), leftIndent=10,
        ),
    }


def _escape_xml(s: str) -> str:
    """reportlab Paragraph parses HTML-ish tags — escape to plain text."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _build_story(
    title: str,
    sections: List[Dict[str, Any]],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    story: List[Any] = []
    story.append(Paragraph(_escape_xml(title), styles["title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["citation"],
    ))
    story.append(Spacer(1, 12))

    for idx, sec in enumerate(sections, start=1):
        heading = str(sec.get("heading", f"Section {idx}"))
        body = str(sec.get("body", ""))
        citations = sec.get("citations") or []

        story.append(Paragraph(f"{idx}. {_escape_xml(heading)}", styles["h1"]))

        # Render each non-empty paragraph in body separately for nicer flow
        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            # Convert single newlines to <br/> so manual line breaks survive
            html_safe = _escape_xml(para).replace("\n", "<br/>")
            story.append(Paragraph(html_safe, styles["body"]))

        if citations:
            story.append(Spacer(1, 4))
            for i, c in enumerate(citations, start=1):
                story.append(Paragraph(
                    f"[{i}] {_escape_xml(str(c))}",
                    styles["citation"],
                ))

        story.append(Spacer(1, 10))

    return story


@register_tool(
    name="generate_pdf",
    description=(
        "Generate a PDF report from a structured title + sections payload. "
        "Use this when the user asks for a downloadable PDF, report, or document. "
        "Each section has a heading (required), body (required, plain text or "
        "simple paragraphs separated by blank lines), and optional citations "
        "(list of source labels). Returns the saved file path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Main title shown on the first page.",
            },
            "sections": {
                "type": "array",
                "minItems": 1,
                "description": "Ordered list of report sections.",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "body": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source labels, e.g. 'AAPL 10-K 2024 Item 1A'",
                        },
                    },
                    "required": ["heading", "body"],
                },
            },
            "filename": {
                "type": "string",
                "description": "Optional custom file stem (no .pdf suffix). "
                               "If omitted, a slug is derived from the title.",
            },
        },
        "required": ["title", "sections"],
    },
)
def generate_pdf(
    title: str,
    sections: List[Dict[str, Any]],
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    if not sections:
        raise ValueError("sections must be a non-empty list")

    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _slugify(filename) if filename else _slugify(title)
    out_path = _EXPORTS_DIR / f"{timestamp}_{stem}.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=title,
    )
    story = _build_story(title, sections, _styles())
    doc.build(story)

    size = out_path.stat().st_size
    logger.info(
        "generate_pdf wrote %s (%d bytes, %d sections)",
        out_path, size, len(sections),
    )
    return {
        "file_path": str(out_path).replace("\\", "/"),
        "size_bytes": size,
        "sections_count": len(sections),
    }
