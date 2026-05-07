"""SEC EDGAR adapter: fetch filings and persist as per-filing markdown.

Tier 1 scope:
- 10-K: Item 1A (Risk Factors) + Item 7 (MD&A) — via obj.get_item_with_part()
- 10-Q: Part I Item 2 (MD&A)                  — via obj.get_item_with_part()
- 8-K:  per-item view() output captured to string

Output: `data/docs_finance/edgar/{TICKER}/{form}_{date}_{accession}.md`

Idempotency: existing accession files are skipped.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Filename safe-chars
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9.-]+")

# Per-form Item extraction plan: (part, item, heading)
# edgartools `obj.get_item_with_part(part, item)` returns the full string
# of that section (tens of thousands of chars for 10-K Item 1A / Item 7).
_TENK_ITEMS: List[Tuple[str, str, str]] = [
    ("I", "1A", "Item 1A. Risk Factors"),
    ("II", "7", "Item 7. MD&A"),
]
_TENQ_ITEMS: List[Tuple[str, str, str]] = [
    ("I", "2", "Part I Item 2. MD&A"),
]


def _safe_attr(obj, *names, default=None):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _extract_structured_items(
    obj,
    plan: List[Tuple[str, str, str]],
) -> str:
    """Pull `(part, item)` sections from a TenK/TenQ object via get_item_with_part.

    Returns concatenated markdown with section headings, or "" if nothing
    substantial was extracted.
    """
    parts: List[str] = []
    for part, item, heading in plan:
        try:
            section = obj.get_item_with_part(part, item)
        except Exception as e:  # noqa: BLE001
            logger.warning("get_item_with_part(%s, %s) failed: %s", part, item, e)
            continue
        if not isinstance(section, str):
            continue
        section = section.strip()
        if len(section) < 200:
            logger.debug("Item %s/%s too short (%d chars), skipping", part, item, len(section))
            continue
        parts.append(f"## {heading}\n\n{section}")
    return "\n\n".join(parts)


def _extract_8k_items(obj) -> str:
    """For CurrentReport (8-K): capture each item's `view()` output.

    edgartools' view() prints item bodies to stdout; we redirect to capture
    them as a string. This avoids parsing the full filing markdown which is
    cluttered with SEC cover-page boilerplate.
    """
    items = getattr(obj, "items", None)
    if not items:
        return ""

    parts: List[str] = []
    for item in items:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                obj.view(item)
        except Exception as e:  # noqa: BLE001
            logger.debug("8-K view(%s) failed: %s", item, e)
            continue
        text = buf.getvalue().strip()
        if not text or len(text) < 100:
            continue
        # First line is usually "Item X.XX <Title>" — keep it as the section header
        parts.append(text)
    return "\n\n".join(parts)


def _filing_full_markdown(filing) -> str:
    """Last-resort fallback when obj-based extraction yields nothing."""
    for method_name in ("markdown", "text", "full_text"):
        method = getattr(filing, method_name, None)
        if callable(method):
            try:
                t = method()
                if t and isinstance(t, str) and len(t) > 500:
                    return t
            except Exception as e:  # noqa: BLE001
                logger.debug("filing.%s() failed: %s", method_name, e)
    return ""


def _extract_filing_text(filing, form: str) -> str:
    """Form-aware structured extraction, with markdown fallback."""
    try:
        obj = filing.obj()
    except Exception as e:  # noqa: BLE001
        logger.warning("filing.obj() failed for %s: %s, falling back to markdown", form, e)
        obj = None

    if obj is not None:
        if form == "10-K":
            text = _extract_structured_items(obj, _TENK_ITEMS)
            if text:
                return text
        elif form == "10-Q":
            text = _extract_structured_items(obj, _TENQ_ITEMS)
            if text:
                return text
        elif form == "8-K":
            text = _extract_8k_items(obj)
            if text:
                return text

    # Fallback: full markdown (still better than nothing for unknown forms)
    fallback = _filing_full_markdown(filing)
    if fallback and form == "8-K":
        # Strip cover-page boilerplate that precedes the first "Item X.XX"
        m = re.search(r"item\s*\d+\.\d+\b", fallback, re.IGNORECASE)
        if m:
            return fallback[m.start():]
    return fallback


def _clean(text: str) -> str:
    # Remove thousands separators in numbers ($1,234 -> $1234) so BM25 doesn't split
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_filings(
    since: datetime,
    tickers: Optional[Iterable[str]] = None,
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    out_dir: Optional[Path] = None,
) -> int:
    """Fetch filings since `since`, write per-filing markdown to disk.

    Returns count of newly written files (existing files are skipped).
    """
    from app.config import settings

    if not settings.edgar_user_agent:
        raise RuntimeError(
            "EDGAR_USER_AGENT not set in .env — SEC requires real email in User-Agent header"
        )

    # Import edgartools lazily so config-only flows don't require it
    from edgar import Company, set_identity

    set_identity(settings.edgar_user_agent)

    base = out_dir or (Path(settings.finance_docs_dir) / "edgar")
    base.mkdir(parents=True, exist_ok=True)

    tickers_list = list(tickers or settings.finance_tracked_tickers)
    since_date = since.date() if isinstance(since, datetime) else since

    written = 0
    for ticker in tickers_list:
        try:
            company = Company(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("Company(%s) failed: %s", ticker, e)
            continue

        for form in forms:
            try:
                filings = company.get_filings(form=form)
            except Exception as e:  # noqa: BLE001
                logger.warning("get_filings(%s, form=%s) failed: %s", ticker, form, e)
                continue

            # edgartools returns filings sorted by date desc — early-break once we cross the cutoff
            for filing in filings:
                fdate = _safe_attr(filing, "filing_date")
                if fdate is None:
                    continue
                if fdate < since_date:
                    break

                accession = _safe_attr(
                    filing, "accession_no", "accession_number", default=""
                )
                if not accession:
                    continue

                safe_acc = _UNSAFE_CHARS.sub("_", str(accession))
                out_file = base / ticker / f"{form}_{fdate}_{safe_acc}.md"
                if out_file.exists():
                    continue

                section_text = _extract_filing_text(filing, form)
                if not section_text or len(section_text) < 200:
                    logger.debug("Skip %s %s %s: section too short (%d)",
                                 ticker, form, accession, len(section_text or ""))
                    continue

                cleaned = _clean(section_text)
                url = _safe_attr(
                    filing, "filing_url", "homepage_url", "url", default=""
                )

                header = (
                    f"# {ticker} {form} {fdate}\n\n"
                    f"- **Ticker**: {ticker}\n"
                    f"- **Form**: {form}\n"
                    f"- **Filing date**: {fdate}\n"
                    f"- **Accession**: {accession}\n"
                    f"- **URL**: {url}\n\n"
                    f"---\n\n"
                )

                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(header + cleaned, encoding="utf-8")
                written += 1
                logger.info("EDGAR wrote %s (%d chars)", out_file.name, len(cleaned))

    return written
