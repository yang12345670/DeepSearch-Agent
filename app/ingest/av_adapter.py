"""Alpha Vantage NEWS_SENTIMENT adapter.

Tier 1 scope: per-ticker news, save title + summary as markdown.

Output: `data/docs_finance/alphavantage/{YYYY-MM}/{sha1(url+title)[:16]}.md`

Idempotency: existing files are skipped (sha1-based dedup).
Rate limiting and request limits are configured centrally in app.config.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

def _doc_id(url: str, title: str) -> str:
    payload = f"{url}|{title}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def _parse_published(s: str) -> datetime:
    # AV format: 20240115T093000
    return datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _pick_primary_ticker(item: dict, queried_ticker: str) -> str:
    """Highest-relevance ticker from ticker_sentiment[]; fallback to queried."""
    sentiments = item.get("ticker_sentiment") or []
    if not sentiments:
        return queried_ticker
    try:
        best = max(
            sentiments,
            key=lambda s: float(s.get("relevance_score", 0) or 0),
        )
        return best.get("ticker", queried_ticker)
    except (ValueError, TypeError):
        return queried_ticker


def _is_rate_limit_response(payload: dict) -> bool:
    if "Note" in payload:
        return True
    info = payload.get("Information", "")
    if isinstance(info, str) and ("rate limit" in info.lower() or "thank you" in info.lower()):
        return True
    return False


def fetch_news(
    since: datetime,
    tickers: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
    limit_per_ticker: Optional[int] = None,
) -> int:
    """Fetch news per-ticker since `since`, write per-article markdown.

    Returns count of newly written files. Existing files are skipped.
    """
    from app.config import settings

    if limit_per_ticker is None:
        limit_per_ticker = settings.alphavantage_limit_per_ticker

    if not settings.alphavantage_api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in .env")

    base = out_dir or (Path(settings.finance_docs_dir) / "alphavantage")
    base.mkdir(parents=True, exist_ok=True)

    tickers_list = list(tickers or settings.finance_tracked_tickers)
    time_from = since.strftime("%Y%m%dT%H%M")

    written = 0
    seen_doc_ids: set[str] = set()

    with httpx.Client(timeout=settings.alphavantage_request_timeout_seconds) as client:
        for i, ticker in enumerate(tickers_list):
            if i > 0:
                logger.info("[av] sleep %.0fs (rate limit)", settings.alphavantage_request_interval_seconds)
                time.sleep(settings.alphavantage_request_interval_seconds)

            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_from": time_from,
                "limit": limit_per_ticker,
                "sort": "LATEST",
                "apikey": settings.alphavantage_api_key,
            }
            try:
                r = client.get(settings.alphavantage_api_url, params=params)
                r.raise_for_status()
                payload = r.json()
            except (httpx.HTTPError, ValueError) as e:  # noqa: BLE001
                # HTTP exception text can contain the API key in the query URL.
                logger.warning("[av] %s request failed (%s)", ticker, type(e).__name__)
                continue

            if _is_rate_limit_response(payload):
                msg = payload.get("Note") or payload.get("Information") or "?"
                logger.warning("[av] %s rate-limited / quota: %s", ticker, msg)
                continue

            feed = payload.get("feed", [])
            if not isinstance(feed, list):
                logger.warning("[av] %s: unexpected payload (no feed)", ticker)
                continue

            logger.info("[av] %s: %d items", ticker, len(feed))

            for item in feed:
                title = (item.get("title") or "").strip()
                summary = (item.get("summary") or "").strip()
                url = (item.get("url") or "").strip()
                if not title or not summary or not url:
                    continue

                doc_id = _doc_id(url, title)
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)

                published_raw = item.get("time_published") or ""
                try:
                    published = _parse_published(published_raw)
                except ValueError:
                    published = datetime.now(timezone.utc)

                ym_dir = base / published.strftime("%Y-%m")
                ym_dir.mkdir(parents=True, exist_ok=True)
                out_file = ym_dir / f"{doc_id}.md"
                if out_file.exists():
                    continue

                primary_ticker = _pick_primary_ticker(item, ticker)
                source_name = item.get("source", "")
                sentiment_label = item.get("overall_sentiment_label", "")
                sentiment_score = item.get("overall_sentiment_score", "")

                header = (
                    f"# {title}\n\n"
                    f"- **Ticker**: {primary_ticker}\n"
                    f"- **Source**: {source_name}\n"
                    f"- **Published**: {published.isoformat()}\n"
                    f"- **URL**: {url}\n"
                    f"- **Sentiment**: {sentiment_label} ({sentiment_score})\n\n"
                    f"---\n\n"
                )
                out_file.write_text(header + summary, encoding="utf-8")
                written += 1

    return written
