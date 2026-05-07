"""CLI: fetch Alpha Vantage NEWS_SENTIMENT into data/docs_finance/alphavantage/.

Usage:
    python scripts/fetch_alphavantage.py                       # since last_success or 7 days back
    python scripts/fetch_alphavantage.py --backfill-days 30
    python scripts/fetch_alphavantage.py --tickers AAPL,MSFT
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.ingest import state
from app.ingest.av_adapter import fetch_news


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Alpha Vantage NEWS_SENTIMENT into data/docs_finance/alphavantage/"
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=7,
        help="If no prior state, fetch this many days back (default: 7)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (default: settings.finance_tracked_tickers)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Per-ticker max items (default: 200)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    settings.ensure_finance_dirs()

    started = datetime.now(timezone.utc)
    fallback = started - timedelta(days=args.backfill_days)
    since = state.get_since("alphavantage", fallback)

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else None
    )

    print(f"[av] Fetching since {since.isoformat()}")
    written = fetch_news(since, tickers=tickers, limit_per_ticker=args.limit)
    print(f"[av] Wrote {written} new articles")

    state.update_source("alphavantage", started)
    print(f"[av] Updated state.last_success_at -> {started.isoformat()}")


if __name__ == "__main__":
    main()
