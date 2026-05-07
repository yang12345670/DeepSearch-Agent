"""CLI: fetch SEC EDGAR filings into data/docs_finance/edgar/.

Usage:
    python scripts/fetch_edgar.py                       # since last_success or 30 days back
    python scripts/fetch_edgar.py --backfill-days 90    # custom backfill window
    python scripts/fetch_edgar.py --tickers AAPL,MSFT   # subset of tickers
    python scripts/fetch_edgar.py --forms 10-K          # subset of forms
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
from app.ingest.edgar_adapter import fetch_filings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR filings into data/docs_finance/edgar/"
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=30,
        help="If no prior state, fetch this many days back (default: 30)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (default: settings.finance_tracked_tickers)",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default="10-K,10-Q,8-K",
        help="Comma-separated forms (default: 10-K,10-Q,8-K)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    settings.ensure_finance_dirs()

    started = datetime.now(timezone.utc)
    fallback = started - timedelta(days=args.backfill_days)
    since = state.get_since("sec_edgar", fallback)

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else None
    )
    forms = tuple(f.strip() for f in args.forms.split(",") if f.strip())

    print(f"[edgar] Fetching since {since.isoformat()} for forms {forms}")
    written = fetch_filings(since, tickers=tickers, forms=forms)
    print(f"[edgar] Wrote {written} new filings")

    state.update_source("sec_edgar", started)
    print(f"[edgar] Updated state.last_success_at -> {started.isoformat()}")


if __name__ == "__main__":
    main()
