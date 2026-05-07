"""End-to-end finance ingestion: fetch EDGAR + AV, then rebuild the index.

Pipeline:
  1) fetch SEC EDGAR filings  -> data/docs_finance/edgar/
  2) fetch Alpha Vantage news -> data/docs_finance/alphavantage/
  3) re-run ingest_docs with DOCS_DIR=data/docs_finance to rebuild
     data/index/{chunks.json, faiss.index}

The original Datawhale corpus at data/docs/ is NOT touched. To roll back:
    python scripts/ingest_docs.py
will rebuild the old Datawhale-based index.

Usage:
    python scripts/ingest_finance.py
    python scripts/ingest_finance.py --sources edgar
    python scripts/ingest_finance.py --skip-index   # fetch only, no index rebuild
    python scripts/ingest_finance.py --edgar-backfill-days 90 --av-backfill-days 14
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.ingest import state
from app.ingest.av_adapter import fetch_news
from app.ingest.edgar_adapter import fetch_filings


def run_fetch_edgar(backfill_days: int) -> int:
    started = datetime.now(timezone.utc)
    fallback = started - timedelta(days=backfill_days)
    since = state.get_since("sec_edgar", fallback)
    print(f"[edgar] fetching since {since.isoformat()}")
    n = fetch_filings(since)
    print(f"[edgar] {n} new filings")
    state.update_source("sec_edgar", started)
    return n


def run_fetch_av(backfill_days: int) -> int:
    started = datetime.now(timezone.utc)
    fallback = started - timedelta(days=backfill_days)
    since = state.get_since("alphavantage", fallback)
    print(f"[av] fetching since {since.isoformat()}")
    n = fetch_news(since)
    print(f"[av] {n} new articles")
    state.update_source("alphavantage", started)
    return n


def run_index_rebuild() -> None:
    """Re-run ingest_docs.py against the finance corpus (DOCS_DIR override)."""
    env = os.environ.copy()
    env["DOCS_DIR"] = settings.finance_docs_dir
    cmd = [sys.executable, str(ROOT / "scripts" / "ingest_docs.py")]
    print(f"[index] DOCS_DIR={env['DOCS_DIR']} -> rebuilding index")
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"ingest_docs.py exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch finance sources + rebuild index in one shot"
    )
    parser.add_argument(
        "--sources",
        default="edgar,av",
        help="Comma-separated: edgar,av (default: both)",
    )
    parser.add_argument("--edgar-backfill-days", type=int, default=30)
    parser.add_argument("--av-backfill-days", type=int, default=7)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only fetch, do not rebuild FAISS index",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    settings.ensure_finance_dirs()

    sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    if "edgar" in sources:
        try:
            run_fetch_edgar(args.edgar_backfill_days)
        except Exception as e:  # noqa: BLE001
            print(f"[edgar] FAILED (state NOT updated): {e}")

    if "av" in sources:
        try:
            run_fetch_av(args.av_backfill_days)
        except Exception as e:  # noqa: BLE001
            print(f"[av] FAILED (state NOT updated): {e}")

    if not args.skip_index:
        run_index_rebuild()
        print("[index] done")


if __name__ == "__main__":
    main()
