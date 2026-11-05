"""SEC Company Facts adapter: annual US-GAAP facts with filing provenance.

Only whole-entity USD facts from 10-K/10-K/A filings are supported. Selection
uses actual observation dates, never the filing's fiscal-year label (which
also labels comparative observations). Custom concepts are not inferred.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from app.config import settings


METRIC_TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "assets": ("Assets",),
    "equity": ("StockholdersEquity",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}
INSTANT_METRICS = {"assets", "equity", "cash"}
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}\Z")
_TICKER = re.compile(r"[A-Z][A-Z0-9.-]{0,14}\Z")
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_CACHE_TTL = 3600


def _iso_date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def _arguments(ticker, metrics, limit, as_of):
    if not isinstance(ticker, str) or not _TICKER.fullmatch(ticker.strip().upper()):
        raise ValueError("ticker must be a US stock symbol, e.g. AAPL")
    if metrics is None:
        metrics = list(METRIC_TAGS)
    if (not isinstance(metrics, list) or not metrics
            or any(not isinstance(m, str) or m not in METRIC_TAGS for m in metrics)):
        raise ValueError("metrics must be a non-empty list of supported financial facts")
    if len(set(metrics)) != len(metrics):
        raise ValueError("metrics must not contain duplicates")
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("limit must be an integer from 1 to 10")
    today = datetime.now(timezone.utc).date()
    cutoff = today if as_of is None else _iso_date(as_of)
    if cutoff > today:
        raise ValueError("as_of cannot be in the future")
    return ticker.strip().upper(), metrics, cutoff


def _valid_observations(taxonomy: dict, tag: str, instant: bool, cutoff: date):
    concept = taxonomy.get(tag, {})
    units = concept.get("units", {}) if isinstance(concept, dict) else {}
    values = units.get("USD", []) if isinstance(units, dict) else []
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if not isinstance(value, dict) or value.get("form") not in {"10-K", "10-K/A"}:
            continue
        try:
            end, filed = _iso_date(value.get("end")), _iso_date(value.get("filed"))
            if end > filed or filed > cutoff:
                continue
            amount = value.get("val")
            if type(amount) not in (int, float) or not math.isfinite(amount):
                continue
            if not isinstance(value.get("accn"), str) or not _ACCESSION.fullmatch(value["accn"]):
                continue
            if instant:
                if value.get("start") is not None:
                    continue
            else:
                start = _iso_date(value.get("start"))
                if not 350 <= (end - start).days + 1 <= 380:
                    continue
        except (ValueError, TypeError, OverflowError):
            continue
        result.append(value)
    return result


def _select_fact(metric: str, observations: dict, start: str, end: str, cik: str):
    instant = metric in INSTANT_METRICS
    base = {
        "metric": metric, "value": None, "unit": "USD",
        "period_start": None if instant else start, "period_end": end,
        "period_type": "instant" if instant else "duration",
        "filed": None, "accession": None, "source_tag": None, "source_url": None,
        "status": "missing",
    }
    candidates = [
        (tag, f) for tag in METRIC_TAGS[metric] for f in observations[tag]
        if f["end"] == end and (instant or f["start"] == start)
    ]
    if not candidates:
        base["reason"] = "No eligible USD US-GAAP fact for this annual period and filing cutoff"
        return base

    # Accession numbers identify submitters, not intraday chronology. Conflicts
    # on the latest filing date must remain ambiguous, including amendments.
    latest = max(f["filed"] for _, f in candidates)
    selected = [(tag, f) for tag, f in candidates if f["filed"] == latest]
    tag, fact = selected[0]  # tag priority follows METRIC_TAGS; never add aliases
    accession = fact["accn"]
    base.update({
        "filed": fact["filed"], "accession": accession, "source_tag": f"us-gaap:{tag}",
        "source_url": (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{accession}-index.html"
        ),
    })
    if len({f["val"] for _, f in selected}) > 1:
        base.update(status="ambiguous", reason="Conflicting observations, revenue concepts or filing versions on the latest filing date")
    else:
        base.update(status="ok", value=fact["val"])
    return base


def normalize_company_facts(payload: dict, *, ticker: str, metrics=None, limit=3, as_of=None) -> dict:
    """Select latest-filed annual observations as known on the cutoff date.

    retrieved_at is this normalized snapshot's creation time. The raw SEC
    response can be up to one hour old; it is not a live-update timestamp.
    """
    ticker, metrics, cutoff = _arguments(ticker, metrics, limit, as_of)
    if not isinstance(payload, dict):
        raise ValueError("SEC returned an invalid Company Facts document")
    cik = str(payload.get("cik", ""))
    if not re.fullmatch(r"\d{1,10}", cik) or int(cik) <= 0:
        raise ValueError("SEC Company Facts is missing a valid CIK")
    cik = cik.zfill(10)
    all_facts = payload.get("facts")
    if not isinstance(all_facts, dict):
        raise ValueError("SEC Company Facts is missing facts")
    taxonomy = all_facts.get("us-gaap", {})
    if not isinstance(taxonomy, dict):
        raise ValueError("SEC Company Facts has an invalid US-GAAP taxonomy")
    observations = {
        tag: _valid_observations(taxonomy, tag, metric in INSTANT_METRICS, cutoff)
        for metric, tags in METRIC_TAGS.items() for tag in tags
    }
    # Discover fiscal intervals from duration facts even if only balance-sheet
    # metrics were requested. No quarterly or arbitrary instant dates leak in.
    by_end: dict[str, set[str]] = {}
    for metric, tags in METRIC_TAGS.items():
        if metric in INSTANT_METRICS:
            continue
        for tag in tags:
            for f in observations[tag]:
                by_end.setdefault(f["end"], set()).add(f["start"])
    warnings = []
    periods = []
    for end in sorted(by_end, reverse=True):
        starts = by_end[end]
        if len(starts) != 1:
            warnings.append(f"Skipped {end}: conflicting annual start dates")
            continue
        periods.append({"start": next(iter(starts)), "end": end})
        if len(periods) == limit:
            break
    facts = [
        _select_fact(metric, observations, p["start"], p["end"], cik)
        for p in periods for metric in metrics
    ]
    if not periods:
        warnings.append("No supported annual US-GAAP periods found by the filing cutoff")
    if any(f["status"] != "ok" for f in facts):
        warnings.append("Some facts are missing or ambiguous; do not substitute zero")
    return {
        "ticker": ticker, "cik": cik, "entity_name": payload.get("entityName", ""),
        "period": "annual", "currency": "USD", "as_of": cutoff.isoformat(),
        "version_policy": "latest_filed_as_of",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "periods": periods, "facts": facts, "warnings": warnings,
    }


def _throttle():
    global _LAST_REQUEST
    with _REQUEST_LOCK:
        delay = 0.25 - (time.monotonic() - _LAST_REQUEST)
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST = time.monotonic()


def _request_json(url: str) -> dict:
    identity = settings.edgar_user_agent.strip()
    if (not identity or not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", identity)
            or any(ord(c) < 32 for c in identity)):
        raise ValueError("Set EDGAR_USER_AGENT to your name/project and contact email in .env")
    cache_dir = Path(settings.financial_cache_dir) / "sec"
    cache_path = cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        age = time.time() - cached["fetched_at"]
        if 0 <= age < _CACHE_TTL and isinstance(cached["payload"], dict):
            return cached["payload"]
    except (OSError, ValueError, KeyError, TypeError):
        pass

    # Keep errors free of request headers, including the configured identity.
    with httpx.Client(headers={"User-Agent": identity, "Accept": "application/json"}, timeout=30) as client:
        for attempt in range(3):
            _throttle()
            try:
                response = client.get(url)
            except httpx.RequestError:
                if attempt == 2:
                    raise RuntimeError("SEC request failed after 3 attempts; retry later") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise RuntimeError(f"SEC temporarily unavailable (HTTP {response.status_code}); retry later")
                try:
                    delay = float(response.headers.get("Retry-After", 2 ** attempt))
                    delay = max(0, delay) if math.isfinite(delay) else 2 ** attempt
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(response.headers["Retry-After"])
                        if retry_at.tzinfo is None:
                            retry_at = retry_at.replace(tzinfo=timezone.utc)
                        delay = max(0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError, KeyError, OverflowError):
                        delay = 2 ** attempt
                if delay > 5:
                    # Respect SEC's wait request without blocking the tool loop.
                    raise RuntimeError("SEC requested a longer retry delay; retry later")
                time.sleep(delay)
                continue
            if response.status_code == 403:
                raise RuntimeError("SEC denied the request (HTTP 403); check EDGAR_USER_AGENT or retry later")
            if response.status_code != 200:
                raise RuntimeError(f"SEC request failed (HTTP {response.status_code})")
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError("SEC returned invalid JSON") from None
            if not isinstance(payload, dict):
                raise RuntimeError("SEC returned an invalid JSON document")
            break

    cache_dir.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as f:
            name = f.name
            json.dump({"fetched_at": time.time(), "payload": payload}, f, allow_nan=False)
        os.replace(name, cache_path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
    return payload


def get_financial_dataset(ticker: str, metrics=None, period="annual", limit=3, as_of=None) -> dict:
    """Fetch official ticker mapping and Company Facts, cached for one hour."""
    ticker, metrics, cutoff = _arguments(ticker, metrics, limit, as_of)
    if period != "annual":
        raise ValueError("Only period='annual' is supported; quarterly and TTM are not inferred")
    tickers = _request_json("https://www.sec.gov/files/company_tickers.json")
    matches = {
        str(row["cik_str"]).zfill(10)
        for row in tickers.values() if isinstance(row, dict)
        and row.get("ticker") == ticker and "cik_str" in row
    }
    if not matches:
        raise ValueError(f"Ticker {ticker} was not found in the SEC company list")
    if len(matches) != 1:
        raise ValueError(f"Ticker {ticker} has an ambiguous SEC mapping")
    cik = matches.pop()
    if not re.fullmatch(r"\d{10}", cik) or int(cik) <= 0:
        raise ValueError("SEC returned an invalid ticker/CIK mapping")
    payload = _request_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    result = normalize_company_facts(payload, ticker=ticker, metrics=metrics, limit=limit, as_of=cutoff.isoformat())
    if result["cik"] != cik:
        raise ValueError("SEC Company Facts CIK does not match the requested company")
    return result
