"""Deterministic percentages calculated only from stored financial evidence."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal, DecimalException, localcontext
import math
import re

from app.agent.tools import register_tool
from app.finance.store import load_dataset


_METRICS = {
    "revenue_growth_yoy": ("revenue", "revenue", "(current_revenue - previous_revenue) / previous_revenue * 100"),
    "net_income_growth_yoy": ("net_income", "net_income", "(current_net_income - previous_net_income) / previous_net_income * 100"),
    "net_margin": ("net_income", "revenue", "net_income / revenue * 100"),
    "operating_cash_flow_margin": ("operating_cash_flow", "revenue", "operating_cash_flow / revenue * 100"),
}
_DEFAULT_METRICS = ["revenue_growth_yoy", "net_income_growth_yoy", "net_margin"]
_FACT_METRICS = {"revenue", "net_income", "assets", "equity", "cash", "operating_cash_flow"}
_FACT_FIELDS = {
    "metric", "value", "unit", "period_start", "period_end", "period_type", "filed",
    "accession", "source_tag", "source_url", "status",
}


def _date(value: str, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD")
    return parsed


def _number(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Fact values must be finite JSON numbers")
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Fact values must be finite JSON numbers")
    return number


def _validate_dataset(dataset: dict) -> None:
    """Reject malformed snapshots before selecting any calculation operands."""
    ticker = dataset.get("ticker")
    cik = dataset.get("cik")
    if not isinstance(ticker, str) or re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker) is None:
        raise ValueError("Snapshot has an invalid ticker")
    if not isinstance(cik, str) or re.fullmatch(r"[0-9]{1,10}", cik) is None:
        raise ValueError("Snapshot has an invalid CIK")
    if not isinstance(dataset.get("entity_name"), str) or not dataset["entity_name"].strip():
        raise ValueError("Snapshot must identify its company")
    if dataset.get("period") != "annual" or dataset.get("currency") != "USD":
        raise ValueError("Only annual USD financial snapshots are supported")
    if dataset.get("version_policy") != "latest_filed_as_of":
        raise ValueError("Unsupported snapshot version policy")
    _date(dataset.get("as_of"), "as_of")
    try:
        datetime.fromisoformat(dataset["retrieved_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Snapshot must have a valid retrieved_at timestamp") from exc
    periods, facts = dataset.get("periods"), dataset.get("facts")
    if not isinstance(periods, list) or not isinstance(facts, list):
        raise ValueError("Snapshot periods and facts must be arrays")
    prior_start = None
    ends = set()
    for period in periods:
        if not isinstance(period, dict):
            raise ValueError("Each annual period must be an object")
        start, end = _date(period.get("start"), "period start"), _date(period.get("end"), "period end")
        if not 350 <= (end - start).days + 1 <= 380:
            raise ValueError("Snapshot contains a non-annual reporting period")
        if end in ends or (prior_start is not None and end >= prior_start):
            raise ValueError("Annual periods must be unique, non-overlapping and newest first")
        ends.add(end)
        prior_start = start
    for fact in facts:
        if not isinstance(fact, dict) or not _FACT_FIELDS.issubset(fact):
            raise ValueError("Snapshot contains an incomplete financial fact")
        if (not isinstance(fact["metric"], str) or fact["metric"] not in _FACT_METRICS
                or not isinstance(fact["status"], str)
                or fact["status"] not in {"ok", "missing", "ambiguous"}):
            raise ValueError("Snapshot contains an unsupported fact metric or status")
        if fact.get("ticker", ticker) != ticker or str(fact.get("cik", cik)).zfill(10) != cik.zfill(10):
            raise ValueError("Fact company does not match the snapshot company")
        if _date(fact["period_end"], "fact period_end") not in ends:
            raise ValueError("Fact period does not belong to the snapshot")
        if fact["period_start"] is not None:
            _date(fact["period_start"], "fact period_start")
        if not isinstance(fact["period_type"], str) or fact["period_type"] not in {"duration", "instant"}:
            raise ValueError("Fact period_type must be duration or instant")
        if fact["status"] != "ok":
            if fact["value"] is not None:
                raise ValueError("Missing or ambiguous facts must have null values")
            continue
        _number(fact["value"])
        _date(fact["filed"], "fact filed")
        if not isinstance(fact["unit"], str) or not fact["unit"]:
            raise ValueError("An available fact must declare its unit")
        if not isinstance(fact["accession"], str) or re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", fact["accession"]) is None:
            raise ValueError("An available fact must have a valid accession")
        if not isinstance(fact["source_tag"], str) or not fact["source_tag"].startswith("us-gaap:"):
            raise ValueError("An available fact must identify its US-GAAP source tag")
        accession = fact["accession"]
        expected_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{accession}-index.html"
        )
        if fact["source_url"] != expected_url:
            raise ValueError("Fact source URL does not match the snapshot issuer and filing accession")


def _select(dataset: dict, metric: str, period: dict) -> tuple[list[dict], str | None]:
    candidates = [f for f in dataset["facts"] if f["metric"] == metric and f["period_end"] == period["end"]]
    if len(candidates) != 1:
        return candidates, f"{metric}: {'no fact' if not candidates else 'multiple facts'} for {period['end']}"
    fact = candidates[0]
    if fact["status"] != "ok":
        return candidates, f"{metric}: {fact['status']} ({fact.get('reason') or 'not usable'})"
    if fact["period_type"] != "duration" or fact["period_start"] != period["start"]:
        return candidates, f"{metric}: reporting period does not match the requested annual interval"
    if fact["unit"] != dataset["currency"]:
        return candidates, f"{metric}: unit does not match snapshot currency"
    if not period["end"] <= fact["filed"] <= dataset["as_of"]:
        return candidates, f"{metric}: filing date is outside the period/as_of boundary"
    return candidates, None


def _calculate(dataset: dict, metric: str, index: int) -> dict:
    period = dataset["periods"][index]
    numerator, denominator, formula = _METRICS[metric]
    result = {
        "metric": metric, "period_start": period["start"], "period_end": period["end"],
        "value": None, "unit": "percent", "status": "unavailable", "formula": formula,
        "inputs": [],
    }
    growth = metric.endswith("_growth_yoy")
    left, left_error = _select(dataset, numerator, period)
    result["inputs"] = left
    denominator_period = period
    if growth:
        if index + 1 >= len(dataset["periods"]):
            result["reason"] = "Previous annual period is not present in the snapshot"
            return result
        denominator_period = dataset["periods"][index + 1]
    right, right_error = _select(dataset, denominator, denominator_period)
    result["inputs"] = left + right
    if growth:
        previous_end = _date(denominator_period["end"], "previous period end")
        if _date(period["start"], "current period start") != previous_end + timedelta(days=1):
            result["reason"] = "Annual periods are not consecutive; year-over-year growth is unavailable"
            return result
    if left_error or right_error:
        result["reason"] = "; ".join(e for e in (left_error, right_error) if e)
        return result
    a, b = left[0], right[0]
    if growth and a["source_tag"] != b["source_tag"]:
        result["reason"] = "Source tags differ across annual periods; the growth basis is inconsistent"
        return result
    if not growth and (a["accession"] != b["accession"] or a["filed"] != b["filed"]):
        result["reason"] = "Ratio inputs do not come from the same filing version"
        return result
    divisor = _number(b["value"])
    if divisor <= 0:
        result["reason"] = "Denominator/base must be positive; zero or negative bases are unavailable"
        return result
    try:
        with localcontext() as context:
            context.prec = 50
            dividend = _number(a["value"])
            value = (dividend - divisor if growth else dividend) / divisor * Decimal(100)
            output_value = float(value)
        if not math.isfinite(output_value):
            raise ValueError("Percentage is outside the finite JSON number range")
    except (DecimalException, OverflowError, ValueError):
        result["reason"] = "Percentage cannot be represented as a finite JSON number"
        return result
    result.update(value=output_value, status="ok")
    return result


@register_tool(
    name="calculate_metrics",
    description=(
        "Calculate annual financial percentages from a dataset_id returned by get_financials. "
        "Reads saved facts directly; never supply or retype amounts. Returns formulas and full "
        "input evidence. Missing or incompatible periods, units or filing versions are unavailable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string", "pattern": "^[0-9a-f]{64}$", "description": "Snapshot ID returned by get_financials."},
            "metrics": {
                "type": "array", "items": {"type": "string", "enum": list(_METRICS)},
                "minItems": 1, "uniqueItems": True,
                "description": "Defaults to revenue_growth_yoy, net_income_growth_yoy and net_margin.",
            },
        },
        "required": ["dataset_id"], "additionalProperties": False,
    },
)
def calculate_metrics(dataset_id: str, metrics: list[str] | None = None) -> dict:
    names = _DEFAULT_METRICS if metrics is None else metrics
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or n not in _METRICS for n in names):
        raise ValueError("metrics must be a non-empty array of supported metric names")
    if len(names) != len(set(names)):
        raise ValueError("metrics must not contain duplicates")
    dataset = load_dataset(dataset_id)
    _validate_dataset(dataset)
    results = [_calculate(dataset, name, i) for i in range(len(dataset["periods"])) for name in names]
    output = {
        "ticker": dataset["ticker"], "cik": dataset["cik"], "dataset_id": dataset_id,
        "as_of": dataset["as_of"], "version_policy": dataset["version_policy"],
        "status": "ok" if any(r["status"] == "ok" for r in results) else "unavailable",
        "formulas": {name: _METRICS[name][2] for name in names}, "results": results,
    }
    if not results:
        output["reason"] = "No annual periods are present in the snapshot"
    return output
