"""Fetch annual company facts and return an immutable calculation input."""

from app.agent.tools import register_tool
from app.config import settings
from app.finance.sec import METRIC_TAGS, get_financial_dataset
from app.finance.store import save_dataset


@register_tool(
    name="get_financials",
    description=(
        "Fetch official SEC annual US-GAAP financial facts in USD for a US ticker. "
        "Returns periods, filing dates, source URLs and an immutable dataset_id. "
        "For growth/margins, fetch the required base facts (at least 2 years for growth), "
        "then pass dataset_id to calculate_metrics without retyping numbers. "
        "Only annual 10-K/10-K/A data; no quarterly, TTM, estimates or market prices. "
        "Uses latest filings available by as_of; missing/ambiguous facts are null."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "US ticker, e.g. AAPL or BRK-B"},
            "metrics": {
                "type": "array", "items": {"type": "string", "enum": list(METRIC_TAGS)},
                "minItems": 1, "uniqueItems": True,
                "description": "Base facts to fetch; defaults to the configured financial fact list.",
            },
            "period": {"type": "string", "enum": ["annual"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": settings.financial_default_limit},
            "as_of": {"type": "string", "description": "Filing cutoff YYYY-MM-DD; defaults to today (UTC)"},
        },
        "required": ["ticker"], "additionalProperties": False,
    },
)
def get_financials(ticker: str, metrics=None, period="annual", limit=None, as_of=None) -> dict:
    dataset = get_financial_dataset(ticker, metrics=metrics, period=period, limit=limit, as_of=as_of)
    return {"dataset_id": save_dataset(dataset), **dataset}
