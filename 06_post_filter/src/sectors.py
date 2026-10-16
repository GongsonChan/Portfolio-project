import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from repository import fetch_sector_info, save_sectors

# Loaded explicitly (not relying on db.py's import-order side effect) so this module works even
# if something imports it before repository/db. Anyone cloning this repo just drops their own
# ANTHROPIC_API_KEY into 01_data_load/.env -- no code change needed.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "01_data_load" / ".env")

MODEL = "claude-haiku-5-5"  # cheap and plenty for a fixed-list classification task -- see
# 06_post_filter/DESIGN.md for why this isn't Opus

Sector = Literal[
    "Information Technology",
    "Health Care",
    "Financials",
    "Materials",
    "Industrials",
    "Energy",
    "Consumer Discretionary",
    "Consumer Staples",
    "Communication Services",
    "Real Estate",
    "Utilities",
]
GICS_SECTORS = list(Sector.__args__)


class _Classification(BaseModel):
    ticker: str
    sector: Sector


class _ClassificationResult(BaseModel):
    classifications: list[_Classification]


def _classify_via_llm(rows: list[dict]) -> dict[str, str]:
    """rows: [{"ticker", "name", "market"}, ...]. One Claude call classifies the whole batch --
    KRX's official API has no industry-classification endpoint (checked) and no well-maintained
    free source covers this project's ~3,700-ticker pool, so this stands in until/unless a real
    data source is wired in (see DESIGN.md)."""
    listing = "\n".join(f"{r['ticker']}\t{r['name']}\t{r['market']}" for r in rows)
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify each of these stocks into exactly one GICS-style sector from "
                    f"this fixed list: {', '.join(GICS_SECTORS)}.\n\n"
                    "Judge by the company's actual business, not its market/exchange.\n\n"
                    "Stocks (ticker, name, market):\n" + listing
                ),
            }
        ],
        output_format=_ClassificationResult,
    )
    return {c.ticker: c.sector for c in response.parsed_output.classifications}


def ensure_sectors(tickers: list[str]) -> dict[str, str]:
    """Returns {ticker: GICS sector} for every given ticker, classifying and caching (writing
    to securities.sector) any that don't already have a valid one. securities.sector is
    sometimes NULL and, for KRX tickers, sometimes holds the market-tier/division name (e.g.
    "우량기업부") instead of a real industry -- both cases are "needs classification", not an
    error."""
    info = fetch_sector_info(tickers)
    valid = info[info["sector"].isin(GICS_SECTORS)]
    known = dict(zip(valid["ticker"], valid["sector"]))

    missing = info[~info["ticker"].isin(known)]
    if not missing.empty:
        rows = missing[["ticker", "name", "market"]].to_dict("records")
        print(f"classifying {len(rows)} tickers with no valid sector via {MODEL}...")
        new_sectors = _classify_via_llm(rows)
        save_sectors(new_sectors)
        known.update(new_sectors)

    return known
