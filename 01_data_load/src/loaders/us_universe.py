import io

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}  # a self-declared "...Bot/1.0" UA got 403'd by Wikipedia; a plain browser UA doesn't

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://www.slickcharts.com/nasdaq100"


def _clean_ticker(ticker: str) -> str:
    # yfinance uses '-' where index providers use '.' for share classes (e.g. BRK.B -> BRK-B)
    return ticker.strip().replace(".", "-")


def get_sp500() -> list[dict]:
    resp = requests.get(SP500_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    return [
        {
            "ticker": _clean_ticker(row["Symbol"]),
            "name": row["Security"],
            "sector": row["GICS Sector"],
            "industry": row["GICS Sub-Industry"],
        }
        for _, row in df.iterrows()
    ]


def get_nasdaq100() -> list[dict]:
    resp = requests.get(NASDAQ100_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    return [
        {
            "ticker": _clean_ticker(row["Symbol"]),
            "name": row["Company"],
            "sector": None,
            "industry": None,
        }
        for _, row in df.iterrows()
    ]


def get_universe() -> list[dict]:
    combined: dict[str, dict] = {}
    for row in get_sp500() + get_nasdaq100():
        combined.setdefault(row["ticker"], row)
    return list(combined.values())
