import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine

_SHARE_CLASS_SUFFIX = re.compile(r"(\d*우선주|보통주)$")  # same KRX issuer-grouping rule as
# 06_post_filter -- duplicated (not imported) to keep modules loosely coupled, same pattern as
# every other module reading peer tables directly instead of importing peer code.


def fetch_trailing_returns(tickers: list[str], as_of_date, window_days: int = 252) -> pd.DataFrame:
    """Wide (date x ticker) daily_return matrix, strictly on or before as_of_date, for the
    simplified sample-covariance Sigma (see DESIGN.md -- no historical DCC snapshots exist)."""
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=window_days * 1.6)  # calendar buffer
    query = text(
        """
        SELECT ticker, date, daily_return FROM daily_prices
        WHERE ticker = ANY(:tickers) AND date > :cutoff AND date <= :as_of_date
        ORDER BY ticker, date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers, "cutoff": cutoff.date(), "as_of_date": as_of_date})
    return df.pivot(index="date", columns="ticker", values="daily_return").tail(window_days)


def fetch_market_caps_asof(tickers: list[str], as_of_date) -> dict[str, float]:
    """Each ticker's latest market_cap on or before as_of_date -- point-in-time, not today's."""
    query = text(
        """
        SELECT dp.ticker, dp.market_cap FROM daily_prices dp
        JOIN (
            SELECT ticker, MAX(date) AS max_date FROM daily_prices
            WHERE ticker = ANY(:tickers) AND date <= :as_of_date GROUP BY ticker
        ) latest ON dp.ticker = latest.ticker AND dp.date = latest.max_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers, "as_of_date": as_of_date})
    return dict(zip(df["ticker"], df["market_cap"]))


def fetch_issuer_groups(tickers: list[str]) -> dict[str, str]:
    query = text("SELECT ticker, name FROM securities WHERE ticker = ANY(:tickers)")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers})
    groups = {}
    for t, name in zip(df["ticker"], df["name"]):
        base = _SHARE_CLASS_SUFFIX.sub("", name or "") if name else ""
        groups[t] = base if base else t
    for t in tickers:
        groups.setdefault(t, t)
    return groups


def save_backtest_results(results: pd.DataFrame, label: str = "default") -> None:
    """results: DataFrame with columns as_of_date, portfolio_return, n_holdings."""
    delete_sql = text("DELETE FROM backtest_period_returns WHERE label = :label")
    insert_sql = text(
        """
        INSERT INTO backtest_period_returns (as_of_date, label, portfolio_return, n_holdings)
        VALUES (:as_of_date, :label, :portfolio_return, :n_holdings)
        """
    )
    rows = [
        {
            "as_of_date": r.as_of_date,
            "label": label,
            "portfolio_return": float(r.portfolio_return),
            "n_holdings": int(r.n_holdings),
        }
        for r in results.itertuples()
    ]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"label": label})
        conn.execute(insert_sql, rows)
