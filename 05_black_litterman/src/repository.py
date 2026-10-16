import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine


def fetch_latest_covariance(label: str = "default") -> tuple[pd.DataFrame, object]:
    """DCC already produces one internally-consistent Sigma for the whole universe as of a
    single date (see 03_sigma_estimation/DESIGN.md) -- no per-ticker date splitting needed
    here, just the most recent snapshot for this label. Returns (Sigma as a square ticker x
    ticker DataFrame, the as_of_date it was computed for)."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM covariance_matrix WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return pd.DataFrame(), None
        long = pd.read_sql(
            text(
                "SELECT ticker_i, ticker_j, covariance FROM covariance_matrix "
                "WHERE label = :label AND as_of_date = :as_of_date"
            ),
            conn,
            params={"label": label, "as_of_date": as_of_date},
        )
    sigma = long.pivot(index="ticker_i", columns="ticker_j", values="covariance")
    sigma.index.name = sigma.columns.name = None
    return sigma, as_of_date


def fetch_latest_views(tickers: list[str], label: str = "default") -> pd.DataFrame:
    """Per-ticker latest Q, mirroring 04_views_estimation's run_inference.py: views are saved
    with each ticker's OWN latest complete date (KOSPI/KOSDAQ and US don't move in lockstep),
    so this asks each ticker for its own latest row rather than assuming one shared as_of_date."""
    query = text(
        """
        SELECT v.ticker, v.q, v.as_of_date FROM views v
        JOIN (
            SELECT ticker, MAX(as_of_date) AS max_date FROM views
            WHERE label = :label AND ticker = ANY(:tickers)
            GROUP BY ticker
        ) latest ON v.ticker = latest.ticker AND v.as_of_date = latest.max_date
        WHERE v.label = :label
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"label": label, "tickers": tickers})


def fetch_omega(tickers: list[str], label: str = "default") -> dict[str, float]:
    query = text("SELECT ticker, omega FROM omega_estimates WHERE label = :label AND ticker = ANY(:tickers)")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"label": label, "tickers": tickers})
    return dict(zip(df["ticker"], df["omega"]))


def fetch_latest_market_caps(tickers: list[str]) -> dict[str, float]:
    """Per-ticker latest market_cap from 01's raw daily_prices -- same per-entity principle as
    fetch_latest_views: a ticker's own latest row, not one shared calendar date for everyone."""
    query = text(
        """
        SELECT dp.ticker, dp.market_cap FROM daily_prices dp
        JOIN (
            SELECT ticker, MAX(date) AS max_date FROM daily_prices
            WHERE ticker = ANY(:tickers) GROUP BY ticker
        ) latest ON dp.ticker = latest.ticker AND dp.date = latest.max_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers})
    return dict(zip(df["ticker"], df["market_cap"]))


def save_portfolio(as_of_date, weights: pd.Series, label: str = "default") -> None:
    """weights: Series indexed by ticker, the full-universe long-only BL+optimizer output (not
    yet top-N filtered or capped -- see 06_post_filter)."""
    delete_sql = text("DELETE FROM portfolio_weights WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        "INSERT INTO portfolio_weights (as_of_date, label, ticker, weight) VALUES (:as_of_date, :label, :ticker, :weight)"
    )
    rows = [{"as_of_date": as_of_date, "label": label, "ticker": t, "weight": float(w)} for t, w in weights.items()]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, rows)
