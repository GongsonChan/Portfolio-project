import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine

LOOKBACK_DAYS = 300  # longest window (252) + skip(21) + buffer, before the *1.6 weekend/holiday pad

MAX_PLAUSIBLE_DAILY_RETURN = 0.5  # KRX's regulatory daily limit is +-30%; US has no such cap but
# a stock passing any reasonable liquidity filter essentially never gaps >50% in one day without
# a halt/split/data artifact behind it. Caught in the wild: ticker 052670 sat flat at a stale
# price with volume=0 for ~7 weeks (trading halt), then "returned" +299 (+29,900%) the day
# trading resumed -- not a real tradeable return, just a halted price being diffed against its
# resumption price. Nulling these out (rather than dropping the row) lets the existing NaN
# handling downstream (compute_targets' window.isna().any(), sequence.py's .dropna(), the
# factor models' median imputer) exclude the ticker for that window instead of training/
# labeling on a number that's mechanically impossible.


def _null_implausible_returns(df: pd.DataFrame) -> pd.DataFrame:
    bad = df["daily_return"].abs() > MAX_PLAUSIBLE_DAILY_RETURN
    if bad.any():
        df = df.copy()
        df.loc[bad, "daily_return"] = None
    return df


def fetch_price_history(as_of_date: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=LOOKBACK_DAYS * 1.6)
    query = text(
        """
        SELECT dp.ticker, dp.date, dp.close_price, dp.volume, dp.market_cap, dp.daily_return
        FROM daily_prices dp
        JOIN securities s ON dp.ticker = s.ticker
        WHERE s.market != 'INDEX' AND dp.date >= :cutoff AND dp.date <= :as_of_date
        ORDER BY dp.ticker, dp.date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cutoff": cutoff.date(), "as_of_date": as_of_date})
    return _null_implausible_returns(df)


def fetch_index_returns(as_of_date: str) -> dict[str, pd.Series]:
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=LOOKBACK_DAYS * 1.6)
    query = text(
        """
        SELECT ticker, date, daily_return FROM daily_prices
        WHERE ticker IN ('^KS11', '^GSPC') AND date >= :cutoff AND date <= :as_of_date
        ORDER BY ticker, date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cutoff": cutoff.date(), "as_of_date": as_of_date})
    return {
        "kospi": df[df["ticker"] == "^KS11"].set_index("date")["daily_return"],
        "spx": df[df["ticker"] == "^GSPC"].set_index("date")["daily_return"],
    }


def fetch_ticker_markets() -> dict[str, str]:
    query = text("SELECT ticker, market FROM securities WHERE market != 'INDEX'")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return dict(zip(df["ticker"], df["market"]))


def fetch_forward_returns(as_of_date: str, horizon_days: int = 21) -> pd.DataFrame:
    """Raw daily_return rows strictly after as_of_date, with enough calendar-day buffer to
    cover `horizon_days` trading days. Used to compute training targets (future return)."""
    buffer_end = pd.Timestamp(as_of_date) + pd.Timedelta(days=horizon_days * 2.0)  # extra room for long holiday weeks (e.g. 추석)
    query = text(
        """
        SELECT dp.ticker, dp.date, dp.daily_return
        FROM daily_prices dp
        JOIN securities s ON dp.ticker = s.ticker
        WHERE s.market != 'INDEX' AND dp.date > :as_of_date AND dp.date <= :buffer_end
        ORDER BY dp.ticker, dp.date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"as_of_date": as_of_date, "buffer_end": buffer_end.date()})
    return _null_implausible_returns(df)


def fetch_forward_index_returns(as_of_date: str, horizon_days: int = 21) -> pd.DataFrame:
    """Same forward window as fetch_forward_returns, but for the index tickers -- used to
    net the matching market index's return out of the stock target (excess return)."""
    buffer_end = pd.Timestamp(as_of_date) + pd.Timedelta(days=horizon_days * 2.0)
    query = text(
        """
        SELECT ticker, date, daily_return FROM daily_prices
        WHERE ticker IN ('^KS11', '^GSPC') AND date > :as_of_date AND date <= :buffer_end
        ORDER BY ticker, date
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"as_of_date": as_of_date, "buffer_end": buffer_end.date()})


def fetch_cond_vol_for_date(as_of_date: str) -> dict[str, float]:
    query = text("SELECT ticker, cond_vol FROM garch_estimates WHERE date = :d")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"d": as_of_date})
    return dict(zip(df["ticker"], df["cond_vol"]))


def fetch_latest_universe(label: str = "default") -> list[str]:
    """Same pattern as 03_sigma_estimation's fetch_latest_universe -- only ever reads the
    'default' universe unless told otherwise, so a custom one-off filter run can't silently
    feed into the automated daily inference pipeline."""
    query = text(
        """
        SELECT ticker FROM universe_selection
        WHERE label = :label
          AND as_of_date = (SELECT MAX(as_of_date) FROM universe_selection WHERE label = :label)
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"label": label})["ticker"].tolist()


def fetch_latest_complete_date_per_ticker(tickers: list[str]) -> dict[str, object]:
    """Per-ticker MAX(as_of_date) WHERE cond_vol IS NOT NULL -- each ticker's own latest date
    that 03_sigma_estimation has actually reached, instead of one shared date picked by
    whichever market/ticker is furthest ahead. Plain MAX(as_of_date) across the whole universe
    isn't safe: KOSPI/KOSDAQ and US don't update in lockstep, so a shared date would be set by
    whichever market is furthest ahead and silently drop every ticker in the market that's
    behind (its cond_vol is NULL at that date). run_inference.py groups the universe by this
    per-ticker value and predicts/saves each group against its own date instead, so a market
    that's a day behind still gets a view (dated to its own latest ready day)."""
    query = text(
        """
        SELECT ticker, MAX(as_of_date) AS as_of_date FROM ml_features
        WHERE ticker = ANY(:tickers) AND cond_vol IS NOT NULL
        GROUP BY ticker
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers})
    return dict(zip(df["ticker"], df["as_of_date"]))


def save_views(as_of_date, q: pd.Series, label: str = "default", omega: pd.Series | None = None) -> None:
    """q: Series indexed by ticker, the final meta-model Q prediction per ticker. omega: Series
    indexed by ticker (from fetch_omega), same index as q or a subset -- a ticker missing from
    it (e.g. no OOF history yet) gets omega=NULL, not a crash or a fabricated value."""
    delete_sql = text("DELETE FROM views WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        "INSERT INTO views (as_of_date, label, ticker, q, omega) VALUES (:as_of_date, :label, :ticker, :q, :omega)"
    )
    omega = omega if omega is not None else pd.Series(dtype=float)
    rows = [
        {
            "as_of_date": as_of_date,
            "label": label,
            "ticker": t,
            "q": float(v),
            "omega": float(omega[t]) if t in omega.index and pd.notna(omega[t]) else None,
        }
        for t, v in q.items()
    ]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, rows)


_OMEGA_UPSERT_SQL = text(
    """
    INSERT INTO omega_estimates (ticker, label, computed_at, omega, n_samples)
    VALUES (:ticker, :label, :computed_at, :omega, :n_samples)
    ON CONFLICT (ticker, label) DO UPDATE SET
        computed_at = EXCLUDED.computed_at, omega = EXCLUDED.omega, n_samples = EXCLUDED.n_samples
    """
)


def save_omega(omega: pd.DataFrame, computed_at, label: str = "default") -> None:
    """omega: DataFrame indexed by ticker with "omega" and "n_samples" columns. One current row
    per (ticker, label), upserted in place -- same "latest state" pattern as model_registry,
    not a views-style daily time series."""
    rows = [
        {"ticker": t, "label": label, "computed_at": computed_at, "omega": float(r.omega), "n_samples": int(r.n_samples)}
        for t, r in omega.iterrows()
    ]
    with engine.begin() as conn:
        conn.execute(_OMEGA_UPSERT_SQL, rows)


def fetch_omega(tickers: list[str], label: str = "default") -> dict[str, float]:
    query = text("SELECT ticker, omega FROM omega_estimates WHERE label = :label AND ticker = ANY(:tickers)")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"label": label, "tickers": tickers})
    return dict(zip(df["ticker"], df["omega"]))


def save_features(features: pd.DataFrame) -> None:
    """features: DataFrame with an as_of_date column (all rows same value) plus ticker + feature columns."""
    as_of_date = features["as_of_date"].iloc[0]
    cols = [c for c in features.columns if c not in ("as_of_date", "ticker")]
    col_list = ", ".join(["as_of_date", "ticker"] + cols)
    placeholders = ", ".join([f":{c}" for c in ["as_of_date", "ticker"] + cols])
    delete_sql = text("DELETE FROM ml_features WHERE as_of_date = :as_of_date")
    insert_sql = text(f"INSERT INTO ml_features ({col_list}) VALUES ({placeholders})")
    # psycopg2 binds Python float('nan') as the literal Postgres NUMERIC value 'NaN', not SQL
    # NULL -- only an actual None becomes NULL. to_dict("records") leaves pandas' missing-value
    # NaN as float('nan'), so every "missing feature" row was silently stored as NaN instead of
    # NULL, which makes "WHERE col IS NOT NULL" checks (e.g. daily_update_features.py's resume
    # point) wrongly treat it as present. DataFrame.where(..., None) can't fix this itself: a
    # float64 column forces the replacement back to NaN to keep its dtype. Have to drop to the
    # plain dict level first, then swap each float NaN for None one value at a time.
    records = features.to_dict("records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date})
        conn.execute(insert_sql, records)
