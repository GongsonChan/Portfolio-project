import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine


def fetch_final_portfolio(label: str = "default") -> tuple[pd.Series, object]:
    """06_post_filter's latest (top-N, capped) weights -- what VaR is actually measured
    against, not 05's uncapped universe-wide output (see DESIGN.md)."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM final_portfolio WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return pd.Series(dtype=float), None
        df = pd.read_sql(
            text("SELECT ticker, weight FROM final_portfolio WHERE label = :label AND as_of_date = :as_of_date"),
            conn,
            params={"label": label, "as_of_date": as_of_date},
        )
    return pd.Series(df["weight"].values, index=df["ticker"]), as_of_date


def fetch_covariance(tickers: list[str], label: str = "default") -> pd.DataFrame:
    """Latest Sigma snapshot from 03, restricted to the given tickers -- left at its native
    DAILY scale (no *21 horizon scaling like 05/07 apply for their 21-day Q). Returns an empty
    frame if no snapshot covers these tickers."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM covariance_matrix WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return pd.DataFrame()
        long = pd.read_sql(
            text(
                """
                SELECT ticker_i, ticker_j, covariance FROM covariance_matrix
                WHERE label = :label AND as_of_date = :as_of_date
                  AND ticker_i = ANY(:tickers) AND ticker_j = ANY(:tickers)
                """
            ),
            conn,
            params={"label": label, "as_of_date": as_of_date, "tickers": tickers},
        )
    if long.empty:
        return pd.DataFrame()
    sigma = long.pivot(index="ticker_i", columns="ticker_j", values="covariance")
    sigma.index.name = sigma.columns.name = None
    return sigma


def fetch_trailing_returns(tickers: list[str], as_of_date, window_days: int = 252) -> pd.DataFrame:
    """Wide (date x ticker) daily_return matrix on or before as_of_date -- same shape as
    07_backtest's version (not imported, see repository pattern note in that module's DESIGN.md)."""
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=window_days * 1.6)
    query = text(
        """
        SELECT ticker, date, daily_return FROM daily_prices
        WHERE ticker = ANY(:tickers) AND date > :cutoff AND date <= :as_of_date
        ORDER BY ticker, date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers, "cutoff": cutoff.date(), "as_of_date": as_of_date})
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="daily_return").tail(window_days)


def fetch_forward_returns(tickers: list[str], start_date, end_date) -> pd.DataFrame:
    """Wide (date x ticker) daily_return matrix strictly AFTER start_date through end_date --
    the realized P&L path used to backtest a VaR estimate made as of start_date."""
    query = text(
        """
        SELECT ticker, date, daily_return FROM daily_prices
        WHERE ticker = ANY(:tickers) AND date > :start_date AND date <= :end_date
        ORDER BY ticker, date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers, "start_date": start_date, "end_date": end_date})
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="daily_return")


def fetch_prior_var_series(label: str, as_of_date, method: str = "parametric", confidence: float = 0.99, limit: int = 60) -> list[float]:
    """1-day VaR values strictly BEFORE as_of_date, oldest-first, for basel_capital's 60-day
    rolling average -- naturally returns fewer than `limit` rows while daily history is still
    accumulating (see DESIGN.md)."""
    query = text(
        """
        SELECT var_value FROM var_estimates
        WHERE label = :label AND method = :method AND confidence = :confidence AND as_of_date < :as_of_date
        ORDER BY as_of_date DESC LIMIT :limit
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn, params={"label": label, "method": method, "confidence": confidence, "as_of_date": as_of_date, "limit": limit}
        )
    return list(df["var_value"].astype(float).iloc[::-1])  # oldest-first, native float (not Decimal)


def fetch_trailing_breach_count(label: str, confidence: float = 0.99, limit: int = 250) -> int:
    """Count of VaR exceptions among the most recent `limit` rows of var_backtest_daily --
    Basel's own prescribed trailing window for the capital multiplier (see basel_capital.py)."""
    query = text(
        """
        SELECT breach FROM var_backtest_daily
        WHERE label = :label AND confidence = :confidence
        ORDER BY date DESC LIMIT :limit
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"label": label, "confidence": confidence, "limit": limit})
    return int(df["breach"].sum()) if not df.empty else 0


def save_basel_capital(as_of_date, label: str, record: dict) -> None:
    """record: {"n_breaches_250d", "multiplier", "var_10day", "avg_var_10day",
    "n_days_in_average", "required_capital"}."""
    delete_sql = text("DELETE FROM basel_capital_requirement WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        """
        INSERT INTO basel_capital_requirement
            (as_of_date, label, n_breaches_250d, multiplier, var_10day, avg_var_10day, n_days_in_average, required_capital)
        VALUES
            (:as_of_date, :label, :n_breaches_250d, :multiplier, :var_10day, :avg_var_10day, :n_days_in_average, :required_capital)
        """
    )
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, {**record, "as_of_date": as_of_date, "label": label})


def save_var_estimates(as_of_date, label: str, rows: list[dict]) -> None:
    """rows: [{"method", "confidence", "var_value", "es_value"}, ...]."""
    delete_sql = text("DELETE FROM var_estimates WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        """
        INSERT INTO var_estimates (as_of_date, label, method, confidence, var_value, es_value)
        VALUES (:as_of_date, :label, :method, :confidence, :var_value, :es_value)
        """
    )
    payload = [{**r, "as_of_date": as_of_date, "label": label} for r in rows]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, payload)


def save_component_var(as_of_date, label: str, confidence: float, component: pd.Series, total_var: float) -> None:
    delete_sql = text(
        "DELETE FROM component_var WHERE as_of_date = :as_of_date AND label = :label AND confidence = :confidence"
    )
    insert_sql = text(
        """
        INSERT INTO component_var (as_of_date, label, ticker, confidence, component_var, share_of_total)
        VALUES (:as_of_date, :label, :ticker, :confidence, :component_var, :share_of_total)
        """
    )
    rows = [
        {
            "as_of_date": as_of_date,
            "label": label,
            "ticker": t,
            "confidence": confidence,
            "component_var": float(v),
            "share_of_total": float(v) / total_var if total_var else 0.0,
        }
        for t, v in component.items()
    ]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label, "confidence": confidence})
        if rows:
            conn.execute(insert_sql, rows)


def save_risk_limit_checks(as_of_date, label: str, records: list[dict]) -> None:
    delete_sql = text("DELETE FROM risk_limit_checks WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        """
        INSERT INTO risk_limit_checks (as_of_date, label, limit_name, limit_value, observed_value, breached)
        VALUES (:as_of_date, :label, :limit_name, :limit_value, :observed_value, :breached)
        """
    )
    payload = [{**r, "as_of_date": as_of_date, "label": label} for r in records]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        if payload:
            conn.execute(insert_sql, payload)


def save_var_backtest_daily(rows: pd.DataFrame, label: str) -> None:
    """rows: DataFrame with columns date, confidence, var_value, realized_return, breach."""
    delete_sql = text("DELETE FROM var_backtest_daily WHERE label = :label")
    insert_sql = text(
        """
        INSERT INTO var_backtest_daily (date, label, confidence, var_value, realized_return, breach)
        VALUES (:date, :label, :confidence, :var_value, :realized_return, :breach)
        ON CONFLICT (date, label, confidence) DO UPDATE SET
            var_value = EXCLUDED.var_value, realized_return = EXCLUDED.realized_return, breach = EXCLUDED.breach
        """
    )
    payload = [{**r, "label": label} for r in rows.to_dict("records")]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"label": label})
        if payload:
            conn.execute(insert_sql, payload)


def save_var_backtest_summary(label: str, method: str, confidence: float, computed_at, stats: dict) -> None:
    insert_sql = text(
        """
        INSERT INTO var_backtest_summary
            (label, method, confidence, n_days, n_breaches, expected_breach_rate,
             kupiec_lr_stat, kupiec_p_value, traffic_light, computed_at)
        VALUES
            (:label, :method, :confidence, :n_days, :n_breaches, :expected_breach_rate,
             :kupiec_lr_stat, :kupiec_p_value, :traffic_light, :computed_at)
        ON CONFLICT (label, method, confidence) DO UPDATE SET
            n_days = EXCLUDED.n_days, n_breaches = EXCLUDED.n_breaches,
            expected_breach_rate = EXCLUDED.expected_breach_rate,
            kupiec_lr_stat = EXCLUDED.kupiec_lr_stat, kupiec_p_value = EXCLUDED.kupiec_p_value,
            traffic_light = EXCLUDED.traffic_light, computed_at = EXCLUDED.computed_at
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, {"label": label, "method": method, "confidence": confidence, "computed_at": computed_at, **stats})
