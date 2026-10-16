import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine

_PARAM_UPSERT_SQL = text(
    """
    INSERT INTO garch_params (ticker, fit_date, mu, omega, alpha, beta)
    VALUES (:ticker, :fit_date, :mu, :omega, :alpha, :beta)
    ON CONFLICT (ticker, fit_date) DO UPDATE SET
        mu = EXCLUDED.mu, omega = EXCLUDED.omega, alpha = EXCLUDED.alpha, beta = EXCLUDED.beta
    """
)

_ESTIMATE_UPSERT_SQL = text(
    """
    INSERT INTO garch_estimates (ticker, date, cond_vol, std_resid)
    VALUES (:ticker, :date, :cond_vol, :std_resid)
    ON CONFLICT (ticker, date) DO UPDATE SET
        cond_vol = EXCLUDED.cond_vol, std_resid = EXCLUDED.std_resid
    """
)


def fetch_all_tickers() -> list[str]:
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT ticker FROM securities"), conn)["ticker"].tolist()


def fetch_returns(ticker: str) -> pd.Series:
    query = text("SELECT date, daily_return FROM daily_prices WHERE ticker = :t ORDER BY date")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"t": ticker})
    return df.dropna().set_index("date")["daily_return"]


def save_garch_result(ticker: str, fit_date, result: dict) -> None:
    """Full weekly refit: writes both the re-estimated params and the updated estimates."""
    with engine.begin() as conn:
        conn.execute(
            _PARAM_UPSERT_SQL,
            {
                "ticker": ticker,
                "fit_date": fit_date,
                "mu": result["mu"],
                "omega": result["omega"],
                "alpha": result["alpha"],
                "beta": result["beta"],
            },
        )
        conn.execute(_ESTIMATE_UPSERT_SQL, _estimate_rows(ticker, result))


def save_garch_estimates(ticker: str, result: dict) -> None:
    """Daily lightweight update: only refreshes cond_vol/std_resid, leaves garch_params untouched
    (params still come from the last weekly refit)."""
    with engine.begin() as conn:
        conn.execute(_ESTIMATE_UPSERT_SQL, _estimate_rows(ticker, result))


def _estimate_rows(ticker: str, result: dict) -> list[dict]:
    return [
        {"ticker": ticker, "date": d, "cond_vol": float(cv), "std_resid": float(sr)}
        for d, cv, sr in zip(result["dates"], result["cond_vol"], result["std_resid"])
    ]


def fetch_latest_params(ticker: str) -> dict | None:
    query = text(
        """
        SELECT mu, omega, alpha, beta FROM garch_params
        WHERE ticker = :ticker ORDER BY fit_date DESC LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"ticker": ticker}).mappings().first()
    return {k: float(v) for k, v in row.items()} if row else None


def fetch_std_resid(tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Wide-format DataFrame: index=date, columns=ticker, values=standardized residual.
    Also returns the subset of requested tickers that had no GARCH data at all (skipped during fitting)."""
    query = text("SELECT ticker, date, std_resid FROM garch_estimates WHERE ticker = ANY(:tickers)")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers})
    wide = df.pivot(index="date", columns="ticker", values="std_resid")
    missing = sorted(set(tickers) - set(wide.columns))
    return wide, missing


def fetch_latest_cond_vol(tickers: list[str]) -> pd.Series:
    query = text(
        """
        SELECT DISTINCT ON (ticker) ticker, cond_vol
        FROM garch_estimates
        WHERE ticker = ANY(:tickers)
        ORDER BY ticker, date DESC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers})
    return df.set_index("ticker")["cond_vol"]


def fetch_latest_universe(label: str = "default") -> list[str]:
    """Only ever reads the system's 'default' universe, so a custom one-off filter run
    (different label) can never silently feed into the automated nightly Sigma/BL pipeline."""
    query = text(
        """
        SELECT ticker FROM universe_selection
        WHERE label = :label
          AND as_of_date = (SELECT MAX(as_of_date) FROM universe_selection WHERE label = :label)
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"label": label})["ticker"].tolist()


_DCC_PARAM_UPSERT_SQL = text(
    """
    INSERT INTO dcc_params (label, fit_date, a, b, converged, log_likelihood)
    VALUES (:label, :fit_date, :a, :b, :converged, :log_likelihood)
    ON CONFLICT (label, fit_date) DO UPDATE SET
        a = EXCLUDED.a, b = EXCLUDED.b, converged = EXCLUDED.converged, log_likelihood = EXCLUDED.log_likelihood
    """
)


def save_dcc_params(label: str, fit_date, result: dict) -> None:
    with engine.begin() as conn:
        conn.execute(
            _DCC_PARAM_UPSERT_SQL,
            {
                "label": label,
                "fit_date": fit_date,
                "a": result["a"],
                "b": result["b"],
                "converged": result["converged"],
                "log_likelihood": result["log_likelihood"],
            },
        )


def fetch_latest_dcc_params(label: str = "default") -> dict | None:
    query = text("SELECT a, b FROM dcc_params WHERE label = :label ORDER BY fit_date DESC LIMIT 1")
    with engine.connect() as conn:
        row = conn.execute(query, {"label": label}).mappings().first()
    return {k: float(v) for k, v in row.items()} if row else None


def save_covariance(as_of_date, cov: pd.DataFrame, label: str = "default") -> None:
    delete_sql = text("DELETE FROM covariance_matrix WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        """
        INSERT INTO covariance_matrix (as_of_date, label, ticker_i, ticker_j, covariance)
        VALUES (:as_of_date, :label, :ticker_i, :ticker_j, :covariance) 
        """
    )
    rows = [
        {"as_of_date": as_of_date, "label": label, "ticker_i": i, "ticker_j": j, "covariance": float(cov.loc[i, j])}
        for i in cov.index
        for j in cov.columns
    ]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, rows)
