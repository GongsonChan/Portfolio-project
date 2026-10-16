import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import engine


def fetch_latest_final_portfolio(label: str = "default") -> tuple[pd.DataFrame, object]:
    """06_post_filter's latest final weights, joined with name/sector for display."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM final_portfolio WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return pd.DataFrame(), None
        df = pd.read_sql(
            text(
                """
                SELECT fp.ticker, fp.weight, s.name, s.sector, s.market
                FROM final_portfolio fp JOIN securities s ON fp.ticker = s.ticker
                WHERE fp.label = :label AND fp.as_of_date = :as_of_date
                ORDER BY fp.weight DESC
                """
            ),
            conn,
            params={"label": label, "as_of_date": as_of_date},
        )
    return df, as_of_date


def fetch_backtest_results(label: str = "default") -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT as_of_date, portfolio_return, n_holdings FROM backtest_period_returns "
                "WHERE label = :label ORDER BY as_of_date"
            ),
            conn,
            params={"label": label},
        )


def summarize_backtest(results: pd.DataFrame) -> dict:
    """Same calculation as 07_backtest/src/backtest.py's summarize() -- duplicated rather than
    imported so this read-only web module doesn't depend on 07's script path."""
    if results.empty:
        return {}
    returns = results["portfolio_return"]
    dates = pd.to_datetime(results["as_of_date"])
    median_gap_days = dates.diff().dropna().dt.days.median()
    periods_per_year = 365.25 / median_gap_days if median_gap_days else float("nan")

    cumulative = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized_return = (1 + cumulative) ** (periods_per_year / n_periods) - 1 if n_periods else float("nan")
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol else float("nan")

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "n_periods": n_periods,
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "equity_curve": list(zip(dates.dt.strftime("%Y-%m-%d"), equity)),
    }
