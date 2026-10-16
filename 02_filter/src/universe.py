import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine
from factors import FACTORS, INVERTED_FACTORS, MOMENTUM_LOOKBACK, compute_raw_factors

DEFAULT_WEIGHTS = {name: 1.0 for name in FACTORS}
DEFAULT_WEIGHTS["momentum"] = 0.3  # recent-riser factor -- lowered from 1.0 so the universe
# isn't dominated by whatever already ran up (same reasoning as high_proximity below; BL/Q
# downstream pick up momentum again through their own channels, so the filter doesn't need to
# double down on it)
DEFAULT_WEIGHTS["high_proximity"] = 0.3  # 52-week-high proximity -- same "recent riser" signal
# as momentum, same reduction


def fetch_price_history(as_of_date: str, lookback_days: int = MOMENTUM_LOOKBACK + 30) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days * 1.6)  # buffer for weekends/holidays
    query = text(
        """
        SELECT ticker, date, close_price, volume, market_cap, daily_return
        FROM daily_prices
        WHERE date >= :cutoff AND date <= :as_of_date
        ORDER BY ticker, date
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"cutoff": cutoff.date(), "as_of_date": as_of_date})


def score_universe(raw_factors: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    z = (raw_factors - raw_factors.mean()) / raw_factors.std()
    for name in INVERTED_FACTORS:
        if name in z.columns:
            z[name] = -z[name]

    score = pd.Series(0.0, index=z.index)
    for name, weight in weights.items():
        if not weight:
            continue
        score += z[name].fillna(0) * weight
    return score.sort_values(ascending=False)


def select_universe(as_of_date: str, n: int = 100, weights: dict | None = None, label: str = "default") -> pd.DataFrame:
    prices = fetch_price_history(as_of_date)
    raw = compute_raw_factors(prices)
    scores = score_universe(raw, weights)
    top = scores.head(n)
    return pd.DataFrame(
        {
            "as_of_date": as_of_date,
            "label": label,
            "ticker": top.index,
            "rank": range(1, len(top) + 1),
            "score": top.values,
        }
    )


def save_universe(selection: pd.DataFrame) -> None:
    as_of_date = selection["as_of_date"].iloc[0]
    label = selection["label"].iloc[0]
    delete_sql = text("DELETE FROM universe_selection WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        """
        INSERT INTO universe_selection (as_of_date, label, ticker, rank, score)
        VALUES (:as_of_date, :label, :ticker, :rank, :score)
        """
    )
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, selection.to_dict("records"))
