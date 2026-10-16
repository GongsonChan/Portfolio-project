import numpy as np
import pandas as pd

from repository import fetch_index_returns, fetch_price_history, fetch_ticker_markets

DEFAULT_WINDOW = 252  # ~1 year, matches the lookback already used by several factor features (mom_12_1, vol_1y, mdd, high_proximity) and MIN_HISTORY_DAYS eligibility
_MARKET_INDEX = {"KOSPI": "kospi", "KOSDAQ": "kospi", "US": "spx"}


def build_sequence_batch(as_of_date: str, window: int = DEFAULT_WINDOW) -> dict[str, np.ndarray]:
    """Returns {ticker: excess_return_sequence} for the `window` trading days ending at as_of_date.
    excess_return = stock daily_return - matching market index's daily_return (same day).
    Not persisted anywhere — meant to be called on demand when assembling sequence-model batches,
    unlike the factor panel (ml_features) which is precomputed/stored because it's expensive to build.
    """
    prices = fetch_price_history(as_of_date) 
    index_returns = fetch_index_returns(as_of_date)
    ticker_markets = fetch_ticker_markets()

    sequences: dict[str, np.ndarray] = {}
    for ticker, group in prices.groupby("ticker"):
        market = ticker_markets.get(ticker)
        index_key = _MARKET_INDEX.get(market)
        if index_key is None:
            continue

        group = group.sort_values("date").set_index("date")
        aligned = pd.concat(
            [group["daily_return"], index_returns[index_key]], axis=1, join="inner"
        ).dropna()
        aligned.columns = ["stock", "market"]
        if len(aligned) < window:
            continue

        excess = (aligned["stock"] - aligned["market"]).tail(window)
        sequences[ticker] = excess.values

    return sequences
