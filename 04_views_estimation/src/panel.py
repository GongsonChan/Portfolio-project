from datetime import date

import pandas as pd

from features import compute_market_features, compute_stock_features, is_eligible
from repository import fetch_cond_vol_for_date, fetch_index_returns, fetch_price_history, fetch_ticker_markets

_MARKET_INDEX = {"KOSPI": "kospi", "KOSDAQ": "kospi", "US": "spx"}


def build_feature_panel(as_of_date: str) -> pd.DataFrame:
    prices = fetch_price_history(as_of_date)
    index_returns = fetch_index_returns(as_of_date)
    cond_vol_map = fetch_cond_vol_for_date(as_of_date)
    ticker_markets = fetch_ticker_markets()
    market_features = compute_market_features(index_returns)
    as_of = date.fromisoformat(as_of_date)

    rows = []
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date")
        # fetch_price_history's "date <= as_of_date" window has no lower bound on how stale the
        # LATEST row is -- a ticker whose own price data hasn't reached as_of_date yet (e.g. 01
        # fetched US through today but KOSPI/KOSDAQ are still a day behind) would otherwise
        # silently compute every price-based feature from yesterday's trailing window and save
        # it mislabeled under today's as_of_date. Nothing downstream should ever claim a date
        # 01's own raw price data hasn't actually reached for that ticker, so require an exact
        # match on the latest row instead of "as of or before".
        if group["date"].iloc[-1] != as_of:
            continue
        if not is_eligible(group):
            continue
        market = ticker_markets.get(ticker)
        index_key = _MARKET_INDEX.get(market)
        if index_key is None:
            continue
        stock_features = compute_stock_features(group, index_returns[index_key], cond_vol_map.get(ticker))
        rows.append({"as_of_date": as_of_date, "ticker": ticker, **stock_features, **market_features})

    return pd.DataFrame(rows)
