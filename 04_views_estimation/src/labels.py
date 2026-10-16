import pandas as pd

from repository import fetch_forward_index_returns, fetch_forward_returns, fetch_ticker_markets

HORIZON_DAYS = 21  # matches the 21-trading-day (monthly) prediction horizon used everywhere else
_MARKET_INDEX = {"KOSPI": "^KS11", "KOSDAQ": "^KS11", "US": "^GSPC"}


def compute_targets(as_of_date: str, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """Excess forward `horizon_days`-trading-day return: the stock's own compounded return
    minus its matching market index's compounded return over the identical window, starting
    the day after as_of_date. Raw (non-excess) return is dominated by common market-wide moves
    that cross-sectional factors/sequences have no hope of predicting -- netting it out here
    mirrors the excess-return transform sequence.py already applies to its input, so both the
    label and the sequence input isolate the same idiosyncratic signal. A ticker is missing
    from the result if its own window or its index's window isn't fully realized yet (as_of_date
    too recent) or the ticker stopped trading before the window completed -- both cases
    correctly mean "no usable label yet", not an error.
    """
    df = fetch_forward_returns(as_of_date, horizon_days) 
    index_df = fetch_forward_index_returns(as_of_date, horizon_days)
    ticker_markets = fetch_ticker_markets()

    index_cum_return = {}
    for index_ticker, group in index_df.groupby("ticker"):
        window = group.sort_values("date")["daily_return"].head(horizon_days)
        if len(window) < horizon_days or window.isna().any():
            continue
        index_cum_return[index_ticker] = (1 + window).prod() - 1

    targets = {}
    for ticker, group in df.groupby("ticker"):
        window = group.sort_values("date")["daily_return"].head(horizon_days)
        if len(window) < horizon_days or window.isna().any():
            continue
        index_ticker = _MARKET_INDEX.get(ticker_markets.get(ticker))
        if index_ticker not in index_cum_return:
            continue
        stock_cum_return = (1 + window).prod() - 1
        targets[ticker] = stock_cum_return - index_cum_return[index_ticker]
    return pd.Series(targets, name="target")
