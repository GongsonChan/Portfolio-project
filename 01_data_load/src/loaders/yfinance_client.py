import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm
from yfinance.config import YfConfig

# yfinance's own per-HTTP-request retry (exponential backoff on transient
# connection/timeout errors -- exactly the WinError 10014 class seen locally)
# ships DISABLED (default 0). Our old fix only wrapped the whole yf.download()
# call in a blind outer retry, which reruns cookie+crumb+chart from scratch on
# every failure instead of retrying just the one request that actually failed.
# Enabling this lets yfinance retry at the right granularity.
YfConfig.network.retries = 4

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
REQUEST_DELAY_SECONDS = 0.5  # deliberate throttle between tickers -- failures climb the longer
# a run sustains ~1-2 tickers/sec with no gap at all, consistent with a request-rate limit that
# small/short test scripts never run long enough to hit


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, threads=False, progress=False)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            # Even a single-ticker yf.download() comes back with a (field, ticker) MultiIndex
            # in this yfinance version, not plain columns. Flatten to just the field level here
            # -- pd.concat(frames, axis=1) below adds the ticker level back as the outer level,
            # so keeping it here too would nest it twice ((ticker, field, ticker)) and break
            # prices[ticker] downstream (no literal "Close" column, just a MultiIndex containing one).
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_SECONDS)
    return None


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """One yf.download call per ticker (not a single bulk multi-ticker call), each retried up
    to MAX_RETRIES times. Yahoo's crumb/cookie auth fails intermittently -- empirically NOT a
    one-time-poisons-the-whole-run thing (the same ticker fails in one run and succeeds
    moments later in the next, with no code change in between), so neither threading nor a
    single-attempt sequential loop reliably fixes it. A short retry-with-delay per ticker is
    the practical mitigation for a flaky external API, on top of the existing gap-fill logic
    that picks up anything still missing on the next scheduled run."""
    frames = {}
    for ticker in tqdm(tickers, unit="ticker", desc="downloading prices"):
        df = _download_one(ticker, start, end)
        if df is not None:
            frames[ticker] = df
        time.sleep(REQUEST_DELAY_SECONDS)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)  # matches group_by="ticker"'s (ticker, field) MultiIndex columns


def get_shares_outstanding(ticker: str) -> float | None:
    try:
        return yf.Ticker(ticker).fast_info.get("shares") # 발행 주식수
    except Exception:
        return None
