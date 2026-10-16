import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21
VOL_WINDOW = 252
LIQUIDITY_WINDOW = 60
MIN_HISTORY_DAYS = 273  # momentum lookback + skip, the longest requirement


def is_eligible(group: pd.DataFrame) -> bool:
    """A ticker is eligible if it has enough recent, mostly-unbroken price history."""
    if len(group) < MIN_HISTORY_DAYS:
        return False
    recent = group.tail(VOL_WINDOW)
    coverage = recent["close_price"].notna().mean()
    return coverage >= 0.9


def compute_momentum(group: pd.DataFrame) -> float:
    window = group["daily_return"].tail(MOMENTUM_LOOKBACK).iloc[:-MOMENTUM_SKIP]
    return (1 + window).prod() - 1


def compute_sharpe(group: pd.DataFrame) -> float:
    window = group["daily_return"].tail(VOL_WINDOW)
    std = window.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return (window.mean() * 252) / (std * np.sqrt(252))


def compute_liquidity(group: pd.DataFrame) -> float:
    window = group.tail(LIQUIDITY_WINDOW)
    dollar_volume = (window["volume"] * window["close_price"]).mean()
    return np.log(dollar_volume) if dollar_volume and dollar_volume > 0 else np.nan


def compute_market_cap_score(group: pd.DataFrame) -> float:
    latest = group["market_cap"].iloc[-1]
    return np.log(latest) if latest and latest > 0 else np.nan


def compute_high_proximity(group: pd.DataFrame) -> float:
    window = group.tail(VOL_WINDOW)
    high = window["close_price"].max()
    latest = window["close_price"].iloc[-1]
    return latest / high if high else np.nan


def compute_mdd(group: pd.DataFrame) -> float:
    window = group["daily_return"].tail(VOL_WINDOW)
    cum = (1 + window).cumprod()
    drawdown = cum / cum.cummax() - 1
    return drawdown.min()  # negative number; closer to 0 is better


FACTORS = {
    "momentum": compute_momentum,  # 12-1개월 누적수익률 (최근 1개월 제외)
    "sharpe": compute_sharpe,  # 최근 1년 일별수익률 기준 연환산 샤프비율
    "liquidity": compute_liquidity,  # 최근 60일 평균 거래대금(volume*close_price)의 log
    "market_cap": compute_market_cap_score,  # 최신 시가총액의 log
    "high_proximity": compute_high_proximity,  # 현재가 / 최근 1년 최고가 (52주 신고가 근접도)
    "mdd": compute_mdd,  # 최근 1년 최대낙폭 (음수, 0에 가까울수록 좋음)
}

# factors where a HIGHER raw value is WORSE, so their z-score must be flipped
INVERTED_FACTORS = {"mdd"}


def compute_raw_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: long-format DataFrame with columns [ticker, date, close_price, volume, market_cap, daily_return],
    sorted by date within each ticker. Returns one row per eligible ticker with raw factor values."""
    rows = []
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date")
        if not is_eligible(group):
            continue
        row = {"ticker": ticker}
        for name, fn in FACTORS.items():
            row[name] = fn(group)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker")
