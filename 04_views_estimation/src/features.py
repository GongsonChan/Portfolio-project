import numpy as np
import pandas as pd

MOM_WINDOWS = {"mom_1m": 21, "mom_3m": 63, "mom_6m": 126}
MOMENTUM_12_1_LOOKBACK = 252
MOMENTUM_12_1_SKIP = 21
VOL_WINDOWS = {"vol_1m": 21, "vol_3m": 63, "vol_1y": 252}
RSI_WINDOW = 14
MA_WINDOWS = {"ma_gap_50": 50, "ma_gap_200": 200}
BB_WINDOW = 20
LIQUIDITY_WINDOW = 60
BETA_WINDOW = 252
VOL_SURGE_SHORT = 5
VOL_SURGE_LONG = 60
MIN_HISTORY_DAYS = 273  # longest window (12-1 momentum) + buffer, same floor as 02_filter
MARKET_WINDOWS = {"1w": 5, "1m": 21, "3m": 63}


def is_eligible(group: pd.DataFrame) -> bool:
    if len(group) < MIN_HISTORY_DAYS:
        return False
    coverage = group.tail(252)["close_price"].notna().mean()
    return coverage >= 0.9


def _cum_return(returns: pd.Series) -> float:
    return (1 + returns).prod() - 1


def _mdd(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    return dd.min()


def _rsi(close: pd.Series, window: int) -> float:
    delta = close.diff().tail(window)
    gain = delta.clip(lower=0).mean()
    loss = -delta.clip(upper=0).mean()
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return 100 - (100 / (1 + gain / loss))


def compute_momentum_features(g: pd.DataFrame) -> dict:
    r = g["daily_return"]
    out = {name: _cum_return(r.tail(w)) for name, w in MOM_WINDOWS.items()}
    window = r.tail(MOMENTUM_12_1_LOOKBACK).iloc[:-MOMENTUM_12_1_SKIP]
    out["mom_12_1"] = _cum_return(window)
    out["reversal_1w"] = _cum_return(r.tail(5))
    return out


def compute_vol_features(g: pd.DataFrame) -> dict:
    r = g["daily_return"]
    out = {name: r.tail(w).std() for name, w in VOL_WINDOWS.items()}
    out["mdd"] = _mdd(r.tail(252))
    out["skew_1m"] = r.tail(21).skew()
    out["max_ret_1m"] = r.tail(21).max()
    return out


def compute_liquidity_features(g: pd.DataFrame) -> dict:
    window = g.tail(LIQUIDITY_WINDOW)
    dollar_volume = window["volume"] * window["close_price"]
    liquidity = np.log(dollar_volume.mean()) if dollar_volume.mean() and dollar_volume.mean() > 0 else np.nan
    amihud_series = window["daily_return"].abs() / dollar_volume.replace(0, np.nan)
    return {"liquidity": liquidity, "amihud": amihud_series.mean()}


def compute_technical_features(g: pd.DataFrame) -> dict:
    close = g["close_price"]
    out = {"rsi_14": _rsi(close, RSI_WINDOW)}
    for name, w in MA_WINDOWS.items():
        ma = close.tail(w).mean()
        out[name] = close.iloc[-1] / ma - 1 if ma else np.nan
    bb_window = close.tail(BB_WINDOW)
    mean_, std_ = bb_window.mean(), bb_window.std()
    out["bb_pctb"] = (close.iloc[-1] - (mean_ - 2 * std_)) / (4 * std_) if std_ and std_ > 0 else np.nan
    vol_short = g["volume"].tail(VOL_SURGE_SHORT).mean()
    vol_long = g["volume"].tail(VOL_SURGE_LONG).mean()
    out["volume_surge"] = vol_short / vol_long if vol_long else np.nan
    return out


def compute_size_features(g: pd.DataFrame) -> dict:
    latest_cap = g["market_cap"].iloc[-1]
    window = g.tail(252)
    high = window["close_price"].max()
    return {
        "size": np.log(latest_cap) if latest_cap and latest_cap > 0 else np.nan,
        "high_proximity": window["close_price"].iloc[-1] / high if high else np.nan,
    }


def compute_beta_features(stock_returns: pd.Series, market_returns: pd.Series, window: int = BETA_WINDOW) -> dict:
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner").tail(window)
    aligned.columns = ["stock", "market"]
    aligned = aligned.dropna()
    if len(aligned) < window * 0.5:
        return {"beta": np.nan, "idio_vol": np.nan}
    var = aligned["market"].var()
    beta = aligned["stock"].cov(aligned["market"]) / var if var else np.nan
    if pd.isna(beta):
        return {"beta": np.nan, "idio_vol": np.nan}
    resid = aligned["stock"] - beta * aligned["market"]
    return {"beta": beta, "idio_vol": resid.std()}


def compute_stock_features(g: pd.DataFrame, market_returns: pd.Series, cond_vol: float | None) -> dict:
    """g: one ticker's price history (date, close_price, volume, market_cap, daily_return), sorted by date.
    market_returns: the matching index's (KOSPI for KR, S&P500 for US) daily_return series, same date index."""
    out = {}
    out.update(compute_momentum_features(g))
    out.update(compute_vol_features(g))
    out.update(compute_liquidity_features(g))
    out.update(compute_technical_features(g))
    out.update(compute_size_features(g))
    stock_returns = g.set_index("date")["daily_return"]
    out.update(compute_beta_features(stock_returns, market_returns))
    out["cond_vol"] = cond_vol
    return out


def compute_market_features(index_returns: dict[str, pd.Series]) -> dict:
    """index_returns: {'kospi': series, 'spx': series} of daily_return, sorted ascending by date."""
    out = {}
    for label, series in index_returns.items():
        for wname, w in MARKET_WINDOWS.items():
            out[f"{label}_ret_{wname}"] = _cum_return(series.tail(w))
        out[f"{label}_vol"] = series.tail(21).std()
    return out
