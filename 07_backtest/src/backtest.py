import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

from repository import fetch_issuer_groups, fetch_market_caps_asof, fetch_trailing_returns

RISK_AVERSION = 2.5  # same fixed values as 05_black_litterman -- this replays 05/06's logic,
TAU = 0.05  # not a different methodology, so they must match
HORIZON_DAYS = 21  # Q/target are 21-trading-day forward excess returns; Sigma built from daily
# returns needs the same *21 horizon scaling 05 applies (see 05_black_litterman/DESIGN.md)

CANDIDATE_POOL = 150  # per-fold universe size -- each OOF fold has ~3,000 tickers, too many to
# run a full covariance/BL on every week across 254 folds; take the |Q|-largest as a stand-in
# for "the universe 02_filter would have picked" (see DESIGN.md)
TOP_N = 10
MAX_WEIGHT = 0.20  # issuer-group cap, same as 06
OMEGA_WINDOW_DAYS = 365
OMEGA_MIN_SAMPLES_PRIOR = 30  # same shrinkage prior as 04's compute_omega.py
SIGMA_WINDOW_DAYS = 252
MIN_SIGMA_OBS = 60  # a ticker needs at least this many overlapping trailing return days to
# trust its row/column of the sample covariance -- otherwise drop it from that fold


def _cap_and_renormalize(weights: pd.Series, cap: float, groups: dict[str, str]) -> pd.Series:
    """Same algorithm as 06_post_filter -- duplicated, not imported, so this module stays a
    self-contained replay rather than depending on another module's script."""
    w = weights.copy()
    group_of = pd.Series(groups).reindex(w.index)
    for _ in range(len(w)):
        group_sum = w.groupby(group_of).sum()
        over_groups = group_sum[group_sum > cap].index
        if len(over_groups) == 0:
            break
        excess = (group_sum[over_groups] - cap).sum()
        over_mask = group_of.isin(over_groups)
        w[over_mask] *= (cap / group_sum[group_of[over_mask]]).values
        under_mask = ~over_mask
        w[under_mask] += excess * (w[under_mask] / w[under_mask].sum())
    return w


def _omega_asof(oof: pd.DataFrame, as_of_date, window_days: int = OMEGA_WINDOW_DAYS) -> dict[str, float]:
    """Same shrinkage logic as 04's compute_omega.py, but using only OOF rows strictly BEFORE
    as_of_date -- using this fold's own (or a later fold's) rows would leak the future into its
    own uncertainty estimate."""
    dates = pd.to_datetime(oof["as_of_date"])
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=window_days)
    hist = oof[(dates < pd.Timestamp(as_of_date)) & (dates > cutoff)]
    if hist.empty:
        return {}
    err = hist["target"] - hist["pred"]
    pooled_var = err.var()
    per_ticker = hist.assign(err=err).groupby("ticker")["err"].agg(var="var", n_samples="count")
    per_ticker["var"] = per_ticker["var"].fillna(pooled_var)
    w = per_ticker["n_samples"] / (per_ticker["n_samples"] + OMEGA_MIN_SAMPLES_PRIOR)
    omega = w * per_ticker["var"] + (1 - w) * pooled_var
    return dict(omega)


def run_one_fold(oof: pd.DataFrame, as_of_date) -> dict | None:
    """Replays 05+06 for one historical rebalance date. Returns None if there isn't enough
    data yet (no OOF history for Omega, or too few tickers with a trustworthy Sigma row)."""
    fold = oof[oof["as_of_date"] == as_of_date]
    # By pred, NOT |pred| -- this is long-only (weight_bounds=(0,1) below), so the candidate
    # pool must be "highest predicted return", not "most confident in either direction". Using
    # abs() here was a real bug: it let in stocks the model expected to fall hard, half-filling
    # the pool with the wrong side of the trade.
    candidates = fold.sort_values("pred", ascending=False).head(CANDIDATE_POOL)
    tickers = list(candidates["ticker"])
    q_map = dict(zip(candidates["ticker"], candidates["pred"]))
    target_map = dict(zip(candidates["ticker"], candidates["target"]))

    omega_map = _omega_asof(oof, as_of_date)
    if not omega_map:
        return None
    cap_map = fetch_market_caps_asof(tickers, as_of_date)
    returns = fetch_trailing_returns(tickers, as_of_date, window_days=SIGMA_WINDOW_DAYS)

    usable = [
        t
        for t in tickers
        if t in omega_map
        and t in cap_map
        and pd.notna(cap_map[t])
        and t in returns.columns
        and returns[t].notna().sum() >= MIN_SIGMA_OBS
    ]
    if len(usable) < 20:
        return None

    sigma = returns[usable].cov() * HORIZON_DAYS
    market_caps = pd.Series({t: cap_map[t] for t in usable})
    absolute_views = pd.Series({t: q_map[t] for t in usable})
    omega_diag = pd.DataFrame(np.diag([omega_map[t] for t in usable]), index=usable, columns=usable)

    bl = BlackLittermanModel(
        sigma,
        pi="market",
        market_caps=market_caps,
        absolute_views=absolute_views,
        omega=omega_diag,
        tau=TAU,
        risk_aversion=RISK_AVERSION,
    )
    mu_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()

    ef = EfficientFrontier(mu_bl, cov_bl, weight_bounds=(0, 1))
    ef.max_quadratic_utility(risk_aversion=RISK_AVERSION)
    weights = pd.Series(ef.clean_weights())

    top = weights.sort_values(ascending=False).head(TOP_N)
    top = top[top > 0]
    if top.empty:
        return None
    groups = fetch_issuer_groups(list(top.index))
    final_weights = _cap_and_renormalize(top / top.sum(), MAX_WEIGHT, groups)

    portfolio_return = sum(final_weights[t] * target_map[t] for t in final_weights.index)
    return {"as_of_date": as_of_date, "portfolio_return": portfolio_return, "n_holdings": len(final_weights)}


REBALANCE_EVERY_N_FOLDS = 3  # OOF folds are weekly (~7 days apart); the target horizon is 21
# trading days, so taking every fold would make consecutive periods' 21-day return windows
# overlap by ~14 days (2/3) -- the same realized days counted in multiple periods' returns,
# inflating the apparent Sharpe. Every 3rd fold instead gives genuinely non-overlapping ~21-day
# periods: rebalancing is 21-trading-day, not weekly.


def run_backtest(oof: pd.DataFrame) -> pd.DataFrame:
    fold_dates = sorted(oof["as_of_date"].unique())[::REBALANCE_EVERY_N_FOLDS]
    rows = []
    for d in fold_dates:
        result = run_one_fold(oof, d)
        if result is None:
            continue
        rows.append(result)
        print(f"  {d}: return={result['portfolio_return']:.4f} n={result['n_holdings']}", flush=True)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    """periods_per_year uses the actual median gap between fold dates, not a hardcoded 52 --
    robust to the recent daily-cadence stretch mixed into otherwise-weekly fold spacing."""
    returns = results["portfolio_return"]
    dates = pd.to_datetime(results["as_of_date"])
    median_gap_days = dates.diff().dropna().dt.days.median()
    periods_per_year = 365.25 / median_gap_days

    cumulative = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized_return = (1 + cumulative) ** (periods_per_year / n_periods) - 1 if n_periods else float("nan")
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol else float("nan")

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        "n_periods": n_periods,
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
