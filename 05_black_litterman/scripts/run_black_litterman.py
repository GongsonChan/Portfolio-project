import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

from repository import fetch_latest_covariance, fetch_latest_market_caps, fetch_latest_views, fetch_omega, save_portfolio

RISK_AVERSION = 2.5  # fixed conventional value (see DESIGN.md) -- used both for the market-
# implied prior and the final max-quadratic-utility optimization
TAU = 0.05  # fixed conventional BL scaling value
HORIZON_DAYS = 21  # covariance_matrix (03_sigma_estimation) is built from GARCH's DAILY
# conditional volatility (see dcc.build_covariance), but Q is a 21-trading-day forward excess
# return -- without rescaling, Sigma is ~21x too small relative to Q's horizon, so the
# optimizer's risk penalty is badly underweighted vs. the return term and over-concentrates in
# whatever has the highest Q. Variance scales linearly with horizon under a random-walk
# assumption (same convention as annualizing daily variance by *252), so multiply by 21 here
# to put Sigma on the same horizon as Q before BL ever sees it.


def run(label: str = "default") -> None:
    """BL + long-only mean-variance optimization over the FULL universe -- no top-N cut, no
    per-ticker/issuer weight caps. That post-processing moved to 06_post_filter, which reads
    this module's full output and applies selection/caps on top of it. 05's job stops at "what
    does Black-Litterman think the universe should look like."""
    sigma, sigma_date = fetch_latest_covariance(label)
    if sigma.empty:
        raise RuntimeError(f"no covariance_matrix snapshot found for label={label}")
    print(f"Sigma: {sigma.shape[0]}x{sigma.shape[0]} tickers as of {sigma_date}")

    universe = list(sigma.index)
    views_df = fetch_latest_views(universe, label)
    q_map = dict(zip(views_df["ticker"], views_df["q"]))
    omega_map = fetch_omega(universe, label)
    cap_map = fetch_latest_market_caps(universe)

    # Sigma/Q/Omega/market-cap each carry their own date (see DESIGN.md) and can disagree on
    # which tickers they currently cover (e.g. a ticker newly added to the universe that
    # doesn't have a Q yet) -- only run BL on the intersection, and say exactly who got dropped
    # and why rather than silently shrinking the universe.
    usable = [t for t in universe if t in q_map and t in omega_map and t in cap_map and pd.notna(cap_map[t])]
    dropped = set(universe) - set(usable)
    if dropped:
        print(f"skipping {len(dropped)}/{len(universe)} tickers missing Q/Ω/market cap: {sorted(dropped)}")

    cov_matrix = sigma.loc[usable, usable] * HORIZON_DAYS
    market_caps = pd.Series({t: cap_map[t] for t in usable})
    absolute_views = pd.Series({t: q_map[t] for t in usable})
    omega_diag = pd.DataFrame(np.diag([omega_map[t] for t in usable]), index=usable, columns=usable)

    bl = BlackLittermanModel(
        cov_matrix,
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

    today = date.today()
    save_portfolio(today, weights, label)
    print(f"saved {len(weights)} portfolio weights for {today} (label={label})")
    print(weights.sort_values(ascending=False).head(20))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
