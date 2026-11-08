import numpy as np
import pandas as pd
from scipy.stats import norm

# z-scores for the one-sided confidence levels VaR uses (loss in the left tail).
Z_SCORES = {0.95: norm.ppf(0.95), 0.99: norm.ppf(0.99)}


def parametric_var(weights: pd.Series, sigma: pd.DataFrame, confidence: float) -> float:
    """Variance-covariance VaR: z_alpha * portfolio stdev. Assumes zero mean return over a
    1-day horizon (standard short-horizon convention -- the mean term is negligible next to the
    volatility term at daily frequency and estimating it adds more noise than signal). `sigma`
    must already be on the SAME horizon as the VaR being computed (1-day here -- see DESIGN.md's
    note on not reusing 05/07's 21-day-scaled Sigma)."""
    w = weights.reindex(sigma.index).fillna(0.0).values
    portfolio_var = w @ sigma.values @ w
    portfolio_vol = np.sqrt(max(portfolio_var, 0.0))
    return float(Z_SCORES[confidence] * portfolio_vol)


def parametric_es(weights: pd.Series, sigma: pd.DataFrame, confidence: float) -> float:
    """Analytic Gaussian Expected Shortfall: portfolio_vol * phi(z_alpha) / (1 - confidence),
    phi = standard normal pdf. Closed-form under the same normality assumption parametric_var
    makes -- included alongside the historical-sim ES specifically so the two can be compared
    (a parametric ES much smaller than historical ES is itself a sign the normal assumption is
    underestimating tail risk)."""
    w = weights.reindex(sigma.index).fillna(0.0).values
    portfolio_vol = np.sqrt(max(w @ sigma.values @ w, 0.0))
    z = Z_SCORES[confidence]
    return float(portfolio_vol * norm.pdf(z) / (1 - confidence))


def component_var(weights: pd.Series, sigma: pd.DataFrame, confidence: float) -> pd.Series:
    """Euler decomposition of parametric VaR into each position's contribution -- the standard
    way a risk desk answers "which position is actually driving our risk" (not the same
    question as position weight: a small, highly-correlated-with-everything-else position can
    contribute disproportionate risk). VaR is linear-homogeneous in weights under the Gaussian
    model, so the component VaRs sum exactly to total portfolio VaR (Euler's theorem) --
    component_i = w_i * (Sigma @ w)_i / portfolio_vol * z_alpha."""
    w = weights.reindex(sigma.index).fillna(0.0)
    sigma_w = sigma.values @ w.values
    portfolio_vol = np.sqrt(max(w.values @ sigma_w, 0.0))
    if portfolio_vol == 0:
        return pd.Series(0.0, index=w.index)
    marginal = sigma_w / portfolio_vol  # d(portfolio_vol)/d(w_i)
    return pd.Series(Z_SCORES[confidence] * w.values * marginal, index=w.index)


def historical_var_es(portfolio_returns: pd.Series, confidence: float) -> tuple[float, float]:
    """Historical simulation VaR + Expected Shortfall from an empirical return distribution
    (no normality assumption). VaR = -lower empirical quantile; ES = -mean of the tail beyond
    it. Both expressed as positive loss fractions, matching parametric_var's sign convention."""
    alpha = 1 - confidence
    var = float(-portfolio_returns.quantile(alpha))
    tail = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(alpha)]
    es = float(-tail.mean()) if not tail.empty else var
    return var, es


def portfolio_return_series(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """weights: Series indexed by ticker. returns: wide (date x ticker) daily_return matrix.
    Fixed-weight approximation (same weights applied across the whole window) -- standard for a
    short historical-simulation lookback, same assumption 05/06/07 already make about holding
    periods being short enough that drift between rebalances is a second-order effect."""
    aligned = returns.reindex(columns=weights.index).fillna(0.0)
    return aligned.dot(weights)
