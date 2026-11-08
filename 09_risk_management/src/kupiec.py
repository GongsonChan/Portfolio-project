import numpy as np
from scipy.stats import binom, chi2


def kupiec_pof_test(n_days: int, n_breaches: int, confidence: float) -> tuple[float, float]:
    """Kupiec (1995) Proportion-of-Failures likelihood-ratio test: is the observed breach rate
    consistent with what this confidence level implies? H0: true breach probability = 1-confidence.
    Returns (LR statistic, p-value) -- LR ~ chi-squared(1) under H0. A low p-value (e.g. <0.05)
    means the VaR model's breach rate is statistically inconsistent with its stated confidence
    (either too conservative or, more worryingly, under-predicting risk)."""
    p = 1 - confidence
    x = n_breaches
    n = n_days
    if n == 0:
        return float("nan"), float("nan")

    phat = x / n
    # Guard the degenerate edges (0 or all breaches) where log(0) would blow up -- treat them as
    # the limiting likelihood instead of raising.
    def _log_lik(prob: float) -> float:
        if prob <= 0 or prob >= 1:
            return 0.0 if (x == 0 or x == n) else float("-inf")
        return x * np.log(prob) + (n - x) * np.log(1 - prob)

    lr_stat = -2 * (_log_lik(p) - _log_lik(phat))
    if not np.isfinite(lr_stat):
        lr_stat = 0.0
    p_value = float(1 - chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value


def traffic_light(n_days: int, n_breaches: int, confidence: float) -> str:
    """Basel traffic-light approach for VaR model backtesting, generalized to any (n_days,
    confidence) via the binomial CDF rather than linearly scaling Basel's published n=250/99%
    exception-count cutoffs (0-4/5-9/10+) -- an earlier version of this function did that linear
    scaling and it was wrong: it doesn't account for sampling variability shrinking as sqrt(n),
    so it produced boundaries far too tight for a larger sample (flagged "red" a result Kupiec's
    own test rated statistically unremarkable, p=0.31). Basel's own cutoffs are themselves just
    the first exception count where cumulative P(X <= k) crosses 95% (green/yellow) and 99.99%
    (yellow/red) under Binomial(n=250, p=0.01) -- using that same probability-based rule at
    whatever (n, p) this backtest actually has reproduces Basel's published boundaries exactly
    at n=250/p=0.01, and generalizes correctly everywhere else."""
    p = 1 - confidence
    cum_prob = binom.cdf(n_breaches, n_days, p)
    if cum_prob < 0.95:
        return "green"
    if cum_prob < 0.9999:
        return "yellow"
    return "red"
