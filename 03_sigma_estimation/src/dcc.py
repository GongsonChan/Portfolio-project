import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _correlation_targeting(std_resid: np.ndarray) -> np.ndarray:
    """Unconditional correlation matrix Qbar from T x N standardized residuals."""
    T = std_resid.shape[0]
    return (std_resid.T @ std_resid) / T


def _dcc_recursion(std_resid: np.ndarray, qbar: np.ndarray, a: float, b: float) -> np.ndarray:
    """Q_t = (1-a-b)*Qbar + a*(e_{t-1} e_{t-1}') + b*Q_{t-1}, then R_t = corr(Q_t). Returns R: (T, N, N)."""
    T, N = std_resid.shape
    Q = np.zeros((T, N, N))
    R = np.zeros((T, N, N)) 
    Q[0] = qbar
    for t in range(T):
        if t > 0:
            outer = np.outer(std_resid[t - 1], std_resid[t - 1])
            Q[t] = (1 - a - b) * qbar + a * outer + b * Q[t - 1]
        d = np.sqrt(np.diag(Q[t]))
        R[t] = Q[t] / np.outer(d, d)
    return R


# (a, b) seeds to restart the optimizer from, spanning low/high persistence regimes.
# N x N likelihood evaluations are expensive for N~100, so this stays short (speed vs. extra safety trade-off).
_SEEDS = [(0.02, 0.95), (0.05, 0.50)]


def _to_unconstrained(a: float, b: float) -> np.ndarray:
    """Map a valid (a, b) pair to free real-valued (u, v) via s=a+b, p=a/s, each passed through a logit."""
    s = min(max(a + b, 1e-4), 1 - 1e-4)
    p = min(max(a / s, 1e-4), 1 - 1e-4)
    return np.array([np.log(s / (1 - s)), np.log(p / (1 - p))])


def _to_ab(params: np.ndarray) -> tuple[float, float]:
    """Inverse of _to_unconstrained: any (u, v) in R^2 maps to a valid a>=0, b>=0, a+b<1."""
    u, v = params
    s = 1 / (1 + np.exp(-u))
    p = 1 / (1 + np.exp(-v))
    return s * p, s * (1 - p)


def _neg_log_likelihood(params: np.ndarray, std_resid: np.ndarray, qbar: np.ndarray) -> float:
    a, b = _to_ab(params)
    R = _dcc_recursion(std_resid, qbar, a, b)  # (T, N, N)
    try:
        signs, logdets = np.linalg.slogdet(R)  # batched over T, much faster than a per-t Python loop
        if np.any(signs <= 0):
            return 1e10
        R_inv = np.linalg.inv(R)  # also batched over T
    except np.linalg.LinAlgError:
        return 1e10
    quad = np.einsum("ti,tij,tj->t", std_resid, R_inv, std_resid)
    e_sq = np.sum(std_resid**2, axis=1)
    ll = -0.5 * np.sum(logdets + quad - e_sq)
    return -ll


def fit_dcc_ab(std_resid_df: pd.DataFrame) -> dict:
    """Weekly/expensive step: estimate (a, b) only, via MLE over the given standardized-residual
    panel. (a, b) are optimized in an unconstrained reparametrized space (see _to_unconstrained/_to_ab)
    so the optimizer never hits the a+b<1 boundary, and multiple seeds are tried to avoid bad local optima.
    """
    std_resid = std_resid_df.values
    qbar = _correlation_targeting(std_resid)

    best = None
    for seed_a, seed_b in _SEEDS:
        result = minimize(
            _neg_log_likelihood,
            x0=_to_unconstrained(seed_a, seed_b),
            args=(std_resid, qbar),
            method="L-BFGS-B",
            options={"maxiter": 100},
        )
        if best is None or result.fun < best.fun:
            best = result

    a, b = _to_ab(best.x)
    return {
        "a": float(a),
        "b": float(b),
        "converged": bool(best.success),
        "log_likelihood": float(-best.fun),
    }


def apply_dcc(std_resid_df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """Daily/cheap step: recompute Qbar fresh from today's panel (no re-optimization, reuses a
    previously-fit a, b) and replay the recursion. Returns the most recent correlation matrix R_t.
    Robust to the universe having drifted slightly since (a, b) were last fit, since Qbar is
    rebuilt from whatever tickers are in std_resid_df now rather than cached."""
    std_resid = std_resid_df.values
    qbar = _correlation_targeting(std_resid)
    R = _dcc_recursion(std_resid, qbar, a, b)
    return pd.DataFrame(R[-1], index=std_resid_df.columns, columns=std_resid_df.columns)


def build_covariance(correlation: pd.DataFrame, cond_vol: pd.Series) -> pd.DataFrame:
    """Sigma = D R D, where D = diag(cond_vol). cond_vol must be aligned to correlation's index/columns."""
    vol = cond_vol.reindex(correlation.index)
    D = np.diag(vol.values)
    return pd.DataFrame(D @ correlation.values @ D, index=correlation.index, columns=correlation.columns)
