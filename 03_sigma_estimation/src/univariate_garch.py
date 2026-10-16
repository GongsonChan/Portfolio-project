import pandas as pd
from arch import arch_model

MIN_OBS = 500  # minimum observations needed for a stable GARCH(1,1) fit
STORE_WINDOW = 500  # only persist the most recent N days (enough for the DCC stage's window)


def fit_garch(returns: pd.Series) -> dict | None:
    """returns: daily_return series (decimal, e.g. 0.01 = 1%), sorted by date ascending, no NaNs."""
    if len(returns) < MIN_OBS:
        return None

    scaled = returns * 100  # arch recommends percentage-scale returns for numerical stability
    model = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
    try:
        res = model.fit(disp="off", show_warning=False)
    except Exception:
        return None

    dates = returns.index[-STORE_WINDOW:]
    return {
        "mu": float(res.params["mu"]),
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
        "dates": dates,
        "cond_vol": (res.conditional_volatility / 100).tail(STORE_WINDOW).values,
        "std_resid": res.std_resid.tail(STORE_WINDOW).values,
    }


def apply_fixed_params(returns: pd.Series, params: dict) -> dict | None:
    """Like fit_garch, but reuses previously-estimated (mu, omega, alpha, beta) instead of
    re-running MLE — just replays the GARCH recursion with today's return added. ~17x faster
    than fit_garch, meant for the daily update between weekly refits."""
    if len(returns) < MIN_OBS:
        return None

    scaled = returns * 100
    model = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
    param_series = pd.Series(
        {"mu": params["mu"], "omega": params["omega"], "alpha[1]": params["alpha"], "beta[1]": params["beta"]}
    )
    try:
        res = model.fix(param_series)
    except Exception:
        return None

    dates = returns.index[-STORE_WINDOW:]
    return {
        "mu": params["mu"],
        "omega": params["omega"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "dates": dates,
        "cond_vol": (res.conditional_volatility / 100).tail(STORE_WINDOW).values,
        "std_resid": res.std_resid.tail(STORE_WINDOW).values,
    }
