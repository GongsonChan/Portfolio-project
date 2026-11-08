import numpy as np

# Basel Committee's 1996 Market Risk Amendment / Basel II internal models approach: the
# regulatory capital multiplier k starts at 3.00 and steps up with backtesting exceptions in
# the trailing 250 trading days, per this published add-on schedule. This is a *different*
# question from kupiec.traffic_light()'s generalized binomial-CDF zones (whether the breach
# rate is statistically consistent with the model at ANY sample size) -- this table is Basel's
# literal, prescribed capital add-on, defined specifically at a 250-day window.
BASE_MULTIPLIER = 3.00
MAX_ADDON = 1.00
_ADDON_STEPS = [  # (breach count threshold, add-on to BASE_MULTIPLIER)
    (5, 0.40),
    (6, 0.50),
    (7, 0.65),
    (8, 0.75),
    (9, 0.85),
]


def capital_multiplier(n_breaches_250d: int) -> float:
    """k for a given count of VaR exceptions in the trailing 250 trading days."""
    if n_breaches_250d >= 10:
        return BASE_MULTIPLIER + MAX_ADDON
    addon = 0.0
    for threshold, step_addon in _ADDON_STEPS:
        if n_breaches_250d >= threshold:
            addon = step_addon
    return BASE_MULTIPLIER + addon


def capital_zone(n_breaches_250d: int) -> str:
    """Basel's green/yellow/red zone for a given trailing-250-day exception count -- the same
    0-4/5-9/10+ cutoffs capital_multiplier() steps on, named for display purposes."""
    if n_breaches_250d >= 10:
        return "red"
    if n_breaches_250d >= 5:
        return "yellow"
    return "green"


ZONE_MESSAGES = {
    "green": "모형 양호 -- 백테스트 결과가 신뢰수준과 통계적으로 부합, 추가 조치 불필요.",
    "yellow": "모형 정확도 저하 경고 -- 예외 발생 빈도가 기대보다 높음, 원인 분석 및 모니터링 강화 필요.",
    "red": "리스크 모형 위험 -- 꼬리 리스크를 심각하게 과소추정하고 있음이 확인됨, 모형 재검토(분포 가정·윈도우 등) 시급.",
}


def scale_to_10day(var_1day: float) -> float:
    """Basel's regulatory VaR horizon is 10 trading days, not 1 -- derived via the sqrt(10)
    iid-returns convention (see DESIGN.md) rather than recomputed from a 10-day return series.
    Cast to a native float (not np.float64) -- psycopg2 can't bind numpy scalars directly, same
    pitfall as the NaN/NULL binding bug documented elsewhere in this project."""
    return float(var_1day) * float(np.sqrt(10))


def required_capital(var_10day_today: float, trailing_10day_vars: list[float], multiplier: float) -> dict:
    """Basel 1996 Amendment capital charge: max(most recent prior day's 10-day VaR, k times the
    60-day rolling average of 10-day VaR). `trailing_10day_vars` is prior days only (today
    excluded), oldest-first; degrades gracefully to whatever history exists while daily
    reporting is still accumulating toward a full 60-day window (see DESIGN.md)."""
    window = trailing_10day_vars[-60:]
    avg_var = (sum(window) / len(window)) if window else var_10day_today
    prior_day_var = window[-1] if window else var_10day_today
    required = max(prior_day_var, multiplier * avg_var)
    return {
        "var_10day": var_10day_today,
        "avg_var_10day": avg_var,
        "n_days_in_average": len(window),
        "multiplier": multiplier,
        "required_capital": required,
    }
