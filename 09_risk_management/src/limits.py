import pandas as pd

# Risk limits -- expressed as a fraction of portfolio value (1.0 = 100%), same convention as
# VaR/ES throughout this module. Exposed as a plain dict (not hardcoded inline) for the same
# reason 02_filter's `weights` / 06's `--top-n` are parameters, not constants: a risk desk
# tunes limits far more often than it changes the VaR methodology itself.
DEFAULT_LIMITS = {
    "var_99_1d": 0.08,              # 1-day 99% VaR (parametric) must stay under 8% of portfolio value
    "var_95_1d": 0.05,              # 1-day 95% VaR (parametric) under 5%
    "es_99_1d": 0.10,               # 1-day 99% ES under 10%
    "es_95_1d": 0.07,               # 1-day 95% ES under 7%
    "max_component_var_share": 0.35,  # no single position's contribution to total VaR exceeds 35%
}


def check_limits(
    var_by_method_conf: dict[tuple[str, float], tuple[float, float]],
    component_shares: pd.Series,
    limits: dict[str, float] = DEFAULT_LIMITS,
) -> list[dict]:
    """var_by_method_conf: {(method, confidence): (var_value, es_value)}. component_shares:
    each position's share of total parametric VaR (component_var / total_var). Returns one
    record per limit, breached or not -- the daily report shows the full picture, not just
    violations, so "everything's fine" is visible too rather than an empty report being
    ambiguous between "checked, all green" and "didn't run"."""
    records = []

    def _add(name: str, observed: float, limit_value: float) -> None:
        records.append(
            {
                "limit_name": name,
                "limit_value": limit_value,
                "observed_value": observed,
                "breached": observed > limit_value,
            }
        )

    if ("parametric", 0.99) in var_by_method_conf:
        _add("var_99_1d", var_by_method_conf[("parametric", 0.99)][0], limits["var_99_1d"])
        _add("es_99_1d", var_by_method_conf[("parametric", 0.99)][1], limits["es_99_1d"])
    if ("parametric", 0.95) in var_by_method_conf:
        _add("var_95_1d", var_by_method_conf[("parametric", 0.95)][0], limits["var_95_1d"])
        _add("es_95_1d", var_by_method_conf[("parametric", 0.95)][1], limits["es_95_1d"])

    if component_shares is not None and not component_shares.empty:
        worst_ticker = component_shares.idxmax()
        _add(f"max_component_var_share (worst: {worst_ticker})", float(component_shares.max()), limits["max_component_var_share"])

    return records
