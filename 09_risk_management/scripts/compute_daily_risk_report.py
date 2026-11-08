import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from basel_capital import ZONE_MESSAGES, capital_multiplier, capital_zone, required_capital, scale_to_10day
from limits import DEFAULT_LIMITS, check_limits
from repository import (
    fetch_covariance,
    fetch_final_portfolio,
    fetch_prior_var_series,
    fetch_trailing_breach_count,
    fetch_trailing_returns,
    save_basel_capital,
    save_component_var,
    save_risk_limit_checks,
    save_var_estimates,
)
from var_models import component_var, historical_var_es, parametric_es, parametric_var, portfolio_return_series

CONFIDENCES = [0.95, 0.99]
HIST_WINDOW_DAYS = 252  # ~1 trading year of history for the empirical return distribution


def run(label: str = "default") -> None: 
    weights, as_of_date = fetch_final_portfolio(label)
    if weights.empty:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(weights)} holdings as of {as_of_date}")

    sigma = fetch_covariance(list(weights.index), label)
    usable_sigma = [t for t in weights.index if t in sigma.index] if not sigma.empty else []
    if len(usable_sigma) < len(weights):
        missing = set(weights.index) - set(usable_sigma)
        print(f"  parametric: skipping {len(missing)} holdings missing from Sigma: {sorted(missing)}")

    returns = fetch_trailing_returns(list(weights.index), as_of_date, window_days=HIST_WINDOW_DAYS)
    usable_hist = [t for t in weights.index if t in returns.columns] if not returns.empty else []
    if len(usable_hist) < len(weights):
        missing = set(weights.index) - set(usable_hist)
        print(f"  historical: skipping {len(missing)} holdings missing trailing returns: {sorted(missing)}")

    rows = []
    var_by_method_conf = {}  # (method, confidence) -> (var, es), fed into the limit checks below
    worst_component_shares = None  # from the highest confidence level computed, for the
    # concentration limit -- component VaR scales with confidence but the SHARE (what the
    # limit actually checks) is confidence-invariant under the Gaussian model, so any level works

    if len(usable_sigma) >= 2:
        w_param = weights.loc[usable_sigma]
        sigma_param = sigma.loc[usable_sigma, usable_sigma]
        for conf in CONFIDENCES:
            var = parametric_var(w_param, sigma_param, conf)
            es = parametric_es(w_param, sigma_param, conf)
            rows.append({"method": "parametric", "confidence": conf, "var_value": var, "es_value": es})
            var_by_method_conf[("parametric", conf)] = (var, es)
            print(f"  [parametric {conf:.0%}] VaR={var:.4f} ES={es:.4f}")

            comp = component_var(w_param, sigma_param, conf)
            save_component_var(as_of_date, label, conf, comp, var)
            worst_component_shares = comp / var if var else comp
    else:
        print("  parametric: not enough tickers with Sigma coverage, skipped")

    if len(usable_hist) >= 2:
        w_hist = weights.loc[usable_hist]
        port_returns = portfolio_return_series(w_hist, returns[usable_hist]).dropna()
        for conf in CONFIDENCES:
            var, es = historical_var_es(port_returns, conf)
            rows.append({"method": "historical", "confidence": conf, "var_value": var, "es_value": es})
            var_by_method_conf[("historical", conf)] = (var, es)
            print(f"  [historical {conf:.0%}] VaR={var:.4f} ES={es:.4f} (n={len(port_returns)} days)")
    else:
        print("  historical: not enough tickers with trailing returns, skipped")

    if not rows:
        raise RuntimeError("neither VaR method produced a result -- check Sigma/returns coverage")

    save_var_estimates(as_of_date, label, rows)
    print(f"saved {len(rows)} VaR/ES estimates for {as_of_date} (label={label})")

    limit_records = check_limits(var_by_method_conf, worst_component_shares, DEFAULT_LIMITS)
    save_risk_limit_checks(as_of_date, label, limit_records)
    print(f"\nlimit checks ({as_of_date}):")
    any_breach = False
    for r in limit_records:
        flag = "BREACH" if r["breached"] else "ok"
        if r["breached"]:
            any_breach = True
        print(f"  [{flag}] {r['limit_name']}: observed={r['observed_value']:.4f} limit={r['limit_value']:.4f}")
    if any_breach:
        print("\n*** one or more risk limits breached -- see rows above ***")

    if ("parametric", 0.99) in var_by_method_conf:
        var_1day_today = var_by_method_conf[("parametric", 0.99)][0]
        var_10day_today = scale_to_10day(var_1day_today)
        prior_1day_series = fetch_prior_var_series(label, as_of_date, method="parametric", confidence=0.99)
        prior_10day_series = [scale_to_10day(v) for v in prior_1day_series]

        n_breaches_250d = fetch_trailing_breach_count(label, confidence=0.99)
        k = capital_multiplier(n_breaches_250d)
        capital = required_capital(var_10day_today, prior_10day_series, k)

        save_basel_capital(
            as_of_date,
            label,
            {
                "n_breaches_250d": n_breaches_250d,
                "multiplier": capital["multiplier"],
                "var_10day": capital["var_10day"],
                "avg_var_10day": capital["avg_var_10day"],
                "n_days_in_average": capital["n_days_in_average"],
                "required_capital": capital["required_capital"],
            },
        )
        zone = capital_zone(n_breaches_250d)
        print(
            f"\nbasel capital multiplier ({as_of_date}): "
            f"n_breaches(250d)={n_breaches_250d} k={k:.2f} [{zone.upper()}] "
            f"VaR(10d,99%)={var_10day_today:.4f} "
            f"avg(10d, n={capital['n_days_in_average']})={capital['avg_var_10day']:.4f} "
            f"-> required capital={capital['required_capital']:.4f} (fraction of portfolio value)"
        )
        print(f"  [{zone}] {ZONE_MESSAGES[zone]}")
        if capital["n_days_in_average"] < 20:
            print(
                f"  (note: only {capital['n_days_in_average']} prior day(s) of VaR history -- "
                "the 60-day average degenerates with this little data, so this number isn't "
                "representative yet. Accumulates daily via daily_update.py.)"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
