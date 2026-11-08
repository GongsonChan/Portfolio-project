import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# 07's backtest.py does its own `sys.path.insert(0, .../07_backtest/src)` + `from repository
# import ...` at module load, so importing it the normal way (flat namespace, same convention
# every other script in this project uses) binds the name `repository` in sys.modules to 07's
# version. 09 has its OWN repository.py with different functions that would collide under that
# same cached name -- loaded explicitly by file path under a distinct module name instead, so
# it can't silently resolve to the wrong (cached) module.
sys.path.insert(0, str(ROOT / "07_backtest" / "src"))
from backtest import REBALANCE_EVERY_N_FOLDS, run_one_fold  # noqa: E402 -- reuses 07's exact
# point-in-time BL+filter reconstruction rather than re-implementing it a third time (see
# 09_risk_management/DESIGN.md)

sys.path.insert(0, str(ROOT / "09_risk_management" / "src"))
from kupiec import kupiec_pof_test, traffic_light  # noqa: E402
from var_models import historical_var_es, parametric_var, portfolio_return_series  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "risk_repository", ROOT / "09_risk_management" / "src" / "repository.py"
)
_risk_repository = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_risk_repository)
fetch_forward_returns = _risk_repository.fetch_forward_returns
save_var_backtest_daily = _risk_repository.save_var_backtest_daily
save_var_backtest_summary = _risk_repository.save_var_backtest_summary

CONFIDENCES = [0.95, 0.99]
METHODS = ["parametric", "historical"]
DEFAULT_OOF = ROOT / "04_views_estimation" / "output" / "meta_oof_lightgbm_leaves7.parquet"


def run(oof_path: str, label: str = "default") -> None:
    oof = pd.read_parquet(oof_path)
    fold_dates = sorted(oof["as_of_date"].unique())[::REBALANCE_EVERY_N_FOLDS]
    print(f"replaying {len(fold_dates)} rebalance periods from {oof_path}", flush=True)

    daily_rows = []  # parametric only -- what var_backtest_daily charts (see DESIGN.md)
    breach_counts = {(m, c): {"n_days": 0, "n_breaches": 0} for m in METHODS for c in CONFIDENCES}

    for i, d in enumerate(fold_dates):
        result = run_one_fold(oof, d) 
        if result is None:
            continue
        end_date = fold_dates[i + 1] if i + 1 < len(fold_dates) else None
        if end_date is None:
            continue  # last period has no realized forward window yet to validate against

        weights = result["final_weights"]
        trailing_returns = result["returns"]  # pre-period, daily scale -- same data used to
        # build this period's Sigma for BL, reused here for both VaR methods

        forward_returns = fetch_forward_returns(list(weights.index), d, end_date)
        if forward_returns.empty: 
            continue
        realized = portfolio_return_series(weights, forward_returns.reindex(columns=weights.index)).dropna()
        if realized.empty:
            continue

        sigma_daily = trailing_returns.cov()
        hist_series = portfolio_return_series(weights, trailing_returns).dropna()
        enough_hist = len(hist_series) > 20

        for conf in CONFIDENCES:
            var_param = parametric_var(weights, sigma_daily, conf)
            var_hist = historical_var_es(hist_series, conf)[0] if enough_hist else None

            for day, ret in realized.items():
                breach_param = -ret > var_param
                daily_rows.append(
                    {
                        "date": day,
                        "confidence": conf,
                        "var_value": var_param,
                        "realized_return": float(ret),
                        "breach": bool(breach_param),
                    }
                )
                breach_counts[("parametric", conf)]["n_days"] += 1
                breach_counts[("parametric", conf)]["n_breaches"] += int(breach_param)
                if var_hist is not None:
                    breach_counts[("historical", conf)]["n_days"] += 1
                    breach_counts[("historical", conf)]["n_breaches"] += int(-ret > var_hist)

        print(f"  {d} -> {end_date}: {len(realized)} realized days, n_holdings={len(weights)}", flush=True)

    rows_df = pd.DataFrame(daily_rows)
    if not rows_df.empty:
        save_var_backtest_daily(rows_df, label)

    today = date.today()
    for (method, conf), counts in breach_counts.items():
        n_days, n_breaches = counts["n_days"], counts["n_breaches"]
        if n_days == 0:
            print(f"[{method} {conf:.0%}] no backtest rows produced, skipping")
            continue
        lr_stat, p_value = kupiec_pof_test(n_days, n_breaches, conf)
        light = traffic_light(n_days, n_breaches, conf)
        save_var_backtest_summary(
            label,
            method,
            conf,
            today,
            {
                "n_days": n_days,
                "n_breaches": n_breaches,
                "expected_breach_rate": 1 - conf,
                "kupiec_lr_stat": lr_stat,
                "kupiec_p_value": p_value,
                "traffic_light": light,
            },
        )
        print(
            f"[{method} {conf:.0%}] n_days={n_days} breaches={n_breaches} "
            f"(expected={n_days * (1 - conf):.1f}) kupiec_p={p_value:.4f} light={light}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=str, default=str(DEFAULT_OOF))
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.oof, args.label)
