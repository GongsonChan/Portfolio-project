import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dcc import apply_dcc, build_covariance, fit_dcc_ab
from repository import (
    fetch_latest_cond_vol,
    fetch_latest_universe,
    fetch_std_resid,
    save_covariance,
    save_dcc_params,
)

if __name__ == "__main__":
    # weekly (weekend) job: re-estimate (a, b) via MLE for the given label's current universe.
    # for the daily update that reuses these params, see update_dcc.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()

    tickers = fetch_latest_universe(label=args.label)
    print(f"universe [{args.label}]: {len(tickers)} tickers")

    std_resid, missing = fetch_std_resid(tickers)
    if missing:
        print(f"GARCH 데이터 없어서 제외된 티커 ({len(missing)}개): {missing}")
    std_resid = std_resid.dropna()
    print(f"std_resid window: {std_resid.shape[0]} days x {std_resid.shape[1]} tickers")

    dcc_result = fit_dcc_ab(std_resid)
    print(f"DCC fit: a={dcc_result['a']:.4f}, b={dcc_result['b']:.4f}, converged={dcc_result['converged']}")

    today = date.today()
    save_dcc_params(args.label, today, dcc_result)

    correlation = apply_dcc(std_resid, dcc_result["a"], dcc_result["b"])
    cond_vol = fetch_latest_cond_vol(tickers)
    sigma = build_covariance(correlation, cond_vol)
    save_covariance(today, sigma, label=args.label)
    print(f"saved {sigma.shape[0]}x{sigma.shape[1]} covariance matrix [{args.label}]")
