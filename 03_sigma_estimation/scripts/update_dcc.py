import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from dcc import apply_dcc, build_covariance, fit_dcc_ab
from repository import (
    engine,
    fetch_latest_cond_vol,
    fetch_latest_dcc_params,
    fetch_latest_universe,
    fetch_std_resid,
    save_covariance,
    save_dcc_params,
)

if __name__ == "__main__":
    # daily job: reuse (a, b) from the last weekly fit_dcc.py run; Qbar and R_t are still
    # recomputed fresh from today's universe/residuals (cheap, and robust to minor universe drift).
    # falls back to a full fit if no cached params exist yet for this label.
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()

    # skip entirely if DCC already reflects GARCH's latest date (e.g. weekend/holiday run)
    with engine.connect() as conn:
        latest_garch_date = conn.execute(text("SELECT MAX(date) FROM garch_estimates")).scalar()
        latest_cov_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM covariance_matrix WHERE label = :label"), {"label": args.label}
        ).scalar()
    if latest_cov_date is not None and latest_garch_date is not None and latest_cov_date >= latest_garch_date:
        print(f"GARCH 최신일({latest_garch_date})까지 DCC[{args.label}] 이미 반영됨 — 스킵")
        sys.exit(0)

    tickers = fetch_latest_universe(label=args.label)
    print(f"universe [{args.label}]: {len(tickers)} tickers")

    std_resid, missing = fetch_std_resid(tickers)
    if missing:
        print(f"GARCH 데이터 없어서 제외된 티커 ({len(missing)}개): {missing}")
    std_resid = std_resid.dropna()
    print(f"std_resid window: {std_resid.shape[0]} days x {std_resid.shape[1]} tickers")

    params = fetch_latest_dcc_params(label=args.label)
    today = date.today()
    if params is None:
        print("저장된 DCC 파라미터 없음 — 전체 재추정으로 폴백")
        dcc_result = fit_dcc_ab(std_resid)
        save_dcc_params(args.label, today, dcc_result)
        a, b = dcc_result["a"], dcc_result["b"]
    else:
        a, b = params["a"], params["b"]
        print(f"캐시된 DCC 파라미터 재사용: a={a:.4f}, b={b:.4f}")

    correlation = apply_dcc(std_resid, a, b)
    cond_vol = fetch_latest_cond_vol(tickers)
    sigma = build_covariance(correlation, cond_vol)
    save_covariance(today, sigma, label=args.label)
    print(f"saved {sigma.shape[0]}x{sigma.shape[1]} covariance matrix [{args.label}]")
