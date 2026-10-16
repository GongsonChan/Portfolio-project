import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text
from tqdm import tqdm

from repository import engine, fetch_all_tickers, fetch_latest_params, fetch_returns, save_garch_estimates, save_garch_result
from univariate_garch import apply_fixed_params, fit_garch

if __name__ == "__main__":
    # skip entirely if there's no price data newer than what GARCH already reflects
    # (e.g. weekend/holiday run) — avoids a ~15-20 min no-op
    with engine.connect() as conn:
        latest_price_date = conn.execute(text("SELECT MAX(date) FROM daily_prices")).scalar()
        latest_garch_date = conn.execute(text("SELECT MAX(date) FROM garch_estimates")).scalar()
    if latest_garch_date is not None and latest_price_date is not None and latest_garch_date >= latest_price_date:
        print(f"가격 데이터 최신일({latest_price_date})까지 GARCH 이미 반영됨 — 스킵")
        sys.exit(0)

    # daily job: reuse the params from the last weekly fit_univariate_garch.py run and just
    # replay the GARCH recursion with today's return added (no MLE re-optimization, ~17x faster).
    # tickers with no stored params yet (e.g. brand new listings) fall back to a full fit.
    tickers = fetch_all_tickers()

    skipped = 0
    fallback_fits = 0
    bar = tqdm(tickers, unit="ticker")
    for ticker in bar:
        bar.set_description(ticker)
        returns = fetch_returns(ticker)
        params = fetch_latest_params(ticker)

        if params is not None:
            result = apply_fixed_params(returns, params)
        else:
            result = fit_garch(returns)
            if result is not None:
                fallback_fits += 1

        if result is None:
            skipped += 1
            continue

        if params is not None:
            save_garch_estimates(ticker, result) 
        else:
            save_garch_result(ticker, date.today(), result)

    print(f"done. updated {len(tickers) - skipped}/{len(tickers)} tickers, skipped {skipped}, fallback full-fits {fallback_fits}")
