import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

from repository import fetch_all_tickers, fetch_returns, save_garch_result
from univariate_garch import fit_garch

if __name__ == "__main__":
    # weekly (weekend) job: full MLE re-estimation of (mu, omega, alpha, beta) for every ticker.
    # for the daily update that reuses these params, see update_univariate_garch.py
    tickers = fetch_all_tickers()
    fit_date = date.today()

    skipped = 0
    bar = tqdm(tickers, unit="ticker")
    for ticker in bar:
        bar.set_description(ticker)
        returns = fetch_returns(ticker)
        result = fit_garch(returns)
        if result is None:
            skipped += 1
            continue
        save_garch_result(ticker, fit_date, result)

    print(f"done. fit {len(tickers) - skipped}/{len(tickers)} tickers, skipped {skipped}")
