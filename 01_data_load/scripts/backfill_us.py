import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

from loaders.repository import upsert_daily_prices, upsert_securities_full
from loaders.us_universe import get_universe
from loaders.yfinance_client import download_prices, get_shares_outstanding
from loaders.yfinance_transform import price_df_to_rows, universe_row_to_security

YEARS_BACK = 10
LOOKBACK_BUFFER_DAYS = 10  # extra days fetched (but not saved) so pct_change() has a prior close
# to diff against even on a narrow 1-day gap-fill window; otherwise the first fetched day's
# daily_return comes out NaN even though a valid previous close already exists in the DB.


def run(start: str | None = None, end: str | None = None) -> None:
    end = end or date.today().isoformat()
    if start is None:
        end_date = date.fromisoformat(end)
        start = end_date.replace(year=end_date.year - YEARS_BACK).isoformat()
    start_date = date.fromisoformat(start)

    universe = get_universe()
    print(f"universe: {len(universe)} tickers")
    upsert_securities_full([universe_row_to_security(r) for r in universe])

    tickers = [r["ticker"] for r in universe]
    download_start = (start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)).isoformat()
    prices = download_prices(tickers, download_start, end)
    available = set(prices.columns.get_level_values(0))

    for ticker in tqdm(tickers, unit="ticker"):
        if ticker not in available:
            continue
        shares = get_shares_outstanding(ticker)
        rows = price_df_to_rows(ticker, prices[ticker], shares)
        rows = [r for r in rows if r["date"] >= start_date]
        upsert_daily_prices(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(args.start, args.end)
