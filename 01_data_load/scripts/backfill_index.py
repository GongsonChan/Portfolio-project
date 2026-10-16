import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loaders.index_data import INDEX_TICKERS, index_price_df_to_rows, index_to_security
from loaders.repository import upsert_daily_prices, upsert_securities_full
from loaders.yfinance_client import download_prices

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

    tickers = list(INDEX_TICKERS.keys())
    upsert_securities_full([index_to_security(t) for t in tickers])

    download_start = (start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)).isoformat()
    prices = download_prices(tickers, download_start, end)
    for ticker in tickers:
        rows = index_price_df_to_rows(ticker, prices[ticker])
        rows = [r for r in rows if r["date"] >= start_date]
        upsert_daily_prices(rows)
        print(f"{ticker}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(args.start, args.end)
