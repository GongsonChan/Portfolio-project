import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

from loaders.krx_client import fetch_daily_trade
from loaders.repository import upsert_daily_prices, upsert_securities_stub
from loaders.krx_transform import trade_row_to_price, trade_row_to_security_stub

MARKETS = ["KOSPI", "KOSDAQ"]
YEARS_BACK = 10
SLEEP_SEC = 0.2


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def run(start: date | None = None, end: date | None = None) -> None:
    end = end or date.today()
    start = start or end.replace(year=end.year - YEARS_BACK)

    days = list(weekdays(start, end))
    bar = tqdm(days, unit="day")
    for d in bar:
        bas_dd = d.strftime("%Y%m%d")
        bar.set_description(bas_dd)
        for market in MARKETS:
            rows = fetch_daily_trade(market, bas_dd) 
            if not rows:
                continue
            upsert_securities_stub([trade_row_to_security_stub(r, market) for r in rows])
            upsert_daily_prices([trade_row_to_price(r) for r in rows])
            time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(
        date.fromisoformat(args.start) if args.start else None,
        date.fromisoformat(args.end) if args.end else None,
    )
