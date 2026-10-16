import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from backfill_krx import run
from db import engine
from sync_krx_base_info import sync_base_info

if __name__ == "__main__":
    sync_base_info()

    # MIN across markets (not a single overall MAX(date)) -- KOSPI/KOSDAQ can drift apart (one
    # market's fetch silently failing/erroring for a stretch while the other keeps advancing),
    # and starting from the more-advanced market's "next day" would permanently skip the
    # behind market's actual gap. Starting from the most-behind market instead just means a
    # few redundant (harmless, upsert-safe) re-fetches for whichever market is already caught up.
    with engine.connect() as conn:
        last_dates = conn.execute(
            text(
                """
                SELECT s.market, MAX(dp.date) AS last_date
                FROM daily_prices dp JOIN securities s ON dp.ticker = s.ticker
                WHERE s.market IN ('KOSPI', 'KOSDAQ')
                GROUP BY s.market
                """
            )
        ).fetchall()

    if last_dates:
        last_date = min(row.last_date for row in last_dates)
        start = last_date + timedelta(days=1)
    else:
        start = date.today()
    run(start=start, end=date.today())
