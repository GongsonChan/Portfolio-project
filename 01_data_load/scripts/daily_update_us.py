import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from backfill_us import run
from db import engine

if __name__ == "__main__":
    # MIN across each ticker's OWN latest date (not a single overall MAX(date) across all US
    # rows) -- same bug as daily_update_krx.py had across markets, just per-ticker here: if a
    # handful of tickers fail a given run (yfinance flakiness) while most succeed, the overall
    # MAX keeps advancing with the majority, so `start` computed from it lands after the
    # stragglers' actual gap and permanently skips them. Starting from the most-behind ticker
    # instead just means a few redundant (harmless, upsert-safe) re-fetches for tickers that
    # were already current.
    with engine.connect() as conn:
        last_date = conn.execute(
            text(
                """
                SELECT MIN(max_date) FROM (
                    SELECT dp.ticker, MAX(dp.date) AS max_date
                    FROM daily_prices dp
                    JOIN securities s ON dp.ticker = s.ticker
                    WHERE s.market = 'US'
                    GROUP BY dp.ticker
                ) t
                """
            )
        ).scalar()

    start = (last_date + timedelta(days=1)).isoformat() if last_date else None
    run(start=start, end=date.today().isoformat())
