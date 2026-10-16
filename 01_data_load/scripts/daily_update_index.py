import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from backfill_index import run
from db import engine
from loaders.index_data import INDEX_TICKERS

if __name__ == "__main__":
    tickers = list(INDEX_TICKERS.keys())
    # MIN across each index ticker's OWN latest date (not one shared MAX across both) -- same
    # per-entity drift bug as daily_update_krx.py/daily_update_us.py: if ^KS11 and ^GSPC ever
    # fall out of sync, a shared MAX would permanently skip whichever one fell behind.
    with engine.connect() as conn:
        last_date = conn.execute(
            text(
                """
                SELECT MIN(max_date) FROM (
                    SELECT ticker, MAX(date) AS max_date FROM daily_prices
                    WHERE ticker = ANY(:tickers) GROUP BY ticker
                ) t
                """
            ),
            {"tickers": tickers},
        ).scalar()

    start = (last_date + timedelta(days=1)).isoformat() if last_date else None
    run(start=start, end=date.today().isoformat())
