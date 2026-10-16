import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import bindparam, text

from db import engine

# 2026-09-22/23/24: ALL 518 US tickers got daily_return=NULL despite having valid adj_close
# values already stored -- some historical daily_update_us.py run(s) evidently fetched a
# window that didn't include a prior close to diff against (buffer protection failed for that
# run), even though adj_close itself is fine. Found via compute_targets() returning 0 rows for
# every as_of_date whose 21-trading-day forward window touches this gap. One-time repair:
# recompute daily_return straight from the adj_close already in the DB, no re-fetch needed.
DATES = ["2026-09-22", "2026-09-23", "2026-09-24"]

_SQL = text(
    """
    WITH ordered AS (
        SELECT ticker, date, adj_close,
               LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev_adj_close
        FROM daily_prices
        WHERE ticker IN (SELECT ticker FROM securities WHERE market = 'US')
    )
    UPDATE daily_prices dp
    SET daily_return = (o.adj_close / o.prev_adj_close) - 1
    FROM ordered o
    WHERE dp.ticker = o.ticker AND dp.date = o.date
      AND dp.date IN :dates AND o.prev_adj_close IS NOT NULL
    """
).bindparams(bindparam("dates", expanding=True))

if __name__ == "__main__":
    with engine.begin() as conn:
        result = conn.execute(_SQL, {"dates": DATES})
        print(f"recomputed daily_return for {result.rowcount} rows across {DATES}")
