import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from db import engine

MAX_PLAUSIBLE_DAILY_RETURN = 0.5  # matches 04_views_estimation/src/repository.py's
# MAX_PLAUSIBLE_DAILY_RETURN -- KRX's regulatory daily limit is +-30%; US has no such cap but
# nothing that passes any reasonable liquidity filter legitimately gaps >50% in a single day
# without a halt/split/data artifact behind it. Found via: ticker 052670 sat flat at a stale
# price with volume=0 for ~7 weeks (trading halt), then "returned" +299 (+29,900%) the day
# trading resumed. A full-history scan turned up 395 such rows across 298 tickers (many showing
# an identical +300% that smells like a price-limit/vendor artifact on rights/warrant-like
# tickers, not regular common stock) -- re-backfilling all of history to fix this at the source
# isn't practical, so this nulls daily_return in place for any row that fails the sanity check.
# Re-running finds 0 rows once clean (idempotent). 04's own repository.py filters the same way
# at read time for defense in depth, but 03_sigma_estimation reads daily_prices directly with
# no such filter, so this DB-level cleanup is what actually protects GARCH fitting.


def run() -> None:
    query = text(
        """
        UPDATE daily_prices
        SET daily_return = NULL
        WHERE ABS(daily_return) > :threshold
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query, {"threshold": MAX_PLAUSIBLE_DAILY_RETURN})
        print(
            f"nulled {result.rowcount} implausible daily_return rows "
            f"(|daily_return| > {MAX_PLAUSIBLE_DAILY_RETURN})"
        )


if __name__ == "__main__":
    run()
