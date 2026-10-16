import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from panel import build_feature_panel
from repository import engine, save_features

RECENCY_WINDOW_DAYS = 30  # "still actively traded" cutoff for the straggler check below --
# securities.is_active is never actually maintained (always True, delisted_date always NULL in
# practice), so recency of real daily_prices rows is the only reliable signal.

if __name__ == "__main__":
    with engine.connect() as conn:
        # Same per-entity drift bug class as 01_data_load's daily_update_krx.py/us.py/index.py --
        # MIN across each ticker's OWN latest cond_vol-complete date, not a single MAX across the
        # whole table. A shared MAX lets the majority of tickers (the ones GARCH already reached)
        # drag the resume pointer past any individual straggler, permanently orphaning that
        # ticker's features for the dates it was still behind on -- found via run_inference.py
        # silently dropping some KOSPI/KOSDAQ tickers to a much older date than the rest of the
        # universe. Restricted to recently-traded tickers only: a handful of long-delisted
        # tickers never get a new cond_vol again (their daily_prices stopped years ago), and an
        # unfiltered MIN would permanently pin the resume point to 2018 trying to "catch them up".
        recency_cutoff = date.today() - timedelta(days=RECENCY_WINDOW_DAYS)
        last_date = conn.execute(
            text(
                """
                SELECT MIN(max_date) FROM (
                    SELECT mf.ticker, MAX(mf.as_of_date) AS max_date
                    FROM ml_features mf
                    WHERE mf.cond_vol IS NOT NULL
                      AND mf.ticker IN (
                          SELECT ticker FROM daily_prices GROUP BY ticker HAVING MAX(date) >= :cutoff
                      )
                    GROUP BY mf.ticker
                ) t
                """
            ),
            {"cutoff": recency_cutoff},
        ).scalar()
        latest_garch_date = conn.execute(text("SELECT MAX(date) FROM garch_estimates")).scalar()

    today = date.today()
    end = min(today, latest_garch_date) if latest_garch_date else today
    start = (last_date + timedelta(days=1)) if last_date else end

    if start > end:
        print(f"GARCH 최신일({latest_garch_date})까지 이미 반영됨 — 스킵")
    else:
        d = start
        while d <= end:
            as_of = d.isoformat()
            panel = build_feature_panel(as_of)
            if panel.empty:
                print(f"{as_of}: no data (휴장일?)")
            else:
                save_features(panel)
                print(f"{as_of}: saved {len(panel)} tickers")
            d += timedelta(days=1)
