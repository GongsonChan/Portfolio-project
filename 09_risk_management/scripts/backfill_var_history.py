import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

from sqlalchemy import text

from db import engine

# One-time (idempotent) seed for basel_capital's 60-day rolling VaR average: var_backtest_daily
# already has a point-in-time-honest parametric VaR series (from run_var_backtest.py replaying
# 07_backtest's run_one_fold for each historical period -- the ACTUAL portfolio that would have
# been held on each day, not today's portfolio projected backward). Copying its tail into
# var_estimates gives basel_capital real history instead of degenerating to n=1 on cold start.
# ON CONFLICT DO NOTHING -- never overwrites a day compute_daily_risk_report.py already saved
# live, and safe to rerun.


def run(label: str = "default", confidence: float = 0.99, days: int = 60) -> None:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO var_estimates (as_of_date, label, method, confidence, var_value, es_value)
                SELECT date, :label, 'parametric', confidence, var_value, NULL
                FROM (
                    SELECT date, confidence, var_value FROM var_backtest_daily
                    WHERE label = :label AND confidence = :confidence
                    ORDER BY date DESC LIMIT :days
                ) recent
                ON CONFLICT (as_of_date, label, method, confidence) DO NOTHING
                """
            ),
            {"label": label, "confidence": confidence, "days": days},
        )
    print(
        f"backfilled up to {days} day(s) of parametric {confidence:.0%} VaR history "
        f"from var_backtest_daily into var_estimates (label={label}, {result.rowcount} row(s) inserted)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    run(args.label, args.confidence, args.days)
