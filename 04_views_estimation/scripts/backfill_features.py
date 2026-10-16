import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqdm import tqdm

from panel import build_feature_panel
from repository import save_features

YEARS_BACK = 10
SNAPSHOT_INTERVAL_DAYS = 7  # weekly snapshots are enough for training; daily_update_features.py fills in since


def weekly_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=SNAPSHOT_INTERVAL_DAYS)


def run(start: date | None = None, end: date | None = None) -> None:
    end = end or date.today()
    start = start or end.replace(year=end.year - YEARS_BACK)

    dates = list(weekly_dates(start, end))
    bar = tqdm(dates, unit="week")
    for d in bar:
        bar.set_description(d.isoformat())
        panel = build_feature_panel(d.isoformat()) 
        if panel.empty:
            continue
        save_features(panel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(
        date.fromisoformat(args.start) if args.start else None,
        date.fromisoformat(args.end) if args.end else None,
    )
