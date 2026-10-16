import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from db import engine

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "exports"


def export_securities(out_dir: Path) -> None:
    df = pd.read_sql("SELECT * FROM securities ORDER BY ticker", engine)
    path = out_dir / "securities_export.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"securities: {len(df)} rows -> {path}")


def export_daily_prices(out_dir: Path, mode: str, value) -> None:
    if mode == "ticker":
        query = "SELECT * FROM daily_prices WHERE ticker = %(ticker)s ORDER BY date"
        df = pd.read_sql(query, engine, params={"ticker": value})
        path = out_dir / f"daily_prices_{value}.csv"
    elif mode == "full":
        df = pd.read_sql("SELECT * FROM daily_prices ORDER BY date, ticker", engine)
        path = out_dir / "daily_prices_export.csv"
    elif mode == "days":
        query = """
            SELECT * FROM daily_prices
            WHERE date >= (SELECT max(date) FROM daily_prices) - %(days)s
            ORDER BY date, ticker
        """
        df = pd.read_sql(query, engine, params={"days": value})
        path = out_dir / f"daily_prices_last{value}d.csv"
    else:
        query = "SELECT * FROM daily_prices ORDER BY date DESC, ticker LIMIT %(rows)s"
        df = pd.read_sql(query, engine, params={"rows": value})
        path = out_dir / f"daily_prices_last{value}rows.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"daily_prices: {len(df)} rows -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--full", action="store_true", help="export the entire daily_prices table (can be large)")
    parser.add_argument("--days", type=int, help="recent N days of daily_prices to export")
    parser.add_argument("--ticker", type=str, help="export full history for a single ticker")
    parser.add_argument("--rows", type=int, default=5000, help="row count to export when no other filter is given (default 1000)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_securities(out_dir)
    if args.ticker:
        export_daily_prices(out_dir, "ticker", args.ticker)
    elif args.full:
        export_daily_prices(out_dir, "full", None)
    elif args.days:
        export_daily_prices(out_dir, "days", args.days)
    else:
        export_daily_prices(out_dir, "rows", args.rows)
