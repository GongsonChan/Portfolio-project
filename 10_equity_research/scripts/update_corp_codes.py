import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dart_client import fetch_corp_codes
from repository import save_corp_codes

# One-time / occasional utility (not part of daily_update.py/weekly_update.py) -- DART's
# corp_code <-> stock_code master list only changes on new listings/delistings, so there's no
# value in re-downloading it every day. Rerun manually after a new company IPOs if its DART
# lookup comes back empty.


def run() -> None:
    rows = fetch_corp_codes()
    listed = [r for r in rows if r["stock_code"]]
    print(f"fetched {len(rows)} DART corp_code entries ({len(listed)} with a KRX stock_code)")
    save_corp_codes(rows)
    print("saved to dart_corp_codes")


if __name__ == "__main__":
    run()
