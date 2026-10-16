import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loaders.krx_client import fetch_base_info, latest_trading_day
from loaders.repository import upsert_securities_full
from loaders.krx_transform import base_info_to_security

MARKETS = ["KOSPI", "KOSDAQ"] 


def sync_base_info() -> None:
    bas_dd = latest_trading_day()
    for market in MARKETS:
        rows = fetch_base_info(market, bas_dd)
        upsert_securities_full([base_info_to_security(r, market) for r in rows])
        print(f"{market}: {len(rows)} securities synced")


if __name__ == "__main__":
    sync_base_info()
