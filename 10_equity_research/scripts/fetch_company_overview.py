import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dart_client import fetch_company_overview, parse_dart_date
from repository import fetch_corp_codes_for_tickers, fetch_final_portfolio_tickers, save_company_overview

# Smoke test for the DART pipe end-to-end AND the first real building block of the "사업 개요"
# report section: for each current holding, looks up its DART corp_code and caches the basic
# company overview. Company overview barely changes (CEO, industry code, founding date), so
# this is meant to be rerun only occasionally -- see 10_equity_research/DESIGN.md's caching note
# -- not wired into daily_update.py.


def run(label: str = "default") -> None:
    tickers = fetch_final_portfolio_tickers(label)
    if not tickers:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(tickers)} holdings")

    corp_codes = fetch_corp_codes_for_tickers(tickers)
    missing = [t for t in tickers if t not in corp_codes]
    if missing:
        print(f"  no DART corp_code found for: {missing} (run update_corp_codes.py first, or these may be newly listed)")

    for ticker in tickers:
        corp_code = corp_codes.get(ticker)
        if corp_code is None:
            continue
        overview = fetch_company_overview(corp_code)
        if overview is None:
            print(f"  [{ticker}] no DART company record for corp_code={corp_code}")
            continue
        save_company_overview(
            corp_code,
            {
                "stock_code": overview.get("stock_code") or ticker,
                "corp_name": overview.get("corp_name"),
                "ceo_name": overview.get("ceo_nm"),
                "induty_code": overview.get("induty_code"),
                "est_dt": parse_dart_date(overview.get("est_dt")),
                "address": overview.get("adres"),
                "homepage_url": overview.get("hm_url") or None,
                "fetched_at": date.today(),
            },
        )
        print(f"  [{ticker}] {overview.get('corp_name')} -- CEO={overview.get('ceo_nm')} est={overview.get('est_dt')}")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
