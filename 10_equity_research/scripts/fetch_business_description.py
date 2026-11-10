import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dart_client import fetch_document, find_latest_annual_report
from filing_parser import extract_business_description
from repository import fetch_corp_codes_for_tickers, fetch_final_portfolio_tickers, save_business_description

# "사업 개요" report section's data source -- the actual "사업의 내용" free text, not
# company.json's structured fields (see fetch_company_overview.py for that). Meant to be rerun
# only occasionally (new annual 사업보고서 filed, ~once/year per company), not daily -- see
# DESIGN.md's caching note. Filtered to 사업보고서 specifically (not 분기/반기보고서) since that's
# where this chapter is most complete.


def run(label: str = "default") -> None:
    tickers = fetch_final_portfolio_tickers(label)
    if not tickers:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(tickers)} holdings")

    corp_codes = fetch_corp_codes_for_tickers(tickers)
    n_ok = 0
    for ticker in tickers:
        corp_code = corp_codes.get(ticker)
        if corp_code is None:
            print(f"  [{ticker}] no DART corp_code, skipping")
            continue

        latest = find_latest_annual_report(corp_code)
        if latest is None:
            print(f"  [{ticker}] no 사업보고서 filing found in the last 450 days, skipping")
            continue

        raw_zip = fetch_document(latest["rcept_no"])
        section_text = extract_business_description(raw_zip, latest["rcept_no"])
        if section_text is None:
            print(f"  [{ticker}] {latest['report_nm']} ({latest['rcept_dt']}) -- extraction FAILED (see filing_parser.py)")
            continue

        save_business_description(
            corp_code,
            {
                "rcept_no": latest["rcept_no"],
                "report_nm": latest["report_nm"],
                "section_text": section_text,
                "fetched_at": date.today(),
            },
        )
        n_ok += 1
        print(f"  [{ticker}] {latest['report_nm']} ({latest['rcept_dt']}) -- extracted {len(section_text)} chars")

    print(f"done -- {n_ok}/{len(tickers)} succeeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
