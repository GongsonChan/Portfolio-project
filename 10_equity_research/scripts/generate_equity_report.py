import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dart_client import fetch_financial_statements, find_latest_annual_report
from financial_statements import extract_key_figures, parse_bsns_year
from report_writer import write_reports
from repository import (
    fetch_business_descriptions,
    fetch_corp_codes_for_tickers,
    fetch_final_portfolio_holdings,
    save_equity_report,
)

REPRT_CODE = "11011"  # 사업보고서 (annual) -- matches find_latest_annual_report's filter

# Replaces the earlier "dump the raw 사업의 내용 text" approach with a proper 3-part report:
# 기업개요(5문장 이내)/재무제표(핵심 수치 + 3문장 요약)/최신동향, each shown in its own panel on
# the web page -- see DESIGN.md. Depends on fetch_business_description.py having already cached
# each holding's 사업의 내용 text. Occasional-refresh cadence (annual report + its financials
# only change once a year per company), not daily.


def run(label: str = "default") -> None:
    holdings = fetch_final_portfolio_holdings(label)
    if not holdings:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(holdings)} holdings")

    tickers = [h["ticker"] for h in holdings]
    corp_codes = fetch_corp_codes_for_tickers(tickers)
    business_texts = fetch_business_descriptions(tickers)

    companies = []
    meta_by_ticker = {}
    for h in holdings:
        ticker, name = h["ticker"], h["name"]
        corp_code = corp_codes.get(ticker)
        if corp_code is None:
            print(f"  [{ticker}] no DART corp_code, skipping")
            continue
        business_text = business_texts.get(ticker)
        if business_text is None:
            print(f"  [{ticker}] no cached 사업의 내용 (run fetch_business_description.py first), skipping")
            continue

        latest = find_latest_annual_report(corp_code)
        bsns_year = parse_bsns_year(latest["report_nm"]) if latest else None
        if bsns_year is None:
            print(f"  [{ticker}] couldn't determine 사업연도 from filing list, skipping")
            continue

        rows = fetch_financial_statements(corp_code, bsns_year, REPRT_CODE, fs_div="CFS")
        fs_div = "CFS"
        if not rows:
            # No subsidiaries -> no consolidated statement -- fall back to 개별재무제표.
            rows = fetch_financial_statements(corp_code, bsns_year, REPRT_CODE, fs_div="OFS")
            fs_div = "OFS"
        if not rows:
            print(f"  [{ticker}] no financial statement data (CFS or OFS) for {bsns_year}, skipping")
            continue

        figures = extract_key_figures(rows)
        companies.append({"ticker": ticker, "name": name, "business_text": business_text, "figures": figures})
        meta_by_ticker[ticker] = {"bsns_year": bsns_year, "fs_div": fs_div, "figures": figures, "corp_code": corp_code}
        print(f"  [{ticker}] {name} -- {bsns_year}년 {fs_div} 재무제표 확보, LLM 리포트 생성 대기")

    if not companies:
        raise RuntimeError("no holdings had both a cached business description and financial statements")

    print(f"\nwriting reports for {len(companies)} companies via LLM...")
    reports = write_reports(companies)

    today = date.today()
    for ticker, meta in meta_by_ticker.items():
        r = reports.get(ticker)
        if r is None:
            print(f"  [{ticker}] LLM response missing this ticker, not saved")
            continue
        save_equity_report(
            meta["corp_code"],
            {
                "bsns_year": meta["bsns_year"],
                "reprt_code": REPRT_CODE,
                "fs_div": meta["fs_div"],
                "overview_summary": r.overview_summary,
                "trends_summary": r.trends_summary,
                "financial_summary": r.financial_summary,
                **meta["figures"],
                "fetched_at": today,
            },
        )
        print(f"  [{ticker}] saved")

    print("\ndone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
