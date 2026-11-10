import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from naver_client import fetch_news
from news_assessment import assess_with_consistency
from repository import fetch_final_portfolio_holdings, save_news_assessment

N_ARTICLES = 10  # per ticker -- plenty to characterize "what's going on recently" without
# bloating the LLM prompt; sort="date" in naver_client already biases toward the newest
N_RUNS = 5  # self-consistency majority vote on risk_level -- see news_assessment.py's
# assess_with_consistency and DESIGN.md's "정성적 위험도 등급의 run-to-run 불안정성"

# This is the genuinely daily-refresh half of 10_equity_research's report (see DESIGN.md's
# caching note) -- news changes every day, unlike the DART-sourced company overview/filings
# half, which only needs occasional re-fetching. Meant to be added to daily_update.py once this
# module is otherwise complete (not wired in yet -- still under active development).


def run(label: str = "default") -> None:
    holdings = fetch_final_portfolio_holdings(label)
    if not holdings:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(holdings)} holdings")

    companies = []
    for h in holdings:
        articles = fetch_news(h["name"], display=N_ARTICLES)
        companies.append({"ticker": h["ticker"], "name": h["name"], "articles": articles})
        print(f"  [{h['ticker']}] {h['name']}: {len(articles)} recent article(s)")

    print(f"assessing {len(companies)} companies via LLM ({N_RUNS}x self-consistency)...")
    assessments = assess_with_consistency(companies, n_runs=N_RUNS)

    today = date.today()
    article_counts = {c["ticker"]: len(c["articles"]) for c in companies}
    for ticker, a in assessments.items():
        save_news_assessment(
            today,
            label,
            ticker,
            {
                "n_articles": article_counts.get(ticker, 0),
                "news_summary": a.news_summary,
                "key_issues": a.key_issues,
                "risk_level": a.risk_level,
                "risk_rationale": a.risk_rationale,
                "n_runs": a.n_runs,
                "n_agree": a.n_agree,
            },
        )
        print(f"  [{ticker}] risk={a.risk_level} ({a.n_agree}/{a.n_runs}) -- {a.key_issues}")

    missing = set(article_counts) - set(assessments)
    if missing:
        print(f"  warning: LLM response missing tickers {sorted(missing)} -- not saved")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
