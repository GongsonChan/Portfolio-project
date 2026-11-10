import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from naver_client import fetch_news
from news_assessment import assess_with_consistency, filter_relevant_articles
from repository import fetch_final_portfolio_holdings, save_news_articles, save_news_assessment

N_ARTICLES = 10  # per ticker, AFTER filtering -- plenty to characterize "what's going on
# recently" without bloating the LLM prompt; sort="date" in naver_client already biases toward
# the newest
FETCH_POOL_SIZE = 30  # fetched BEFORE filtering -- some conglomerates' sports teams share the
# exact company name (e.g. 삼성화재's V-리그 volleyball team), flooding a plain name search with
# game coverage; fetch extra headroom so filter_relevant_articles still leaves N_ARTICLES after
# dropping that noise (see news_assessment.py)
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

    today = date.today()
    companies = []
    for h in holdings:
        raw_articles = fetch_news(h["name"], display=FETCH_POOL_SIZE)
        relevant = filter_relevant_articles(raw_articles)
        articles = relevant[:N_ARTICLES]
        n_dropped = len(raw_articles) - len(relevant)
        companies.append({"ticker": h["ticker"], "name": h["name"], "articles": articles})
        save_news_articles(today, label, h["ticker"], articles)
        dropped_note = f", {n_dropped} dropped as irrelevant" if n_dropped else ""
        print(f"  [{h['ticker']}] {h['name']}: {len(articles)} relevant article(s){dropped_note}")

    print(f"assessing {len(companies)} companies via LLM ({N_RUNS}x self-consistency)...")
    assessments = assess_with_consistency(companies, n_runs=N_RUNS)

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
