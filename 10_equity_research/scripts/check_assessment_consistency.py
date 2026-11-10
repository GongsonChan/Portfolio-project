import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from naver_client import fetch_news
from news_assessment import assess_via_llm
from repository import fetch_final_portfolio_holdings

N_ARTICLES = 10
N_RUNS = 5

# Diagnostic only -- does NOT write to equity_news_assessment (that table holds the production
# daily result; this would just overwrite it with throwaway repeated-run data). Checks whether
# assess_via_llm's risk_level is stable across repeated calls on the SAME news input, since
# there's no ground truth to backtest a qualitative LLM judgment against (unlike 09's VaR, which
# has Kupiec) -- consistency across reruns is the one thing that IS checkable here.


def run(label: str = "default") -> None:
    holdings = fetch_final_portfolio_holdings(label)
    if not holdings:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(holdings)} holdings -- fetching news once, fixed across all {N_RUNS} runs")

    companies = []
    for h in holdings:
        articles = fetch_news(h["name"], display=N_ARTICLES)
        companies.append({"ticker": h["ticker"], "name": h["name"], "articles": articles})

    results: dict[str, list[str]] = {c["ticker"]: [] for c in companies}
    n_failed_runs = 0
    for run_idx in range(N_RUNS):
        print(f"\n--- run {run_idx + 1}/{N_RUNS} ---")
        try:
            assessments = assess_via_llm(companies)
        except Exception as e:
            n_failed_runs += 1
            print(f"  run failed, skipping: {e}")
            continue
        for ticker, a in assessments.items():
            results[ticker].append(a.risk_level)
            print(f"  [{ticker}] {a.risk_level}")
    if n_failed_runs:
        print(f"\n({n_failed_runs}/{N_RUNS} runs failed outright and are excluded from the summary below)")

    print("\n=== consistency summary ===")
    n_stable = 0
    for ticker, levels in results.items():
        counts = Counter(levels)
        stable = len(counts) == 1
        n_stable += stable
        flag = "STABLE" if stable else "VARIED"
        print(f"  [{flag}] {ticker}: {dict(counts)}")

    print(f"\n{n_stable}/{len(results)} tickers gave the identical risk_level across all {N_RUNS} runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
