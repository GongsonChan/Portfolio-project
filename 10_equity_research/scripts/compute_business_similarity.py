import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repository import fetch_business_descriptions, fetch_final_portfolio_tickers, save_business_similarity
from similarity import compute_similarity_matrix

# Stage 2 of 10_equity_research -- business-model similarity via embeddings, complementing
# 09_risk_management's return-correlation-based component VaR with a different axis of
# concentration (see DESIGN.md's "왜 필요한가"). Depends on fetch_business_description.py having
# already cached each holding's "사업의 내용" text; same occasional-refresh cadence (not daily).


def run(label: str = "default") -> None:
    tickers = fetch_final_portfolio_tickers(label)
    if not tickers:
        raise RuntimeError(f"no final_portfolio snapshot found for label={label}")
    print(f"portfolio: {len(tickers)} holdings")

    texts = fetch_business_descriptions(tickers)
    missing = [t for t in tickers if t not in texts]
    if missing:
        print(f"  no cached 사업의 내용 for: {missing} (run fetch_business_description.py first)")
    if len(texts) < 2:
        raise RuntimeError(f"only {len(texts)} holding(s) have a cached business description -- need at least 2")

    print(f"embedding {len(texts)} descriptions via ko-sroberta-multitask (first run downloads the model)...")
    sim = compute_similarity_matrix(texts)
    save_business_similarity(date.today(), label, sim)

    print("\nmost similar pairs (excluding self):")
    pairs = []
    for i in sim.index:
        for j in sim.columns:
            if i < j:
                pairs.append((i, j, sim.loc[i, j]))
    pairs.sort(key=lambda p: p[2], reverse=True)
    for i, j, s in pairs[:10]:
        print(f"  {i} <-> {j}: {s:.3f}")

    print("\ndone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.label)
