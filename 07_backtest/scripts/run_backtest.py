import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from backtest import run_backtest, summarize
from repository import save_backtest_results

DEFAULT_OOF = (
    Path(__file__).resolve().parent.parent.parent / "04_views_estimation" / "output" / "meta_oof_lightgbm_leaves7.parquet"
)


def run(oof_path: str, label: str = "default") -> None:
    oof = pd.read_parquet(oof_path)
    print(f"replaying {oof['as_of_date'].nunique()} fold dates from {oof_path}", flush=True)

    results = run_backtest(oof)
    if results.empty:
        raise RuntimeError("no fold produced a usable backtest period -- check Omega/Sigma data availability")

    save_backtest_results(results, label)
    stats = summarize(results)
    print(f"\nsaved {len(results)} period returns (label={label})")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=str, default=str(DEFAULT_OOF))
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.oof, args.label)
