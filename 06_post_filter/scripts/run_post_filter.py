import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from repository import fetch_issuer_groups, fetch_latest_portfolio, save_final_portfolio
from sectors import ensure_sectors

DEFAULT_TOP_N = 10  # exposed as a parameter (not hardcoded) -- same "default with override"
# pattern as 02_filter's `weights` and 04's `label`, for a future web admin
DEFAULT_MAX_WEIGHT = 0.20  # issuer-group weight cap
DEFAULT_MAX_PER_SECTOR = 2  # sector COUNT cap -- how many names from the same GICS sector can
# land in the final top-N (separate concern from the weight cap above: this limits how many
# slots a sector takes, not how much weight one issuer gets)


def _select_top_n(weights: pd.Series, top_n: int, max_per_sector: int, sector_map: dict[str, str]) -> pd.Series:
    """Walk down the full candidate list (already sorted descending) and take the first top_n
    that don't bust the per-sector slot limit -- skipped names are just skipped, not re-added
    later, so a sector-dense universe (see 05's "삼성생명 빼고는 거의 다 반도체" result) still
    ends up diversified instead of silently topping out early."""
    selected = []
    sector_counts: dict[str, int] = {}
    for ticker in weights.index:
        sector = sector_map.get(ticker, "Unknown")
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) >= top_n:
            break
    return weights.loc[selected]


def _cap_and_renormalize(weights: pd.Series, cap: float, groups: dict[str, str]) -> pd.Series:
    """Iteratively clip any ISSUER GROUP's total over `cap` (not just one ticker -- a common/
    preferred pair like 005930/005935 is the same company's market-cap risk twice over) and
    redistribute the excess proportionally among the not-yet-capped groups, repeating until
    stable. Plain renormalization after a top-N cut can push a group that was exactly at the
    cap back over it (the cut tickers' weight has to go somewhere), so this re-enforces the cap
    on the final, delivered weights rather than just on 05's pre-cut output."""
    w = weights.copy()
    group_of = pd.Series(groups).reindex(w.index)
    for _ in range(len(w)):
        group_sum = w.groupby(group_of).sum()
        over_groups = group_sum[group_sum > cap].index
        if len(over_groups) == 0:
            break
        excess = (group_sum[over_groups] - cap).sum()
        over_mask = group_of.isin(over_groups)
        w[over_mask] *= (cap / group_sum[group_of[over_mask]]).values
        under_mask = ~over_mask
        w[under_mask] += excess * (w[under_mask] / w[under_mask].sum())
    return w


def run(
    label: str = "default",
    top_n: int = DEFAULT_TOP_N,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
) -> None:
    weights, source_date = fetch_latest_portfolio(label)
    if weights.empty:
        raise RuntimeError(f"no portfolio_weights snapshot found for label={label}")
    print(f"05 output: {len(weights)} tickers as of {source_date}")

    candidates = weights[weights > 0].sort_values(ascending=False)
    sector_map = ensure_sectors(list(candidates.index))

    top = _select_top_n(candidates, top_n, max_per_sector, sector_map)
    groups = fetch_issuer_groups(list(top.index))
    final_weights = _cap_and_renormalize(top / top.sum(), max_weight, groups)

    today = date.today()
    save_final_portfolio(today, final_weights, label)
    print(f"saved {len(final_weights)} final weights for {today} (label={label}, source 05 date={source_date})")
    for t, w in final_weights.sort_values(ascending=False).items():
        print(f"  {t} ({sector_map.get(t, '?')}): {w:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="default")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--max-per-sector", type=int, default=DEFAULT_MAX_PER_SECTOR)
    args = parser.parse_args()
    run(args.label, args.top_n, args.max_weight, args.max_per_sector)
