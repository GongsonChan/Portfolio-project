import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from model_registry import fetch_latest_models
from repository import save_omega

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TRAILING_DAYS = 365  # TRAIN.md step 8: per-ticker Ω from the trailing ~1 year of OOF data
MIN_SAMPLES_FOR_FULL_TRUST = 30  # shrinkage prior strength: a ticker with this many trailing
# OOF rows gets its Ω roughly half from its own variance, half from the pooled one; a ticker
# with far fewer (TRAIN.md flags ~52/year as already noisy) gets pulled most of the way to the
# pooled estimate instead of trusting a near-single-digit sample count


def run(label: str = "default") -> None:
    rows = fetch_latest_models(model_type="meta")
    if not rows:
        raise RuntimeError("no meta model found in model_registry")
    champion = max(rows, key=lambda r: r["oof_corr"])
    name = champion["model_name"]
    print(f"meta champion: {name}")

    oof = pd.read_parquet(OUTPUT_DIR / f"meta_oof_{name}.parquet")
    # Anchor the trailing window to the OOF data's OWN latest date, not today's wall-clock date
    # -- OOF accumulates weekly and embargoed, so it's always somewhat behind "today", and using
    # date.today() here would just shrink the window for no real reason (same lesson as
    # 04's daily feature panel: bound windows by what the data actually has, not the calendar).
    latest = pd.Timestamp(oof["as_of_date"].max())
    cutoff = latest - pd.Timedelta(days=TRAILING_DAYS)
    recent = oof[pd.to_datetime(oof["as_of_date"]) > cutoff].copy()
    if recent.empty:
        raise RuntimeError("no OOF rows in the trailing window -- not enough OOF history yet")
    recent["err"] = recent["target"] - recent["pred"]

    pooled_var = recent["err"].var()
    per_ticker = recent.groupby("ticker")["err"].agg(var="var", n_samples="count")
    per_ticker["var"] = per_ticker["var"].fillna(pooled_var)  # a single-sample ticker has an
    # undefined (NaN) sample variance -- fall back fully to the pooled estimate for it rather
    # than let a NaN silently reach the DB (see save_features' NaN-vs-NULL lesson from today).

    w = per_ticker["n_samples"] / (per_ticker["n_samples"] + MIN_SAMPLES_FOR_FULL_TRUST)
    per_ticker["omega"] = w * per_ticker["var"] + (1 - w) * pooled_var

    # computed_at = today's actual run date, not the OOF data's own latest date -- it answers
    # "when was this Ω last (re)computed", same meaning as model_registry.trained_at. The
    # OOF window itself (cutoff..latest) is printed below but not stored; omega_estimates
    # tracks current state per ticker, not a time series (see omega.sql).
    save_omega(per_ticker[["omega", "n_samples"]], computed_at=date.today(), label=label)
    print(
        f"saved Ω for {len(per_ticker)} tickers "
        f"(OOF window: {cutoff.date()}..{latest.date()}, pooled var={pooled_var:.6f})"
    )


if __name__ == "__main__":
    run()
