import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))

import joblib
import numpy as np
import pandas as pd
import torch

from model_registry import fetch_latest_models
from repository import fetch_latest_complete_date_per_ticker, fetch_latest_universe, fetch_omega, save_views
from sequence import build_sequence_batch
from sequence_models import ARCHITECTURES, DEVICE
from training_data import load_feature_panel

NON_FEATURE_COLS = ("as_of_date", "ticker")


def _load_factor_or_meta_model(model_type: str, model_name: str | None = None):
    """model_name=None picks the row with the best oof_corr for that type -- only meaningful
    for 'meta', where multiple candidate names can coexist in the registry (whichever won is
    the current champion); factor/sequence each only ever have one row per name anyway."""
    rows = fetch_latest_models(model_type=model_type)
    if model_name is not None:
        rows = [r for r in rows if r["model_name"] == model_name]
    if not rows:
        raise RuntimeError(f"no {model_type} model found in model_registry (name={model_name})")
    best = max(rows, key=lambda r: r["oof_corr"])
    return best["model_name"], joblib.load(best["file_path"])


def load_models():
    factor_rows = fetch_latest_models(model_type="factor")
    factor_models = {r["model_name"]: joblib.load(r["file_path"]) for r in factor_rows}

    sequence_rows = fetch_latest_models(model_type="sequence")
    sequence_models = {}
    for r in sequence_rows:
        model = ARCHITECTURES[r["model_name"]]().to(DEVICE)
        model.load_state_dict(torch.load(r["file_path"], map_location=DEVICE))
        model.eval()
        sequence_models[r["model_name"]] = model

    meta_name, meta_model = _load_factor_or_meta_model("meta")
    print(f"meta model in use: {meta_name}")

    return factor_models, sequence_models, meta_model


def _predict_group(as_of_date: str, tickers: list[str], factor_models, sequence_models, meta_model, meta_feature_cols) -> pd.Series:
    """Predict Q for one batch of tickers that all share the same as_of_date."""
    panel = load_feature_panel(start=as_of_date, end=as_of_date)
    panel = panel[panel["ticker"].isin(tickers)].set_index("ticker")
    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    # psycopg2 hands back NUMERIC columns as Decimal objects (dtype=object); build_training_set's
    # concat across many dates happens to upcast these to float64 along the way, but a single-date
    # read here doesn't, so LightGBM's dtype check rejects it unless cast explicitly.
    panel[feature_cols] = panel[feature_cols].astype(float)

    sequences = build_sequence_batch(as_of_date)  # unmodified, computed for the whole eligible
    # pool and filtered down here rather than touching sequence.py, so this never risks changing
    # behavior the training/OOF pipeline depends on.

    # cond_vol comes straight from garch_estimates (03_sigma_estimation), so a ticker 03 excludes
    # from the covariance matrix (no GARCH fit -- e.g. too little history) shows up here as a
    # NULL cond_vol. A Q prediction for a ticker BL has no covariance row for is useless (can't
    # be optimized against), so this must stay excluded in lockstep with 03's DCC/covariance step
    # rather than quietly getting an imputed/NaN-tolerant prediction. Tickers are grouped by their
    # own latest cond_vol-complete date before reaching here, so this mainly catches a missing
    # sequence window (e.g. too-new listing) rather than a GARCH lag.
    usable = [
        t for t in tickers if t in panel.index and t in sequences and pd.notna(panel.loc[t, "cond_vol"])
    ]
    skipped = len(tickers) - len(usable)
    if skipped:
        print(f"  {as_of_date}: skipping {skipped}/{len(tickers)} tickers missing features or a full sequence window")
    if not usable:
        return pd.Series(dtype=float, name="q")

    factor_preds = pd.DataFrame(
        {name: model.predict(panel.loc[usable, feature_cols]) for name, model in factor_models.items()},
        index=usable,
    )

    seq_X = np.stack([sequences[t] for t in usable]).astype(np.float32)
    seq_X_t = torch.as_tensor(seq_X, dtype=torch.float32).to(DEVICE)
    sequence_preds = pd.DataFrame(index=usable)
    with torch.no_grad():
        for name, model in sequence_models.items():
            sequence_preds[name] = model(seq_X_t).cpu().numpy()

    meta_input = pd.concat([factor_preds, sequence_preds], axis=1)[meta_feature_cols]
    return pd.Series(meta_model.predict(meta_input), index=usable, name="q")


def run(as_of_date: str | None, label: str = "default") -> None:
    universe = fetch_latest_universe(label)
    print(f"universe ({label}): {len(universe)} tickers")

    factor_models, sequence_models, meta_model = load_models()
    factor_names = sorted(factor_models.keys())
    sequence_names = sorted(sequence_models.keys())
    meta_feature_cols = sorted(factor_names + sequence_names)  # matches meta_data.py's pivot()
    # column order (alphabetical by model name), which is what the meta-model was fit on.

    if as_of_date is not None:
        # Explicit --date: one shared date for the whole universe (e.g. a one-off backtest-style
        # check), same as before.
        groups = {as_of_date: universe}
    else:
        # Default: each ticker predicted as of ITS OWN latest cond_vol-complete date, not one
        # shared date picked by whichever market/ticker happens to be furthest ahead -- see
        # fetch_latest_complete_date_per_ticker's docstring. KOSPI/KOSDAQ running a day behind
        # US no longer drops the whole Korean side of the universe out of that day's views.
        per_ticker_date = fetch_latest_complete_date_per_ticker(universe)
        missing = [t for t in universe if t not in per_ticker_date]
        if missing:
            print(f"skipping {len(missing)} universe tickers with no cond_vol-complete date at all")
        groups = {}
        for t, d in per_ticker_date.items():
            groups.setdefault(str(d), []).append(t)

    # Ω is a per-ticker, weekly-refreshed value (omega_estimates/compute_omega.py), not tied to
    # any of the as_of_date groups above -- fetched once for the whole universe rather than
    # per group.
    omega_map = pd.Series(fetch_omega(universe, label))

    total_saved = 0
    for group_date, tickers in sorted(groups.items()):
        q = _predict_group(group_date, tickers, factor_models, sequence_models, meta_model, meta_feature_cols)
        if q.empty:
            continue
        save_views(group_date, q, label, omega=omega_map)
        print(f"saved {len(q)} views for {group_date} (label={label})")
        total_saved += len(q)
    print(f"total: {total_saved} views saved across {len(groups)} date group(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="default: each ticker's own latest cond_vol-complete date")
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()
    run(args.date, args.label)
