import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from cv import make_walk_forward_folds
from models import FACTOR_MODELS
from training_data import build_training_set

NON_FEATURE_COLS = ("as_of_date", "ticker", "target")


def run_oof(
    start: str | None = None,
    end: str | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Walk-forward OOF predictions for the given factor models (default: every model in the
    registry -- pass a subset to mirror the future admin on/off toggle described in TRAIN.md).
    Returns one row per (as_of_date, ticker, model) with columns as_of_date, ticker, target,
    model, pred, fold."""
    if model_names is None:
        model_names = list(FACTOR_MODELS.keys()) 

    print("building training set...", flush=True)
    t0 = time.time()
    full = build_training_set(start, end)
    feature_cols = [c for c in full.columns if c not in NON_FEATURE_COLS]
    dates = sorted(full["as_of_date"].unique())
    folds = make_walk_forward_folds(dates)
    print(f"training set ready in {time.time() - t0:.0f}s: {full.shape}, {len(folds)} folds", flush=True)

    oof_rows = []
    for fold_idx, fold in enumerate(folds):
        train_df = full[full["as_of_date"].isin(fold["train"])]
        val_df = full[full["as_of_date"].isin(fold["val"])]
        if train_df.empty or val_df.empty:
            continue

        for name in model_names:
            t0 = time.time()
            model = FACTOR_MODELS[name]()
            model.fit(train_df[feature_cols], train_df["target"])
            preds = model.predict(val_df[feature_cols])
            oof_rows.append(
                val_df[["as_of_date", "ticker", "target"]].assign(model=name, pred=preds, fold=fold_idx)
            )
            print(
                f"fold {fold_idx + 1}/{len(folds)} [{name}] train={len(train_df)} val={len(val_df)} "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    return pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
