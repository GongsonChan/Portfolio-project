import time

import numpy as np
import pandas as pd

from cv import make_walk_forward_folds
from sequence_models import DEVICE, SEQUENCE_MODELS
from sequence_training_data import build_sequence_training_set


def run_sequence_oof(
    start: str | None = None,
    end: str | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Walk-forward OOF predictions for the sequence models, using the exact same fold
    boundaries (cv.make_walk_forward_folds) as oof.py's factor-model run so the two OOF tables
    line up by (as_of_date, ticker) and can be merged before meta-model training."""
    if model_names is None:
        model_names = list(SEQUENCE_MODELS.keys()) 

    print(f"device: {DEVICE}", flush=True)
    print("building sequence training set...", flush=True)
    t0 = time.time()
    dates_arr, tickers_arr, X, y = build_sequence_training_set(start, end) 
    unique_dates = sorted(np.unique(dates_arr))
    folds = make_walk_forward_folds(unique_dates)
    print(
        f"sequence training set ready in {time.time() - t0:.0f}s: X={X.shape}, {len(folds)} folds",
        flush=True,
    )

    oof_rows = []
    for fold_idx, fold in enumerate(folds):
        train_mask = np.isin(dates_arr, fold["train"])
        val_mask = np.isin(dates_arr, fold["val"])
        if not train_mask.any() or not val_mask.any():
            continue

        for name in model_names:
            t0 = time.time()
            model = SEQUENCE_MODELS[name]()
            model.fit(X[train_mask], y[train_mask])
            preds = model.predict(X[val_mask])
            oof_rows.append(
                pd.DataFrame(
                    {
                        "as_of_date": dates_arr[val_mask],
                        "ticker": tickers_arr[val_mask],
                        "target": y[val_mask],
                        "model": name,
                        "pred": preds,
                        "fold": fold_idx,
                    }
                )
            )
            print(
                f"fold {fold_idx + 1}/{len(folds)} [{name}] train={train_mask.sum()} "
                f"val={val_mask.sum()} {time.time() - t0:.1f}s",
                flush=True,
            )

    return pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
