import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))

import numpy as np
import pandas as pd

from cv import make_update_fold
from sequence_models import DEVICE, SEQUENCE_MODELS
from sequence_training_data import build_sequence_training_set

OUTPUT = Path(__file__).resolve().parent.parent / "output" / "sequence_oof_result.parquet"


def run() -> None:
    existing = pd.read_parquet(OUTPUT)
    last_covered = existing["as_of_date"].max()
    last_fold = int(existing["fold"].max())
    print(f"existing sequence OOF covers up to {last_covered} (fold {last_fold})", flush=True)
    print(f"device: {DEVICE}", flush=True)

    print("building sequence training set...", flush=True)
    t0 = time.time()
    dates_arr, tickers_arr, X, y = build_sequence_training_set()
    unique_dates = sorted(np.unique(dates_arr))
    print(f"sequence training set ready in {time.time() - t0:.0f}s: X={X.shape}", flush=True)

    fold = make_update_fold(unique_dates, last_covered)
    if fold is None:
        print("nothing new yet -- sequence OOF is already current", flush=True)
        return

    train_mask = np.isin(dates_arr, fold["train"])
    val_mask = np.isin(dates_arr, fold["val"])
    print(
        f"catch-up fold {last_fold + 1}: train={train_mask.sum()} val={val_mask.sum()} "
        f"({fold['val'][0]}..{fold['val'][-1]})",
        flush=True,
    )

    new_rows = []
    for name, factory in SEQUENCE_MODELS.items():
        t0 = time.time()
        model = factory()
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[val_mask])
        new_rows.append(
            pd.DataFrame(
                {
                    "as_of_date": dates_arr[val_mask],
                    "ticker": tickers_arr[val_mask],
                    "target": y[val_mask],
                    "model": name,
                    "pred": preds,
                    "fold": last_fold + 1,
                }
            )
        )
        print(f"  [{name}] {time.time() - t0:.1f}s", flush=True)

    updated = pd.concat([existing] + new_rows, ignore_index=True)
    updated.to_parquet(OUTPUT)
    print(f"appended {sum(len(r) for r in new_rows)} rows, saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
