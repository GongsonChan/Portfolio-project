import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from cv import make_update_fold
from models import FACTOR_MODELS
from training_data import build_training_set

OUTPUT = Path(__file__).resolve().parent.parent / "output" / "oof_result.parquet"
NON_FEATURE_COLS = ("as_of_date", "ticker", "target")


def run() -> None:
    existing = pd.read_parquet(OUTPUT)
    last_covered = existing["as_of_date"].max()
    last_fold = int(existing["fold"].max())
    print(f"existing OOF covers up to {last_covered} (fold {last_fold})", flush=True)

    print("building training set...", flush=True)
    t0 = time.time()
    full = build_training_set()
    dates = sorted(full["as_of_date"].unique())
    print(f"training set ready in {time.time() - t0:.0f}s: {full.shape}", flush=True)

    fold = make_update_fold(dates, last_covered) 
    if fold is None:
        print("nothing new yet -- OOF is already current", flush=True)
        return

    feature_cols = [c for c in full.columns if c not in NON_FEATURE_COLS]
    train_df = full[full["as_of_date"].isin(fold["train"])]
    val_df = full[full["as_of_date"].isin(fold["val"])]
    print(
        f"catch-up fold {last_fold + 1}: train={len(train_df)} val={len(val_df)} "
        f"({fold['val'][0]}..{fold['val'][-1]})",
        flush=True,
    )

    new_rows = []
    for name, factory in FACTOR_MODELS.items():
        t0 = time.time()
        model = factory()
        model.fit(train_df[feature_cols], train_df["target"])
        preds = model.predict(val_df[feature_cols])
        new_rows.append(
            val_df[["as_of_date", "ticker", "target"]].assign(
                model=name, pred=preds, fold=last_fold + 1
            )
        )
        print(f"  [{name}] {time.time() - t0:.1f}s", flush=True)

    updated = pd.concat([existing] + new_rows, ignore_index=True)
    updated.to_parquet(OUTPUT)
    print(f"appended {sum(len(r) for r in new_rows)} rows, saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    run()
