import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))

import pandas as pd

from cv import make_update_fold
from meta_data import build_meta_training_set
from meta_models import META_MODELS

DEFAULT_FACTOR = Path(__file__).resolve().parent.parent / "output" / "oof_result.parquet"
DEFAULT_SEQUENCE = Path(__file__).resolve().parent.parent / "output" / "sequence_oof_result.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
NON_FEATURE_COLS = ("as_of_date", "ticker", "target")


def run() -> None:
    """Incremental counterpart to run_meta_oof.py (mirrors update_oof.py/update_sequence_oof.py).
    Missing before today: run_meta_oof.py only ever does a full walk-forward re-run, so the Ω
    calculation (TRAIN.md step 8) had no way to get fresh meta-level OOF rows without redoing
    the whole history every time. One catch-up fold per META_MODELS candidate, same as the
    factor/sequence updaters -- each candidate's own existing parquet is the only source of
    truth for "how far it's already covered", no shared/global date across candidates."""
    print("merging factor + sequence OOF...", flush=True)
    meta_df = build_meta_training_set(str(DEFAULT_FACTOR), str(DEFAULT_SEQUENCE))
    dates = sorted(meta_df["as_of_date"].unique())
    feature_cols = [c for c in meta_df.columns if c not in NON_FEATURE_COLS]

    for name in META_MODELS:
        path = OUTPUT_DIR / f"meta_oof_{name}.parquet"
        if not path.exists():
            print(f"[{name}] no existing OOF file -- run run_meta_oof.py first", flush=True)
            continue

        existing = pd.read_parquet(path)
        last_covered = existing["as_of_date"].max()
        last_fold = int(existing["fold"].max())

        fold = make_update_fold(dates, last_covered)
        if fold is None:
            print(f"[{name}] nothing new yet -- covers up to {last_covered}", flush=True)
            continue

        train_df = meta_df[meta_df["as_of_date"].isin(fold["train"])]
        val_df = meta_df[meta_df["as_of_date"].isin(fold["val"])]

        model = META_MODELS[name]()
        model.fit(train_df[feature_cols], train_df["target"])
        preds = model.predict(val_df[feature_cols])
        new_rows = val_df[["as_of_date", "ticker", "target"]].assign(
            model=name, pred=preds, fold=last_fold + 1
        )

        updated = pd.concat([existing, new_rows], ignore_index=True)
        updated.to_parquet(path)
        print(
            f"[{name}] catch-up fold {last_fold + 1}: appended {len(new_rows)} rows "
            f"({fold['val'][0]}..{fold['val'][-1]})",
            flush=True,
        )


if __name__ == "__main__":
    run()
