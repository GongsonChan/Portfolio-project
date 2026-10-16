import pandas as pd

from cv import make_walk_forward_folds
from meta_models import META_MODELS

NON_FEATURE_COLS = ("as_of_date", "ticker", "target")


def run_meta_oof(
    meta_df: pd.DataFrame,
    model_name: str,
    min_train_weeks: int = 50,
    embargo_weeks: int = 5,
    val_weeks: int = 12,
) -> pd.DataFrame:
    """Walk-forward OOF for one meta-model candidate, over the merged base-model-prediction
    table (meta_data.build_meta_training_set). Uses a much smaller min_train_weeks than the
    base-model folds (cv.MIN_TRAIN_WEEKS=150) -- the meta-model's input is just a handful of
    numbers per row, not 31 raw features or a 252-day sequence, so it doesn't need years of
    data to "warm up" before the first fold is trustworthy. embargo_weeks stays the same since
    the target is still the same overlapping 21-trading-day forward return."""
    feature_cols = [c for c in meta_df.columns if c not in NON_FEATURE_COLS]
    dates = sorted(meta_df["as_of_date"].unique())
    folds = make_walk_forward_folds(
        dates, min_train_weeks=min_train_weeks, embargo_weeks=embargo_weeks, val_weeks=val_weeks
    )

    oof_rows = []
    for fold_idx, fold in enumerate(folds):
        train_df = meta_df[meta_df["as_of_date"].isin(fold["train"])]
        val_df = meta_df[meta_df["as_of_date"].isin(fold["val"])]
        if train_df.empty or val_df.empty:
            continue

        model = META_MODELS[model_name]()
        model.fit(train_df[feature_cols], train_df["target"])
        preds = model.predict(val_df[feature_cols])
        oof_rows.append(
            val_df[["as_of_date", "ticker", "target"]].assign(model=model_name, pred=preds, fold=fold_idx)
        )

    return pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
