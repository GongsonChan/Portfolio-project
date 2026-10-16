import numpy as np
import pandas as pd

NON_FEATURE_COLS = ("as_of_date", "ticker", "target", "model", "pred", "fold")


def build_meta_training_set(factor_oof_path: str, sequence_oof_path: str) -> pd.DataFrame:
    """Merges the factor-model and sequence-model OOF tables (oof.py / sequence_oof.py output)
    into one wide meta-training table: one row per (as_of_date, ticker), one column per base
    model's OOF prediction, plus the shared target. The two pipelines don't share the exact
    same walk-forward fold boundaries (sequence_training_data.py has an extra per-date filter
    -- sequence availability -- that the factor pipeline doesn't), so this merges by actual
    (as_of_date, ticker) values rather than trusting either side's `fold` column; rows missing
    any base model's prediction (covered by only one pipeline) are dropped, since the
    meta-model needs a complete input vector.
    """
    factor_oof = pd.read_parquet(factor_oof_path)
    sequence_oof = pd.read_parquet(sequence_oof_path)
    combined = pd.concat([factor_oof, sequence_oof], ignore_index=True)

    # Sanity check: target is a pure function of (as_of_date, ticker) via labels.compute_targets,
    # so it must be identical (up to float precision) regardless of which pipeline computed it.
    # Uses a tolerance, not exact equality: sequence_training_data.py casts targets to float32
    # for the torch models, so the same value legitimately differs from the factor pipeline's
    # float64 by ~1e-9 -- that's not real drift, just precision, and exact nunique() flagged
    # every single row for it. A genuine mismatch (e.g. one side re-run before a labels.py
    # change) would show up as a difference many orders of magnitude larger than that.
    target_check = combined.groupby(["as_of_date", "ticker"])["target"].agg(lambda s: s.max() - s.min())
    inconsistent = int((target_check > 1e-4).sum())
    if inconsistent:
        print(
            f"warning: {inconsistent} (as_of_date, ticker) rows have target values that differ "
            "by more than float-precision noise between the factor and sequence OOF tables -- "
            "one of them is probably stale"
        )

    # .pivot (not pivot_table) deliberately: it raises on a duplicate (index, columns) pair
    # instead of silently averaging, which would otherwise mask a fold-overlap bug.
    preds = combined.pivot(index=["as_of_date", "ticker"], columns="model", values="pred")
    targets = combined.groupby(["as_of_date", "ticker"])["target"].first()

    wide = preds.join(targets).dropna().reset_index()
    return wide
