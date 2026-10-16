import pandas as pd

EMBARGO_WEEKS = 5  # weekly-spaced dates to skip between train end and val start; 5*7=35 calendar
                   # days comfortably covers the 21-trading-day (~29 calendar day) label horizon,
                   # so a validation date's target window can't overlap any training date's window
MIN_TRAIN_WEEKS = 150  # ~3 years minimum history before the first fold is trusted to train on
VAL_WEEKS = 12  # ~3 months of validation dates per fold


def make_walk_forward_folds(
    dates: list,
    min_train_weeks: int = MIN_TRAIN_WEEKS,
    embargo_weeks: int = EMBARGO_WEEKS,
    val_weeks: int = VAL_WEEKS,
) -> list[dict]:
    """dates: as_of_dates (already weekly-thinned), any order. Returns a list of
    {"train": [...], "val": [...]} folds, expanding-window style: each fold's train set is
    everything up to its own val window minus the embargo gap, with consecutive folds' val
    windows butted right up against each other (no gap between them) so every date past the
    initial training window gets exactly one OOF prediction -- the embargo only ever shows up
    between a fold's own train end and its own val start, never between one fold's val and the
    next's, so it doesn't leave a permanent "never validated" gap in the walk-forward."""
    dates = sorted(dates)
    folds = []
    val_start = min_train_weeks + embargo_weeks
    while True:
        val_end = val_start + val_weeks
        if val_end > len(dates):
            break
        train_end = val_start - embargo_weeks
        folds.append({"train": dates[:train_end], "val": dates[val_start:val_end]})
        val_start = val_end
    return folds


def make_update_fold(
    dates: list, last_covered_date, embargo_weeks: int = EMBARGO_WEEKS
) -> dict | None:
    """Incremental counterpart to make_walk_forward_folds, for update_oof.py-style scripts
    (mirrors the fit_*.py/update_*.py split used by 01-03): given the full candidate date list
    and the most recent date an existing OOF file already covers, returns ONE catch-up fold
    {"train": [...], "val": [...]} spanning every date past that point -- or None if nothing
    new is available yet. Unlike a walk-forward fold, val isn't capped at VAL_WEEKS: it's just
    "whatever's newly available," so this handles both a large one-time backlog and a routine
    weekly single-date trickle with the same code path, without re-running any already-computed
    fold.

    val starts immediately after last_covered_date, with NO extra embargo gap there -- same
    "no gap between one fold's val and the next's" rule as make_walk_forward_folds, since
    last_covered_date was already safely embargoed against when IT was validated. The embargo
    only applies where it always does: between this new fold's own train_end and its own
    val_start, which is why train_end comes out embargo_weeks short of last_covered_date (the
    same rolling lag make_walk_forward_folds has between consecutive calls -- it catches up on
    the next update once newer dates push last_idx forward)."""
    dates = sorted(dates)
    last_idx = dates.index(last_covered_date)
    val_start = last_idx + 1
    if val_start >= len(dates):
        return None
    train_end = val_start - embargo_weeks
    return {"train": dates[:train_end], "val": dates[val_start:]}
