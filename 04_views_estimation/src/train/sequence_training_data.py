import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import text

from labels import compute_targets
from repository import engine
from sequence import build_sequence_batch
from training_data import _thin_to_weekly


def _fetch_dates(start: str | None, end: str | None) -> list:
    """Same source table as the factor pipeline (ml_features) so, after the same weekly
    thinning, both pipelines walk the identical as_of_date list -- required for their OOF
    outputs to line up by (as_of_date, ticker) later."""
    query = "SELECT DISTINCT as_of_date FROM ml_features WHERE 1=1"
    params = {}
    if start:
        query += " AND as_of_date >= :start"
        params["start"] = start
    if end:
        query += " AND as_of_date <= :end"
        params["end"] = end
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).fetchall()
    return [r[0] for r in rows]


def build_sequence_training_set(
    start: str | None = None, end: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds raw excess-return sequences (sequence.py) paired with forward-return targets
    (labels.py), one row per (as_of_date, ticker). Unlike ml_features, sequences aren't
    precomputed/stored, so this calls build_sequence_batch per date -- much slower than the
    factor pipeline's load_feature_panel (a single table read).
    Returns (dates, tickers, X, y): X is (N, window) float32, y is (N,) float32.
    """
    dates = _thin_to_weekly(_fetch_dates(start, end)) 

    all_dates, all_tickers, all_X, all_y = [], [], [], []
    for as_of_date in dates:
        sequences = build_sequence_batch(str(as_of_date))
        targets = compute_targets(str(as_of_date))
        if not sequences or targets.empty:
            continue
        common = targets.index.intersection(sequences.keys())
        for ticker in common:
            all_dates.append(as_of_date)
            all_tickers.append(ticker)
            all_X.append(sequences[ticker])
            all_y.append(targets[ticker])

    return (
        np.array(all_dates),
        np.array(all_tickers),
        np.asarray(all_X, dtype=np.float32),
        np.asarray(all_y, dtype=np.float32),
    )
