import pandas as pd
from sqlalchemy import text

from labels import compute_targets
from repository import engine

MIN_SPACING_DAYS = 7  # training rows must come from dates at least this far apart (see FEATURE_DESIGN.md)


def load_feature_panel(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM ml_features WHERE 1=1"
    params = {}
    if start:
        query += " AND as_of_date >= :start"
        params["start"] = start
    if end:
        query += " AND as_of_date <= :end"
        params["end"] = end
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def _thin_to_weekly(dates) -> list:
    """Greedily keep dates at least MIN_SPACING_DAYS apart, so daily_update_features.py's
    daily accumulation doesn't skew training density relative to the weekly historical backfill."""
    kept = []
    last = None
    for d in sorted(dates):
        if last is None or (pd.Timestamp(d) - pd.Timestamp(last)).days >= MIN_SPACING_DAYS:
            kept.append(d)
            last = d
    return kept


def build_training_set(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Joins the stored feature panel with forward-return targets, thinned to weekly spacing.
    as_of_dates too recent to have a realized target are automatically dropped (empty join)."""
    features = load_feature_panel(start, end)
    dates = _thin_to_weekly(features["as_of_date"].unique())

    rows = []
    for as_of_date in dates:
        day_features = features[features["as_of_date"] == as_of_date]
        targets = compute_targets(str(as_of_date)) 
        if targets.empty:
            continue
        merged = day_features.merge(targets, left_on="ticker", right_index=True, how="inner")
        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
