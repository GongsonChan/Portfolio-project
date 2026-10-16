from sqlalchemy import text

from repository import engine

_UPSERT_SQL = text(
    """
    INSERT INTO model_registry (model_name, model_type, label, trained_at, oof_corr, file_path)
    VALUES (:model_name, :model_type, :label, :trained_at, :oof_corr, :file_path)
    ON CONFLICT (model_name, model_type, label) DO UPDATE SET
        trained_at = EXCLUDED.trained_at, oof_corr = EXCLUDED.oof_corr, file_path = EXCLUDED.file_path
    """
)


def save_model_meta(
    model_name: str,
    model_type: str,
    trained_at,
    oof_corr: float | None,
    file_path: str,
    label: str = "default",
) -> None:
    with engine.begin() as conn:
        conn.execute(
            _UPSERT_SQL,
            {
                "model_name": model_name,
                "model_type": model_type,
                "label": label,
                "trained_at": trained_at,
                "oof_corr": oof_corr,
                "file_path": file_path,
            },
        )


def fetch_latest_models(model_type: str | None = None, label: str = "default") -> list[dict]:
    """Returns the current (model_name, model_type, trained_at, oof_corr, file_path) row for
    every model of the given type (or all types if omitted) under this label -- there's only
    ever one row per (model_name, model_type, label) since save_model_meta upserts in place."""
    query = "SELECT model_name, model_type, trained_at, oof_corr, file_path FROM model_registry WHERE label = :label"
    params = {"label": label}
    if model_type:
        query += " AND model_type = :model_type"
        params["model_type"] = model_type
    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]
