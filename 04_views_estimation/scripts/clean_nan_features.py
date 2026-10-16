import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from repository import engine

# save_features used to let pandas' None->NaN auto-coercion (for a missing feature value, e.g.
# no garch_estimates row yet) flow straight into the INSERT. psycopg2 binds Python float('nan')
# as the literal NUMERIC value 'NaN', not SQL NULL, so every "missing feature" row was silently
# stored as NaN instead of NULL. Fixed going forward in repository.save_features; this is the
# one-time repair of rows already written that way.
NAN_COLUMNS = ["cond_vol", "liquidity", "amihud", "bb_pctb", "volume_surge"]

if __name__ == "__main__":
    with engine.begin() as conn:
        for col in NAN_COLUMNS:
            result = conn.execute(text(f"UPDATE ml_features SET {col} = NULL WHERE {col}::text = 'NaN'"))
            print(f"{col}: nulled {result.rowcount} rows")
