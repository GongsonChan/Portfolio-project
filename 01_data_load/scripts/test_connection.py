import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import inspect

from db import engine

with engine.connect() as conn:
    inspector = inspect(engine)
    print("connected. tables:", inspector.get_table_names())
