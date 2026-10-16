from sqlalchemy import text

from db import engine

_STUB_UPSERT_SQL = text("""
    INSERT INTO securities (ticker, name, market, currency, is_active)
    VALUES (:ticker, :name, :market, :currency, TRUE)
    ON CONFLICT (ticker) DO UPDATE SET
        name = EXCLUDED.name,
        market = EXCLUDED.market,
        updated_at = now()
""")

_FULL_UPSERT_SQL = text("""
    INSERT INTO securities (ticker, name, market, sector, currency, is_active, listed_date)
    VALUES (:ticker, :name, :market, :sector, :currency, TRUE, :listed_date)
    ON CONFLICT (ticker) DO UPDATE SET
        name = EXCLUDED.name,
        market = EXCLUDED.market,
        sector = EXCLUDED.sector,
        listed_date = EXCLUDED.listed_date,
        updated_at = now()
""")

_PRICE_UPSERT_SQL = text("""
    INSERT INTO daily_prices (ticker, date, close_price, adj_close, market_cap, volume, daily_return)
    VALUES (:ticker, :date, :close_price, :adj_close, :market_cap, :volume, :daily_return)
    ON CONFLICT (ticker, date) DO UPDATE SET
        close_price = EXCLUDED.close_price,
        adj_close = EXCLUDED.adj_close,
        market_cap = EXCLUDED.market_cap,
        volume = EXCLUDED.volume,
        daily_return = EXCLUDED.daily_return
""")


def upsert_securities_stub(rows: list[dict]) -> None:
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(_STUB_UPSERT_SQL, rows)


def upsert_securities_full(rows: list[dict]) -> None:
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(_FULL_UPSERT_SQL, rows)


def upsert_daily_prices(rows: list[dict]) -> None:
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(_PRICE_UPSERT_SQL, rows)
