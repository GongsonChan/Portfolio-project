import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine

_SHARE_CLASS_SUFFIX = re.compile(r"(\d*우선주|보통주)$")  # KRX naming convention: a company's
# common and preferred shares get separate tickers (e.g. 005930/005935 = 삼성전자보통주/
# 삼성전자1우선주) -- stripping this suffix recovers the shared issuer name so they can be
# capped together (see fetch_issuer_groups). Moved here from 05_black_litterman when top-N/cap
# post-processing moved to this module.


def fetch_latest_portfolio(label: str = "default") -> tuple[pd.Series, object]:
    """05_black_litterman's full-universe, uncapped BL output -- the raw input this module
    filters and caps. Returns (weights indexed by ticker, the as_of_date it was computed on)."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM portfolio_weights WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return pd.Series(dtype=float), None
        df = pd.read_sql(
            text("SELECT ticker, weight FROM portfolio_weights WHERE label = :label AND as_of_date = :as_of_date"),
            conn,
            params={"label": label, "as_of_date": as_of_date},
        )
    return pd.Series(df["weight"].values, index=df["ticker"]), as_of_date


def fetch_issuer_groups(tickers: list[str]) -> dict[str, str]:
    """Maps each ticker to its issuer group -- same base company name with the KRX common/
    preferred suffix stripped, or the ticker itself if there's no such suffix (US tickers,
    or a KRX name that doesn't match the pattern). A ticker whose name doesn't match the
    pattern is its own singleton group."""
    query = text("SELECT ticker, name FROM securities WHERE ticker = ANY(:tickers)")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"tickers": tickers})
    groups = {}
    for t, name in zip(df["ticker"], df["name"]):
        base = _SHARE_CLASS_SUFFIX.sub("", name or "") if name else ""
        groups[t] = base if base else t
    for t in tickers:
        groups.setdefault(t, t)
    return groups


def fetch_sector_info(tickers: list[str]) -> pd.DataFrame:
    query = text("SELECT ticker, name, market, sector FROM securities WHERE ticker = ANY(:tickers)")
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"tickers": tickers})


def save_sectors(sector_map: dict[str, str]) -> None:
    """One-time cache write: a ticker's sector rarely changes, so this is only called for
    tickers sectors.ensure_sectors couldn't find a valid GICS sector for already."""
    with engine.begin() as conn:
        for ticker, sector in sector_map.items():
            conn.execute(
                text("UPDATE securities SET sector = :sector WHERE ticker = :ticker"),
                {"ticker": ticker, "sector": sector},
            )


def save_final_portfolio(as_of_date, weights: pd.Series, label: str = "default") -> None:
    """weights: Series indexed by ticker, top-N filtered + issuer-group capped + renormalized."""
    delete_sql = text("DELETE FROM final_portfolio WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        "INSERT INTO final_portfolio (as_of_date, label, ticker, weight) VALUES (:as_of_date, :label, :ticker, :weight)"
    )
    rows = [{"as_of_date": as_of_date, "label": label, "ticker": t, "weight": float(w)} for t, w in weights.items()]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, rows)
