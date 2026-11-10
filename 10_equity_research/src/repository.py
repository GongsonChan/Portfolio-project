import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import pandas as pd
from sqlalchemy import text

from db import engine


def save_corp_codes(rows: list[dict]) -> None:
    """Full replace -- DART's corp_code master list is small (~100k rows incl. unlisted) and
    this only runs occasionally (scripts/update_corp_codes.py), so a wipe-then-insert is simpler
    than diffing and is cheap enough at this scale."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM dart_corp_codes"))
        if rows:
            conn.execute(
                text(
                    "INSERT INTO dart_corp_codes (corp_code, corp_name, stock_code, modify_date) "
                    "VALUES (:corp_code, :corp_name, :stock_code, :modify_date)"
                ),
                rows,
            )


def fetch_corp_code(stock_code: str) -> str | None:
    """KRX ticker -> DART corp_code, the lookup every other DART call needs first."""
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT corp_code FROM dart_corp_codes WHERE stock_code = :stock_code"),
            {"stock_code": stock_code},
        ).scalar()


def fetch_corp_codes_for_tickers(stock_codes: list[str]) -> dict[str, str]:
    """Batch version of fetch_corp_code -- ticker -> corp_code map for a whole portfolio."""
    if not stock_codes:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT stock_code, corp_code FROM dart_corp_codes WHERE stock_code = ANY(:stock_codes)"),
            {"stock_codes": stock_codes},
        ).fetchall()
    return {stock_code: corp_code for stock_code, corp_code in rows}


def fetch_final_portfolio_tickers(label: str = "default") -> list[str]:
    """06_post_filter's latest holdings -- same "calculate on the final portfolio, not 05's raw
    universe" convention 09_risk_management already follows (see its DESIGN.md)."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM final_portfolio WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return []
        df = pd.read_sql(
            text("SELECT ticker FROM final_portfolio WHERE label = :label AND as_of_date = :d"),
            conn,
            params={"label": label, "d": as_of_date},
        )
    return list(df["ticker"])


def save_company_overview(corp_code: str, record: dict) -> None:
    insert_sql = text(
        """
        INSERT INTO dart_company_overview
            (corp_code, stock_code, corp_name, ceo_name, induty_code, est_dt, address, homepage_url, fetched_at)
        VALUES
            (:corp_code, :stock_code, :corp_name, :ceo_name, :induty_code, :est_dt, :address, :homepage_url, :fetched_at)
        ON CONFLICT (corp_code) DO UPDATE SET
            stock_code = EXCLUDED.stock_code, corp_name = EXCLUDED.corp_name, ceo_name = EXCLUDED.ceo_name,
            induty_code = EXCLUDED.induty_code, est_dt = EXCLUDED.est_dt, address = EXCLUDED.address,
            homepage_url = EXCLUDED.homepage_url, fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, {**record, "corp_code": corp_code})


def fetch_company_overview(corp_code: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM dart_company_overview WHERE corp_code = :corp_code"),
            {"corp_code": corp_code},
        ).mappings().first()
    return dict(row) if row else None


def fetch_final_portfolio_holdings(label: str = "default") -> list[dict]:
    """06_post_filter's latest holdings joined with the common display name (securities.name) --
    used as the Naver news query term instead of DART's corp_name, which carries a "(주)" suffix
    unlike the name this project already uses everywhere else (08_web's holdings table, etc.)."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM final_portfolio WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return []
        df = pd.read_sql(
            text(
                """
                SELECT fp.ticker, s.name FROM final_portfolio fp JOIN securities s ON fp.ticker = s.ticker
                WHERE fp.label = :label AND fp.as_of_date = :d
                """
            ),
            conn,
            params={"label": label, "d": as_of_date},
        )
    return df.to_dict("records")


def save_news_assessment(as_of_date, label: str, ticker: str, record: dict) -> None:
    """record: {"n_articles", "news_summary", "key_issues", "risk_level", "risk_rationale",
    "n_runs", "n_agree"}."""
    insert_sql = text(
        """
        INSERT INTO equity_news_assessment
            (as_of_date, label, ticker, n_articles, news_summary, key_issues, risk_level, risk_rationale, n_runs, n_agree)
        VALUES
            (:as_of_date, :label, :ticker, :n_articles, :news_summary, :key_issues, :risk_level, :risk_rationale, :n_runs, :n_agree)
        ON CONFLICT (as_of_date, label, ticker) DO UPDATE SET
            n_articles = EXCLUDED.n_articles, news_summary = EXCLUDED.news_summary, key_issues = EXCLUDED.key_issues,
            risk_level = EXCLUDED.risk_level, risk_rationale = EXCLUDED.risk_rationale,
            n_runs = EXCLUDED.n_runs, n_agree = EXCLUDED.n_agree
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, {**record, "as_of_date": as_of_date, "label": label, "ticker": ticker})


def save_business_description(corp_code: str, record: dict) -> None:
    """record: {"rcept_no", "report_nm", "section_text", "fetched_at"}."""
    insert_sql = text(
        """
        INSERT INTO dart_business_description (corp_code, rcept_no, report_nm, section_text, fetched_at)
        VALUES (:corp_code, :rcept_no, :report_nm, :section_text, :fetched_at)
        ON CONFLICT (corp_code) DO UPDATE SET
            rcept_no = EXCLUDED.rcept_no, report_nm = EXCLUDED.report_nm,
            section_text = EXCLUDED.section_text, fetched_at = EXCLUDED.fetched_at
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, {**record, "corp_code": corp_code})


def fetch_business_descriptions(stock_codes: list[str]) -> dict[str, str]:
    """ticker -> "사업의 내용" text, for whichever tickers already have one cached (see
    fetch_business_description.py) -- tickers missing a cached description are simply absent
    from the result, not an error."""
    if not stock_codes:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT c.stock_code, d.section_text FROM dart_business_description d
                JOIN dart_corp_codes c ON d.corp_code = c.corp_code
                WHERE c.stock_code = ANY(:stock_codes) AND d.section_text IS NOT NULL
                """
            ),
            {"stock_codes": stock_codes},
        ).fetchall()
    return {stock_code: section_text for stock_code, section_text in rows}


def save_business_similarity(as_of_date, label: str, sim: pd.DataFrame) -> None:
    """sim: (ticker x ticker) DataFrame, same full-matrix convention as
    03_sigma_estimation.save_covariance."""
    delete_sql = text("DELETE FROM business_similarity WHERE as_of_date = :as_of_date AND label = :label")
    insert_sql = text(
        """
        INSERT INTO business_similarity (as_of_date, label, ticker_i, ticker_j, similarity)
        VALUES (:as_of_date, :label, :ticker_i, :ticker_j, :similarity)
        """
    )
    rows = [
        {"as_of_date": as_of_date, "label": label, "ticker_i": i, "ticker_j": j, "similarity": float(sim.loc[i, j])}
        for i in sim.index
        for j in sim.columns
    ]
    with engine.begin() as conn:
        conn.execute(delete_sql, {"as_of_date": as_of_date, "label": label})
        conn.execute(insert_sql, rows)


_EQUITY_REPORT_FIELDS = [
    "bsns_year", "reprt_code", "fs_div", "overview_summary", "trends_summary", "financial_summary",
    "revenue", "revenue_prior", "operating_income", "operating_income_prior",
    "net_income", "net_income_prior", "total_assets", "total_assets_prior",
    "total_liabilities", "total_liabilities_prior", "total_equity", "total_equity_prior",
    "operating_cash_flow", "operating_cash_flow_prior", "fetched_at",
]


def save_equity_report(corp_code: str, record: dict) -> None:
    """record: the 21 fields in _EQUITY_REPORT_FIELDS -- see equity_report_tables.sql."""
    columns = ", ".join(_EQUITY_REPORT_FIELDS)
    placeholders = ", ".join(f":{f}" for f in _EQUITY_REPORT_FIELDS)
    update_clause = ", ".join(f"{f} = EXCLUDED.{f}" for f in _EQUITY_REPORT_FIELDS)
    insert_sql = text(
        f"""
        INSERT INTO equity_report (corp_code, {columns}) VALUES (:corp_code, {placeholders})
        ON CONFLICT (corp_code) DO UPDATE SET {update_clause}
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, {**record, "corp_code": corp_code})


def fetch_equity_report(corp_code: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM equity_report WHERE corp_code = :corp_code"), {"corp_code": corp_code}
        ).mappings().first()
    return dict(row) if row else None
