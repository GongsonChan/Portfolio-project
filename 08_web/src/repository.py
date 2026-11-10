import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "01_data_load" / "src"))

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import engine

METHOD_LABELS = {
    "parametric": "정규분포 가정",
    "historical": "역사적 시뮬레이션",
}
METHOD_ORDER = {m: i for i, m in enumerate(METHOD_LABELS)}
LIMIT_LABELS = {
    "var_99_1d": "1일 99% VaR",
    "var_95_1d": "1일 95% VaR",
    "es_99_1d": "1일 99% ES (CVaR)",
    "es_95_1d": "1일 95% ES (CVaR)",
    "max_component_var_share": "종목별 VaR 집중도",
}
_WORST_TICKER_RE = re.compile(r"^(.+?) \(worst: (.+)\)$")
RISK_LEVEL_CLASS = {"낮음": "green", "중간": "yellow", "높음": "red"}  # reuses the green/yellow/
# red badge palette already defined for 09's Basel zone (see dashboard.html's <style>)


def fetch_latest_final_portfolio(label: str = "default") -> tuple[pd.DataFrame, object]:
    """06_post_filter's latest final weights, joined with name/sector for display."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM final_portfolio WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return pd.DataFrame(), None
        df = pd.read_sql(
            text(
                """
                SELECT fp.ticker, fp.weight, s.name, s.sector, s.market
                FROM final_portfolio fp JOIN securities s ON fp.ticker = s.ticker
                WHERE fp.label = :label AND fp.as_of_date = :as_of_date
                ORDER BY fp.weight DESC
                """
            ),
            conn,
            params={"label": label, "as_of_date": as_of_date},
        )
    return df, as_of_date


def fetch_backtest_results(label: str = "default") -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT as_of_date, portfolio_return, n_holdings FROM backtest_period_returns "
                "WHERE label = :label ORDER BY as_of_date"
            ),
            conn,
            params={"label": label},
        )


def fetch_risk_report(label: str = "default") -> dict:
    """09_risk_management's latest daily risk report (VaR/ES, component VaR shares, limit
    checks) -- queried directly against its tables rather than importing its repository.py,
    same self-contained pattern as summarize_backtest below for 07's backtest results."""
    with engine.connect() as conn:
        as_of_date = conn.execute(
            text("SELECT MAX(as_of_date) FROM var_estimates WHERE label = :label"), {"label": label}
        ).scalar()
        if as_of_date is None:
            return {}
        estimates = pd.read_sql(
            text(
                "SELECT method, confidence, var_value, es_value FROM var_estimates "
                "WHERE label = :label AND as_of_date = :d"
            ),
            conn,
            params={"label": label, "d": as_of_date},
        )
        if not estimates.empty:
            estimates["method_label"] = estimates["method"].map(lambda m: METHOD_LABELS.get(m, m))
            estimates["_order"] = estimates["method"].map(lambda m: METHOD_ORDER.get(m, 99))
            estimates = estimates.sort_values(["_order", "confidence"]).drop(columns="_order")
        # share_of_total is confidence-invariant under the Gaussian model (z_alpha cancels in
        # the ratio), so either confidence's rows give the same per-ticker breakdown -- 0.99
        # picked arbitrarily.
        components = pd.read_sql(
            text(
                "SELECT ticker, share_of_total FROM component_var "
                "WHERE label = :label AND as_of_date = :d AND confidence = 0.99 "
                "ORDER BY share_of_total DESC"
            ),
            conn,
            params={"label": label, "d": as_of_date},
        )
        limits = pd.read_sql(
            text(
                "SELECT limit_name, limit_value, observed_value, breached FROM risk_limit_checks "
                "WHERE label = :label AND as_of_date = :d ORDER BY breached DESC, limit_name"
            ),
            conn,
            params={"label": label, "d": as_of_date},
        )

        # risk_limit_checks.limit_name is a raw identifier (e.g. "max_component_var_share
        # (worst: 000660)") meant for DB/logs, not display -- resolve to a Korean label + the
        # security's name instead of its ticker here, same name_map pattern as the component
        # table below.
        tickers = set(components["ticker"]) if not components.empty else set()
        for name in limits["limit_name"] if not limits.empty else []:
            m = _WORST_TICKER_RE.match(name)
            if m:
                tickers.add(m.group(2))
        name_map = {}
        if tickers:
            rows = conn.execute(
                text("SELECT ticker, name FROM securities WHERE ticker = ANY(:tickers)"),
                {"tickers": list(tickers)},
            ).fetchall()
            name_map = {t: n for t, n in rows}

    def _label_limit(name: str) -> str:
        m = _WORST_TICKER_RE.match(name)
        if m:
            base, ticker = m.groups()
            return f"{LIMIT_LABELS.get(base, base)} ({name_map.get(ticker, ticker)})"
        return LIMIT_LABELS.get(name, name)

    if not limits.empty:
        limits["limit_name"] = limits["limit_name"].apply(_label_limit)
    if not components.empty:
        components["name"] = components["ticker"].map(lambda t: name_map.get(t, t))

    # es_value is NULL for cornish_fisher (no standard closed-form ES) -- pandas reads that as
    # NaN, not None, so the template's `is not none` check needs it converted explicitly.
    if not estimates.empty:
        estimates = estimates.astype(object).where(pd.notnull(estimates), None)

    return {
        "as_of_date": as_of_date,
        "estimates": estimates.to_dict("records"),
        "components": components.to_dict("records"),
        "limits": limits.to_dict("records"),
        "any_breach": bool(limits["breached"].any()) if not limits.empty else False,
    }


# Mirrors 09_risk_management/src/basel_capital.py's capital_zone()/ZONE_MESSAGES -- duplicated
# rather than imported (same self-contained-queries pattern this module uses throughout) since
# it's a pure function of n_breaches_250d, which basel_capital_requirement already stores, so
# there's no need to also persist the derived zone/message.
ZONE_MESSAGES = {
    "green": "모형 양호 — 백테스트 결과가 신뢰수준과 통계적으로 부합, 추가 조치 불필요.",
    "yellow": "모형 정확도 저하 경고 — 예외 발생 빈도가 기대보다 높음, 원인 분석 및 모니터링 강화 필요.",
    "red": "리스크 모형 위험 — 꼬리 리스크를 심각하게 과소추정하고 있음이 확인됨, 모형 재검토 시급.",
}


def _capital_zone(n_breaches_250d: int) -> str:
    if n_breaches_250d >= 10:
        return "red"
    if n_breaches_250d >= 5:
        return "yellow"
    return "green"


def fetch_basel_capital(label: str = "default") -> dict:
    """09's latest Basel 1996 Amendment capital multiplier calculation (see
    09_risk_management/src/basel_capital.py)."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT as_of_date, n_breaches_250d, multiplier, var_10day, avg_var_10day, "
                "n_days_in_average, required_capital FROM basel_capital_requirement "
                "WHERE label = :label ORDER BY as_of_date DESC LIMIT 1"
            ),
            {"label": label},
        ).mappings().first()
    if not row:
        return {}
    result = dict(row)
    result["zone"] = _capital_zone(result["n_breaches_250d"])
    result["zone_message"] = ZONE_MESSAGES[result["zone"]]
    return result


def fetch_var_backtest_summary(label: str = "default") -> list[dict]:
    """09's Kupiec + Basel traffic-light re-validation of the VaR model itself (weekly)."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                "SELECT method, confidence, n_days, n_breaches, expected_breach_rate, "
                "kupiec_p_value, traffic_light, computed_at FROM var_backtest_summary "
                "WHERE label = :label ORDER BY method, confidence"
            ),
            conn,
            params={"label": label},
        )
    return df.to_dict("records")


def summarize_backtest(results: pd.DataFrame) -> dict:
    """Same calculation as 07_backtest/src/backtest.py's summarize() -- duplicated rather than
    imported so this read-only web module doesn't depend on 07's script path."""
    if results.empty:
        return {}
    returns = results["portfolio_return"]
    dates = pd.to_datetime(results["as_of_date"])
    median_gap_days = dates.diff().dropna().dt.days.median()
    periods_per_year = 365.25 / median_gap_days if median_gap_days else float("nan")

    cumulative = (1 + returns).prod() - 1
    n_periods = len(returns)
    annualized_return = (1 + cumulative) ** (periods_per_year / n_periods) - 1 if n_periods else float("nan")
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_vol if annualized_vol else float("nan")

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "n_periods": n_periods,
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "equity_curve": list(zip(dates.dt.strftime("%Y-%m-%d"), equity)),
    }


_FINANCIAL_ROW_LABELS = [
    ("revenue", "매출액"),
    ("operating_income", "영업이익"),
    ("net_income", "당기순이익"),
    ("total_assets", "자산총계"),
    ("total_liabilities", "부채총계"),
    ("total_equity", "자본총계"),
    ("operating_cash_flow", "영업활동현금흐름"),
]


def _build_financial_rows(report: dict) -> list[dict]:
    """report: a row from equity_report (10_equity_research/scripts/generate_equity_report.py).
    Converts the flat current/prior NUMERIC columns into a display table with YoY % change
    computed once here rather than in Jinja (None-guarded division is awkward in templates)."""
    rows = []
    for key, label in _FINANCIAL_ROW_LABELS:
        # NUMERIC columns come back as Decimal, not float -- Decimal / 1e8 (a float) raises
        # TypeError in the template, so cast here rather than in Jinja.
        current = float(report[key]) if report.get(key) is not None else None
        prior = float(report[f"{key}_prior"]) if report.get(f"{key}_prior") is not None else None
        change_pct = (current - prior) / abs(prior) * 100 if current is not None and prior else None
        rows.append({"label": label, "current": current, "prior": prior, "change_pct": change_pct})
    return rows


def fetch_equity_research(ticker: str, label: str = "default") -> dict:
    """10_equity_research's per-ticker report: DART company overview + "사업의 내용", latest
    news/qualitative-risk assessment, and business-model similarity to the rest of the current
    portfolio. Queried directly (not via 10's repository.py), same self-contained pattern this
    module uses for 09's tables."""
    with engine.connect() as conn:
        name = conn.execute(text("SELECT name FROM securities WHERE ticker = :ticker"), {"ticker": ticker}).scalar()

        overview = conn.execute(
            text(
                """
                SELECT o.corp_code, o.ceo_name, o.induty_code, o.est_dt, o.address, o.homepage_url
                FROM dart_company_overview o JOIN dart_corp_codes c ON o.corp_code = c.corp_code
                WHERE c.stock_code = :ticker
                """
            ),
            {"ticker": ticker},
        ).mappings().first()

        report = None
        if overview:
            report = conn.execute(
                text("SELECT * FROM equity_report WHERE corp_code = :corp_code"),
                {"corp_code": overview["corp_code"]},
            ).mappings().first()

        news = conn.execute(
            text(
                """
                SELECT as_of_date, news_summary, key_issues, risk_level, risk_rationale, n_agree, n_runs
                FROM equity_news_assessment WHERE label = :label AND ticker = :ticker
                ORDER BY as_of_date DESC LIMIT 1
                """
            ),
            {"label": label, "ticker": ticker},
        ).mappings().first()

        similarity = pd.read_sql(
            text(
                """
                SELECT bs.ticker_j AS ticker, s.name, bs.similarity FROM business_similarity bs
                JOIN securities s ON bs.ticker_j = s.ticker
                WHERE bs.label = :label AND bs.ticker_i = :ticker AND bs.ticker_j != :ticker
                  AND bs.as_of_date = (SELECT MAX(as_of_date) FROM business_similarity WHERE label = :label)
                ORDER BY bs.similarity DESC
                """
            ),
            conn,
            params={"label": label, "ticker": ticker},
        )

    news_dict = dict(news) if news else None
    if news_dict:
        news_dict["risk_class"] = RISK_LEVEL_CLASS.get(news_dict["risk_level"], "")

    report_dict = dict(report) if report else None
    financial_rows = _build_financial_rows(report_dict) if report_dict else []

    return {
        "ticker": ticker,
        "name": name,
        "overview": dict(overview) if overview else None,
        "report": report_dict,
        "financial_rows": financial_rows,
        "news": news_dict,
        "similarity": similarity.to_dict("records"),
    }
