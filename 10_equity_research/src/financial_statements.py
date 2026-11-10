import re

# Standardized IFRS taxonomy tags (account_id). Confirmed against TWO real filings with
# genuinely different statement structures: Samsung Electronics (제조업 -- separate IS section,
# "영업이익" under the DART extension tag dart_OperatingIncomeLoss) and Shinhan Financial Group
# (금융지주 -- NO separate IS section at all, everything under CIS instead; "영업이익" under the
# different standard tag ifrs-full_ProfitLossFromOperatingActivities; and no "매출액"/Revenue tag
# exists at all -- financial institutions report 이자수익/수수료수익/보험수익 separately instead
# of a single revenue line, so "revenue" is correctly None for this sector, not a bug to fix).
# Each item tries every (sj_div, account_id) combination in order before falling back to a
# Korean-name substring match.
KEY_ITEMS = {
    "revenue": {
        "sj_divs": ["IS", "CIS"],
        "account_ids": ["ifrs-full_Revenue"],
        "fallback_nm": ["매출액", "수익(매출액)", "영업수익"],
    },
    "operating_income": {
        "sj_divs": ["IS", "CIS"],
        "account_ids": ["dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities"],
        "fallback_nm": ["영업이익"],
    },
    "net_income": {
        "sj_divs": ["IS", "CIS"],
        "account_ids": ["ifrs-full_ProfitLoss"],
        "fallback_nm": ["당기순이익"],
    },
    "total_assets": {"sj_divs": ["BS"], "account_ids": ["ifrs-full_Assets"], "fallback_nm": ["자산총계"]},
    "total_liabilities": {"sj_divs": ["BS"], "account_ids": ["ifrs-full_Liabilities"], "fallback_nm": ["부채총계"]},
    "total_equity": {"sj_divs": ["BS"], "account_ids": ["ifrs-full_Equity"], "fallback_nm": ["자본총계"]},
    "operating_cash_flow": {
        "sj_divs": ["CF"],
        "account_ids": ["ifrs-full_CashFlowsFromUsedInOperatingActivities"],
        "fallback_nm": ["영업활동현금흐름", "영업활동으로인한현금흐름"],
    },
}

_YEAR_RE = re.compile(r"\((\d{4})\.")


def parse_bsns_year(report_nm: str) -> str | None:
    """"사업보고서 (2025.12)" -> "2025" -- the fiscal year a periodic report covers, needed by
    fnlttSinglAcntAll.json (DART doesn't infer this from rcept_no)."""
    m = _YEAR_RE.search(report_nm)
    return m.group(1) if m else None


def _to_number(raw: str | None) -> float | None:
    if raw in (None, "", "-"):
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def extract_key_figures(rows: list[dict]) -> dict:
    """rows: fetch_financial_statements()'s raw account rows. Returns {key: current_amount,
    key+'_prior': prior_period_amount} for each of KEY_ITEMS -- None for anything not found
    (real filings sometimes omit a line item, or -- for "revenue" at a financial institution --
    genuinely don't report that concept at all) rather than guessing."""
    by_section: dict[str, list[dict]] = {}
    for r in rows:
        by_section.setdefault(r.get("sj_div"), []).append(r)

    result = {}
    for key, spec in KEY_ITEMS.items():
        candidates = [r for sj in spec["sj_divs"] for r in by_section.get(sj, [])]
        match = next((r for r in candidates if r.get("account_id") in spec["account_ids"]), None)
        if match is None:
            fallback_names = {n.replace(" ", "") for n in spec["fallback_nm"]}
            match = next((r for r in candidates if r.get("account_nm", "").replace(" ", "") in fallback_names), None)
        result[key] = _to_number(match["thstrm_amount"]) if match else None
        result[f"{key}_prior"] = _to_number(match["frmtrm_amount"]) if match else None
    return result
