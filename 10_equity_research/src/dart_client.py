import io
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# Loaded explicitly (same pattern as 06_post_filter/src/sectors.py) so this module works
# regardless of import order, and anyone cloning the repo just fills in DART_API_KEY in
# 01_data_load/.env -- no code change needed.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "01_data_load" / ".env")

API_KEY = os.environ["DART_API_KEY"]
BASE_URL = "https://opendart.fss.or.kr/api"

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 5

# DART status codes that mean "request succeeded, there's just nothing to return" -- NOT errors.
# Everything else (invalid key, rate limit, etc.) raises.
_NO_DATA_STATUS = "013"


class DartApiError(RuntimeError):
    pass


def parse_dart_date(value: str | None):
    """DART returns dates as bare 'YYYYMMDD' strings (est_dt, rcept_dt, ...), not ISO -- convert
    before handing to a DATE column."""
    if not value:
        return None
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def _get(endpoint: str, params: dict, binary: bool = False):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{BASE_URL}/{endpoint}",
                params={"crtfc_key": API_KEY, **params},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.content if binary else resp.json()
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
    raise last_error


def fetch_corp_codes() -> list[dict]:
    """Downloads DART's full corp_code <-> stock_code master list (a zipped XML file covering
    every entity DART has ever seen a filing from, listed or not) and parses it into rows.
    This is the one DART call that returns a ZIP instead of JSON -- everything else below is
    JSON. Rarely changes (new listings only), so this is meant to be run occasionally
    (scripts/update_corp_codes.py), not on every report generation."""
    raw_zip = _get("corpCode.xml", params={}, binary=True)
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")
    root = ET.fromstring(xml_bytes)
    rows = []
    for node in root.iter("list"):
        stock_code = (node.findtext("stock_code") or "").strip()
        rows.append(
            {
                "corp_code": node.findtext("corp_code"),
                "corp_name": node.findtext("corp_name"),
                "stock_code": stock_code or None,  # most DART entities are unlisted -- blank, not "000000"
                "modify_date": node.findtext("modify_date"),
            }
        )
    return rows


def fetch_company_overview(corp_code: str) -> dict | None:
    """company.json -- basic corporate info (CEO, industry code, founding date, address, etc.)
    for a single corp_code. Returns None if DART has no record (status 013), raises otherwise."""
    data = _get("company.json", params={"corp_code": corp_code})
    if data.get("status") == _NO_DATA_STATUS:
        return None
    if data.get("status") != "000":
        raise DartApiError(f"company.json failed for corp_code={corp_code}: {data}")
    return data


def fetch_filing_list(corp_code: str, days_back: int = 450, pblntf_ty: str = "A") -> list[dict]:
    """list.json -- recent disclosures for a corp_code. pblntf_ty='A' = 정기공시 (사업/분기/반기
    보고서 등), which is what 10_equity_research needs for "사업 개요/실적 요약". `days_back=450`
    (~15 months) comfortably covers at least one full annual cycle even right after a fiscal
    year-end. Returns [] if DART has nothing in range (status 013), raises on real errors."""
    end = date.today()
    begin = end - timedelta(days=days_back)
    data = _get(
        "list.json",
        params={
            "corp_code": corp_code,
            "bgn_de": begin.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "pblntf_ty": pblntf_ty,
            "page_count": 100,
        },
    )
    if data.get("status") == _NO_DATA_STATUS:
        return []
    if data.get("status") != "000":
        raise DartApiError(f"list.json failed for corp_code={corp_code}: {data}")
    return data.get("list", [])


def find_latest_annual_report(corp_code: str) -> dict | None:
    """The most recent plain 사업보고서 filing (NOT "[첨부정정]사업보고서" -- an attachment-only
    correction that doesn't carry the full "사업의 내용"/재무제표 content, found via real testing
    against 000810/012450). Shared by fetch_business_description.py and
    generate_equity_report.py so the "which filing do we use" logic lives in one place."""
    filings = fetch_filing_list(corp_code)
    annual_reports = sorted(
        (f for f in filings if f.get("report_nm", "").startswith("사업보고서")),
        key=lambda f: f["rcept_dt"],
        reverse=True,
    )
    return annual_reports[0] if annual_reports else None


def fetch_document(rcept_no: str) -> bytes:
    """document.xml -- the raw filing document (usually one XML file inside a ZIP) for a given
    receipt number from fetch_filing_list(). This is the "사업의 내용" free text itself, unlike
    company.json's structured-field overview -- see src/filing_parser.py for extraction."""
    return _get("document.xml", params={"rcept_no": rcept_no}, binary=True)


def fetch_financial_statements(corp_code: str, bsns_year: str, reprt_code: str = "11011", fs_div: str = "CFS") -> list[dict]:
    """fnlttSinglAcntAll.json -- the full set of standard financial-statement line items (BS,
    IS, CF, ...) for one filing, as structured rows (account_id/account_nm/thstrm_amount/...) --
    the actual source for "재무제표" report numbers, unlike the free-text 사업의 내용. fs_div:
    'CFS' (연결, consolidated) preferred; a company with no subsidiaries has no consolidated
    statement (status 013) and the caller should retry with 'OFS' (개별)."""
    data = _get(
        "fnlttSinglAcntAll.json",
        params={"corp_code": corp_code, "bsns_year": bsns_year, "reprt_code": reprt_code, "fs_div": fs_div},
    )
    if data.get("status") == _NO_DATA_STATUS:
        return []
    if data.get("status") != "000":
        raise DartApiError(f"fnlttSinglAcntAll.json failed for corp_code={corp_code}: {data}")
    return data.get("list", [])
