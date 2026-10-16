import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

AUTH_KEY = os.environ["KRX_API"]
BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/sto"

ENDPOINTS = {
    "KOSPI": {"base_info": "stk_isu_base_info", "daily_trade": "stk_bydd_trd"},
    "KOSDAQ": {"base_info": "ksq_isu_base_info", "daily_trade": "ksq_bydd_trd"},
}

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 5


def _request(endpoint: str, bas_dd: str) -> list[dict]:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{BASE_URL}/{endpoint}",
                headers={"AUTH_KEY": AUTH_KEY},
                params={"basDd": bas_dd},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("OutBlock_1", [])
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
    raise last_error


def fetch_base_info(market: str, bas_dd: str) -> list[dict]:
    return _request(ENDPOINTS[market]["base_info"], bas_dd)


def fetch_daily_trade(market: str, bas_dd: str) -> list[dict]:
    return _request(ENDPOINTS[market]["daily_trade"], bas_dd)


def latest_trading_day(market: str = "KOSPI", max_lookback: int = 10) -> str:
    d = date.today()
    for _ in range(max_lookback):
        bas_dd = d.strftime("%Y%m%d")
        if fetch_daily_trade(market, bas_dd):
            return bas_dd
        d -= timedelta(days=1)
    raise RuntimeError("no recent trading day found within lookback window")


def to_num(value):
    if value in (None, "", "-"):
        return None
    return float(str(value).replace(",", ""))


def to_date(value):
    if value in (None, "", "-"):
        return None
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
