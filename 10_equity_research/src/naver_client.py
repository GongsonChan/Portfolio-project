import html
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / "01_data_load" / ".env")

# These credentials are from NAVER API HUB (ncloud.com's unified API gateway for Naver's open
# APIs), NOT the older developers.naver.com Open API -- the two are separate products with
# different endpoints/headers/key formats (X-NCP-APIGW-API-KEY-ID/X-NCP-APIGW-API-KEY here vs.
# X-Naver-Client-Id/X-Naver-Client-Secret there), confirmed by a 401 on the old endpoint and a
# working 200 on this one with the same key pair.
CLIENT_ID = os.environ["NAVER_CLIENT_ID"]
CLIENT_SECRET = os.environ["NAVER_CLIENT_SECRET"]
BASE_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 5

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """Naver wraps matched query terms in <b></b> and HTML-escapes the rest -- strip both so
    the LLM (and any future display) sees plain text."""
    return html.unescape(_TAG_RE.sub("", text)).strip()


def fetch_news(query: str, display: int = 10, sort: str = "date") -> list[dict]:
    """Recent news articles for a query (company name). sort='date' (not Naver's default 'sim')
    because this project cares about *recent* issues, not best keyword match. Returns
    [{"title", "description", "link", "pub_date"}, ...], HTML-cleaned."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                BASE_URL,
                headers={"X-NCP-APIGW-API-KEY-ID": CLIENT_ID, "X-NCP-APIGW-API-KEY": CLIENT_SECRET},
                params={"query": query, "display": display, "sort": sort},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return [
                {
                    "title": _clean(item["title"]),
                    "description": _clean(item["description"]),
                    "link": item["link"],
                    "pub_date": item["pubDate"],
                }
                for item in items
            ]
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
    raise last_error
