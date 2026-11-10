import re
from collections import Counter
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Same explicit-load pattern as 06_post_filter/src/sectors.py -- works regardless of import
# order, and a fresh clone just needs ANTHROPIC_API_KEY in 01_data_load/.env.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "01_data_load" / ".env")

MODEL = "claude-haiku-5-5"  # same cheap-model choice as 06's sector classification -- this is
# summarization + a constrained-label judgment, not deep reasoning

RiskLevel = Literal["낮음", "중간", "높음"]

# Several Korean conglomerates' sports teams are named identically to the parent company (e.g.
# 삼성화재 is both the insurer AND its V-리그 volleyball team, "삼성화재 블루밍스"; 삼성화재배 is
# also a baduk/Go tournament it sponsors) -- a plain Naver news search on the company name pulls
# in a lot of game-result/standings/tournament coverage that has nothing to do with the
# business. Naver's search API doesn't reliably support boolean/exclude query syntax (tested:
# appending a qualifier like "주가" cleans up 삼성화재 but makes OTHER tickers' results worse,
# diluted by generic whole-market roundup articles instead), so this filters by keyword after
# fetching. Tried the obvious terms first (배구, V리그, 감독, ...) and still let through real
# match coverage that uses volleyball/baduk-specific jargon instead of those words (e.g.
# "7:13->28:26 뒤집었다", "우스만 V리그 데뷔 후 최다 33득점 맹활약", "목진석, ...16강 승자로
# [PS]") -- expanded against those ACTUAL observed titles. Best-effort heuristic, not
# exhaustive; may need more terms if a new noise pattern shows up for a different ticker later.
_IRRELEVANT_KEYWORDS = [
    "배구단", "배구", "V리그", "바둑기사", "바둑", "기사(棋士)",
    "농구단", "프로농구", "KBL", "축구단", "K리그", "야구단", "프로야구", "KBO",
    "감독", "선수단", "선수", "결승", "리그 순위", "구단", "연승", "연패", "개막",
    "득점", "맹활약", "공격하는", "공격 시도", "공격성공률", "리시브", "세트", "서브권",
    "듀스", "진땀승", "16강", "32강", "8강", "4강", "승자전", "패자전", "[PS]", "[포토]",
    # 현대모비스(KBL 농구단 "현대모비스 피버스")에서 추가로 발견된 패턴
    "피버스", "치어리더", "야투율", "외인", "꼴찌", "폭발", "[화보]", "[현장분석]", "[BK 리뷰]",
    "순위픽", "신인픽", "[사진]",
]
_IRRELEVANT_RE = re.compile("|".join(re.escape(k) for k in _IRRELEVANT_KEYWORDS))


def filter_relevant_articles(articles: list[dict]) -> list[dict]:
    """Drops articles whose title matches the sports-team noise pattern above."""
    return [a for a in articles if not _IRRELEVANT_RE.search(a["title"])]


class NewsAssessment(BaseModel):
    ticker: str
    news_summary: str
    key_issues: str
    risk_level: RiskLevel
    risk_rationale: str

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        # The model occasionally echoes "000270 기아" instead of just "000270" (non-deterministic
        # -- found by rerunning the same input repeatedly, see scripts/check_assessment_consistency.py).
        # Defend at the parsing layer rather than trusting prompt compliance alone.
        v = v.strip()
        return v.split()[0] if v else v


class _NewsAssessmentResult(BaseModel):
    assessments: list[NewsAssessment]


def assess_via_llm(companies: list[dict]) -> dict[str, NewsAssessment]:
    """companies: [{"ticker", "name", "articles": [{"title", "description", "pub_date"}, ...]}, ...].
    One Claude call assesses the whole portfolio (same batching rationale as 06's
    _classify_via_llm -- independent per-ticker judgments, cheap to batch into one request).
    Produces both a news summary/issues (what's happening) and a qualitative risk read (so what)
    in the same pass since they share the same input -- see 10_equity_research/DESIGN.md's "왜
    필요한가" on how this complements 09_risk_management's quantitative VaR/ES."""
    sections = []
    for c in companies:
        if c["articles"]:
            article_lines = "\n".join(f"- ({a['pub_date']}) {a['title']}: {a['description']}" for a in c["articles"])
        else:
            article_lines = "(최근 뉴스 없음)"
        sections.append(f"### {c['ticker']} {c['name']}\n{article_lines}")

    prompt = (
        "다음은 포트폴리오 보유 종목별 최근 뉴스 목록이다. 각 종목(ticker)마다 아래 네 가지를 채워라. "
        "ticker 필드에는 아래 '### ' 뒤에 나오는 6자리 숫자 코드만 정확히 그대로 쓰고, 종목명이나 "
        "다른 텍스트는 절대 포함하지 마라 (예: '000270', '000270 기아'는 틀림).\n"
        "1. news_summary: 최근 동향을 2~4문장으로 요약\n"
        "2. key_issues: 지금 이 종목의 핵심 이슈를 한두 문장으로 요약\n"
        "3. risk_level: 이 뉴스들을 근거로 한 정성적 위험도. 아래 기준으로 낮음/중간/높음 중 하나를 "
        "엄격하게 골라라 (경계선에서 임의로 고르지 말고, 해당하는 구체적 근거가 뉴스에 있을 때만 "
        "그 등급을 선택):\n"
        "   - 낮음: 특이 뉴스 없음, 또는 일상적 실적/영업 발표 수준. 구조적 리스크 신호 없음.\n"
        "   - 중간: 진행 중인 M&A/사업재편의 통합·자금 부담, 업황/공급망 변동성에 대한 우려, "
        "경영진 교체, 아직 재무적 영향이 확정되지 않은 조사/소송 등 -- "
        "불확실성은 있지만 아직 확정된 중대 손실/제재는 아닌 경우.\n"
        "   - 높음: 소송 패소, 중대 제재/과징금, 회계 부정 의혹, 대규모 리콜·손실 공시, "
        "신용등급 강등, 상장폐지 우려 등 뉴스에 **명확하고 확정적인** 중대 리스크 사건이 있는 경우.\n"
        "4. risk_rationale: 그 위험도 평가의 근거를 1~2문장으로, 위 기준 중 어디에 해당하는지 포함해서\n\n"
        "뉴스가 없는 종목은 news_summary/key_issues에 '최근 특이 뉴스 없음'이라 쓰고 risk_level은 낮음으로.\n\n"
        + "\n\n".join(sections)
    )

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,  # 4096 was cutting the structured JSON off mid-object on some runs --
        # found via scripts/check_assessment_consistency.py (see DESIGN.md's "알려진 버그/교훈").
        messages=[{"role": "user", "content": prompt}],
        output_format=_NewsAssessmentResult,
    )
    return {a.ticker: a for a in response.parsed_output.assessments}


class ConsensusAssessment(BaseModel):
    ticker: str
    news_summary: str
    key_issues: str
    risk_level: RiskLevel
    risk_rationale: str
    n_runs: int
    n_agree: int  # of n_runs, how many produced this exact risk_level


def assess_with_consistency(companies: list[dict], n_runs: int = 5) -> dict[str, ConsensusAssessment]:
    """Runs assess_via_llm n_runs times on the SAME input and takes a majority vote on
    risk_level per ticker. claude-haiku-5-5 doesn't expose temperature control (confirmed by a
    400 "deprecated for this model" error when tried -- see DESIGN.md), so the sampling-driven
    label flicker check_assessment_consistency.py found (낮음/중간 flipping on 5/10 tickers
    across repeated calls) can't be reduced at the API-parameter level. This makes that
    uncertainty visible and reportable (n_agree/n_runs) instead of silently keeping whichever
    single run happened to execute."""
    all_runs: list[dict[str, NewsAssessment]] = [assess_via_llm(companies) for _ in range(n_runs)]

    tickers = {t for run in all_runs for t in run}
    consensus = {}
    for ticker in tickers:
        votes = [run[ticker] for run in all_runs if ticker in run]
        if not votes:
            continue
        majority_level, n_agree = Counter(v.risk_level for v in votes).most_common(1)[0]
        # Representative text comes from a run that actually produced the majority level, so the
        # rationale stays consistent with the reported risk_level instead of mixing runs.
        representative = next(v for v in votes if v.risk_level == majority_level)
        consensus[ticker] = ConsensusAssessment(
            ticker=ticker,
            news_summary=representative.news_summary,
            key_issues=representative.key_issues,
            risk_level=majority_level,
            risk_rationale=representative.risk_rationale,
            n_runs=len(votes),
            n_agree=n_agree,
        )
    return consensus
