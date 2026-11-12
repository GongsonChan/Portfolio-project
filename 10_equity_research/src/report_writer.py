from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Same explicit-load pattern as 06_post_filter/src/sectors.py.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "01_data_load" / ".env")

MODEL = "claude-haiku-5-5"


class EquityReportSection(BaseModel):
    """Public shape: each *_summary is one "\n\n"-joined string, ready to hand straight to a TEXT
    column (see save_equity_report) -- the paragraph split already happened in _RawReportSection."""
    ticker: str
    overview_summary: str   # <=8 sentences, 3 paragraphs: what the company actually does
    trends_summary: str     # recent business focus / contracts, from the filing itself
    financial_summary: str  # <=5 sentences, 2 paragraphs, grounded ONLY in the given figures


class _RawReportSection(BaseModel):
    """What the LLM actually fills in -- one named field per paragraph instead of a single string
    or a list. Two earlier attempts both failed:
    1. A single string + "문단은 줄바꿈 두 번으로 구분해라" in the prose -- the model ignored the
       formatting instruction entirely and wrote one unbroken block every time (verified: 0 "\n"
       characters in the saved DB text).
    2. list[str] with "정확히 3개의 배열 원소로 나눠라" -- the model interpreted "array of
       paragraphs" as "array of sentences" instead (got 7 one-sentence elements for a field asked
       to be 3 paragraphs/8 sentences). An array has no inherent semantics for what belongs in
       each slot.
    Named fields (overview_who/overview_how/overview_position, financial_income/
    financial_position) carry that semantic in the field name itself, which the model follows
    far more reliably than either prose formatting or positional array slots."""
    ticker: str
    overview_who: str       # ①사업부문/보고부문 구성 ②제품·서비스 라인업 (매출비중 등) -- 무엇을 하는 회사인지
    overview_how: str       # ③생산/서비스 거점 ④원재료·공급망 ⑤주요 거래처·고객사·판매망 -- 어떻게 만들고 파는지
    overview_position: str  # ⑥경쟁환경/시장지위 ⑦신사업·투자방향 ⑧연결대상 종속기업 현황
    trends_summary: str     # recent business focus / contracts, from the filing itself
    financial_income: str   # 매출/영업이익/순이익의 전기 대비 증감 방향과 규모
    financial_position: str  # 자산/부채/자본/영업현금흐름 증감 + 수익성/재무건전성 종합 평가

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        # Same defensive normalization as news_assessment.NewsAssessment -- the model
        # occasionally echoes "000270 기아" instead of just "000270".
        v = v.strip()
        return v.split()[0] if v else v


class _EquityReportResult(BaseModel):
    reports: list[_RawReportSection]


def _join_paragraphs(*paragraphs: str) -> str:
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def _fmt_eok(value: float | None) -> str:
    """원 단위 숫자를 억원으로 변환해서 LLM 프롬프트에 넣기 좋은 크기로 축약."""
    return f"{value / 1e8:,.0f}억원" if value is not None else "데이터 없음"


def write_reports(companies: list[dict]) -> dict[str, EquityReportSection]:
    """companies: [{"ticker", "name", "business_text", "figures": {...from
    financial_statements.extract_key_figures...}}, ...]. One batched LLM call for the whole
    portfolio (same pattern as 06's sector classification / news_assessment.assess_via_llm) --
    produces three SEPARATE sections meant for three separate panels on the report page, not a
    single blob, and explicitly NOT a copy of the raw filing text (see DESIGN.md)."""
    sections = []
    for c in companies:
        f = c["figures"]
        fig_lines = (
            f"매출액 {_fmt_eok(f.get('revenue'))} (전기 {_fmt_eok(f.get('revenue_prior'))}), "
            f"영업이익 {_fmt_eok(f.get('operating_income'))} (전기 {_fmt_eok(f.get('operating_income_prior'))}), "
            f"당기순이익 {_fmt_eok(f.get('net_income'))} (전기 {_fmt_eok(f.get('net_income_prior'))}), "
            f"자산총계 {_fmt_eok(f.get('total_assets'))}, 부채총계 {_fmt_eok(f.get('total_liabilities'))}, "
            f"자본총계 {_fmt_eok(f.get('total_equity'))}, "
            f"영업활동현금흐름 {_fmt_eok(f.get('operating_cash_flow'))} (전기 {_fmt_eok(f.get('operating_cash_flow_prior'))})"
        )
        sections.append(
            f"### {c['ticker']} {c['name']}\n"
            f"[사업보고서 '사업의 내용' 발췌]\n{c['business_text'][:8000]}\n\n"  # full stored
            # text (filing_parser's own max_chars cap) -- 8문장 체크리스트를 채우려면 4000자로
            # 잘랐을 때보다 원재료가 더 필요함
            f"[최근 재무제표 핵심 수치, 연결기준]\n{fig_lines}"
        )

    prompt = (
        "다음은 포트폴리오 보유 종목별 사업보고서 발췌 + 최근 재무제표 핵심 수치다. 각 종목(ticker, "
        "'### ' 뒤의 6자리 숫자 코드만 정확히)마다 아래 세 가지를 작성해라 -- 원문을 그대로 베끼지 말고 "
        "직접 읽고 새로 정리한 요약이어야 한다:\n"
        "1. 이 회사가 뭘 하는 회사인지, 아래 세 필드에 나눠서 **총 8문장**을 채워라 (짧게 쓰지 말고 각 "
        "필드가 요구하는 항목을 전부 언급하되, 사업보고서 발췌문에 해당 정보가 있을 때만 -- 없으면 억지로 "
        "지어내지 말고 건너뛰고, 있는 정보는 한 문장으로 뭉뚱그리지 말고 구체적으로 풀어써라):\n"
        "   - overview_who (2-3문장): ①주요 사업부문/보고부문 구성 ②핵심 제품·서비스 라인업 (매출 비중 "
        "등 포함)\n"
        "   - overview_how (2-3문장): ③생산/서비스 거점(국내외 공장·법인 등) ④원재료·공급망 ⑤주요 "
        "거래처·고객사·판매망\n"
        "   - overview_position (2-3문장): ⑥경쟁 환경 또는 시장 지위 ⑦최근 신사업/투자 방향 ⑧연결대상 "
        "종속기업 현황\n"
        "2. trends_summary: 최근 집중하고 있는 사업/투자 분야, 주요 거래처·계약·수주 현황 등 사업보고서에 "
        "나온 최신 동향 정리 (외부 뉴스가 아니라 사업보고서 자체에 적힌 내용 기준).\n"
        "3. **[최근 재무제표 핵심 수치]에 적힌 숫자만** 근거로 아래 두 필드에 나눠서 **총 5문장**을 채워라 "
        "-- 어떤 항목이 '데이터 없음'이면 그 항목은 언급하지 말고 건너뛰어라 (사업보고서 발췌문이나 다른 "
        "어디에서도 숫자를 가져오지 마라. 예: 금융/지주 회사는 '매출액' 항목 자체가 없는 게 정상이니 억지로 "
        "추정하거나 다른 수치로 대체하지 말 것):\n"
        "   - financial_income (2-3문장): 매출/영업이익/순이익의 전기 대비 증감 방향과 규모\n"
        "   - financial_position (2-3문장): 자산/부채/자본/영업현금흐름의 증감, 마지막 문장에서 "
        "수익성/재무건전성 종합 평가\n\n"
        + "\n\n".join(sections)
    )

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16384,  # raised from 8192 after lengthening overview_summary (5->8 sentences)
        # and financial_summary (3->5) made the batched 10-company JSON overflow 8192 again
        # ("EOF while parsing a string" mid-JSON) -- same failure class as the original
        # "JSON 중간 절단" bug, now re-triggered by longer per-company output instead of more
        # companies.
        # Found via a raw (non-structured) call that this prompt's size pushes claude-haiku-5-5
        # into spending 5000+ of the 8192 output tokens on internal "thinking" before writing any
        # JSON, truncating the actual report (stop_reason="max_tokens", parsed_output=None).
        # This task is summarization grounded in given text/figures, not multi-step reasoning, so
        # thinking is disabled outright rather than just raising max_tokens further.
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
        output_format=_EquityReportResult,
    )
    if response.parsed_output is None:
        raise RuntimeError(f"LLM response didn't parse (stop_reason={response.stop_reason}) -- see report_writer.py")
    return {
        r.ticker: EquityReportSection(
            ticker=r.ticker,
            overview_summary=_join_paragraphs(r.overview_who, r.overview_how, r.overview_position),
            trends_summary=r.trends_summary,
            financial_summary=_join_paragraphs(r.financial_income, r.financial_position),
        )
        for r in response.parsed_output.reports
    }
