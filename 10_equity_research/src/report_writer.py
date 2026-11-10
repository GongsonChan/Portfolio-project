from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Same explicit-load pattern as 06_post_filter/src/sectors.py.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "01_data_load" / ".env")

MODEL = "claude-haiku-5-5"


class EquityReportSection(BaseModel):
    ticker: str
    overview_summary: str   # <=5 sentences: what the company actually does
    trends_summary: str     # recent business focus / contracts, from the filing itself
    financial_summary: str  # <=3 sentences, grounded ONLY in the given figures

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: str) -> str:
        # Same defensive normalization as news_assessment.NewsAssessment -- the model
        # occasionally echoes "000270 기아" instead of just "000270".
        v = v.strip()
        return v.split()[0] if v else v


class _EquityReportResult(BaseModel):
    reports: list[EquityReportSection]


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
            f"[사업보고서 '사업의 내용' 발췌]\n{c['business_text'][:4000]}\n\n"
            f"[최근 재무제표 핵심 수치, 연결기준]\n{fig_lines}"
        )

    prompt = (
        "다음은 포트폴리오 보유 종목별 사업보고서 발췌 + 최근 재무제표 핵심 수치다. 각 종목(ticker, "
        "'### ' 뒤의 6자리 숫자 코드만 정확히)마다 아래 세 가지를 작성해라 -- 원문을 그대로 베끼지 말고 "
        "직접 읽고 새로 정리한 요약이어야 한다:\n"
        "1. overview_summary: 이 회사가 뭘 하는 회사인지 5문장 이내로 요약 (주요 사업부문, 핵심 제품/서비스).\n"
        "2. trends_summary: 최근 집중하고 있는 사업/투자 분야, 주요 거래처·계약·수주 현황 등 사업보고서에 "
        "나온 최신 동향 정리 (외부 뉴스가 아니라 사업보고서 자체에 적힌 내용 기준).\n"
        "3. financial_summary: **[최근 재무제표 핵심 수치]에 적힌 숫자만** 근거로 3문장 이내로 요약 -- "
        "전기 대비 증감 방향과 규모, 수익성/재무건전성 평가 포함. 어떤 항목이 '데이터 없음'이면 그 항목은 "
        "언급하지 말고 건너뛰어라 -- 사업보고서 발췌문이나 다른 어디에서도 숫자를 가져오지 마라 (예: 금융/지주 "
        "회사는 '매출액' 항목 자체가 없는 게 정상이니 억지로 추정하거나 다른 수치로 대체하지 말 것).\n\n"
        + "\n\n".join(sections)
    )

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,
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
    return {r.ticker: r for r in response.parsed_output.reports}
