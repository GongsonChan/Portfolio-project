CREATE TABLE IF NOT EXISTS equity_news_assessment (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    ticker VARCHAR(20) NOT NULL,
    n_articles INT NOT NULL,        -- how many Naver articles the summary was built from (0 is
    -- valid -- means genuinely no recent news, not a fetch failure)
    news_summary TEXT NOT NULL,
    key_issues TEXT NOT NULL,
    risk_level VARCHAR(10) NOT NULL,   -- '낮음' | '중간' | '높음' -- LLM's qualitative read,
    -- complements 09_risk_management's quantitative VaR/ES (see 10/DESIGN.md's "왜 필요한가")
    risk_rationale TEXT NOT NULL,
    n_runs INT NOT NULL DEFAULT 1,   -- self-consistency: how many times the LLM assessment was
    n_agree INT NOT NULL DEFAULT 1,  -- rerun on the same input, and how many agreed with the
    -- stored risk_level -- e.g. 4/5 means the label is less certain than 5/5 (see DESIGN.md's
    -- "정성적 위험도 등급의 run-to-run 불안정성")
    PRIMARY KEY (as_of_date, label, ticker)
);
