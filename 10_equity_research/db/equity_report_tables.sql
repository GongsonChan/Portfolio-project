CREATE TABLE IF NOT EXISTS equity_report (
    corp_code VARCHAR(8) PRIMARY KEY REFERENCES dart_corp_codes(corp_code),
    bsns_year VARCHAR(4) NOT NULL,
    reprt_code VARCHAR(10) NOT NULL,
    fs_div VARCHAR(5) NOT NULL,        -- 'CFS' (연결) | 'OFS' (개별) -- whichever actually had data
    overview_summary TEXT NOT NULL,    -- LLM, <=8 sentences -- NOT the raw 사업의 내용 text
    trends_summary TEXT NOT NULL,      -- LLM, business focus / contracts from the filing
    financial_summary TEXT NOT NULL,   -- LLM, <=5 sentences, grounded in the figures below
    revenue NUMERIC, revenue_prior NUMERIC,
    operating_income NUMERIC, operating_income_prior NUMERIC,
    net_income NUMERIC, net_income_prior NUMERIC,
    total_assets NUMERIC, total_assets_prior NUMERIC,
    total_liabilities NUMERIC, total_liabilities_prior NUMERIC,
    total_equity NUMERIC, total_equity_prior NUMERIC,
    operating_cash_flow NUMERIC, operating_cash_flow_prior NUMERIC,
    fetched_at DATE NOT NULL
);
