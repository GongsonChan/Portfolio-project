CREATE TABLE IF NOT EXISTS dart_corp_codes (
    corp_code VARCHAR(8) PRIMARY KEY,  -- DART's own 8-digit identifier -- NOT the KRX ticker,
    -- every other DART endpoint keys off this, so this table is the ticker <-> corp_code bridge
    corp_name VARCHAR(200) NOT NULL,
    stock_code VARCHAR(6),             -- KRX 6-digit ticker; NULL for unlisted entities that
    -- still appear in DART's master list (non-listed subsidiaries etc.)
    modify_date DATE
);

CREATE TABLE IF NOT EXISTS dart_company_overview (
    corp_code VARCHAR(8) PRIMARY KEY REFERENCES dart_corp_codes(corp_code),
    stock_code VARCHAR(6),
    corp_name VARCHAR(200),
    ceo_name VARCHAR(200),
    induty_code VARCHAR(20),   -- 표준산업분류 코드 -- 06의 LLM 섹터 분류와는 다른, DART 자체 분류
    est_dt DATE,
    address VARCHAR(500),
    homepage_url VARCHAR(300),
    fetched_at DATE NOT NULL   -- cache staleness check: DART filings only change quarterly, so
    -- this doesn't need daily refetching (see 10_equity_research/DESIGN.md's caching note)
);

CREATE TABLE IF NOT EXISTS dart_business_description (
    corp_code VARCHAR(8) PRIMARY KEY REFERENCES dart_corp_codes(corp_code),
    rcept_no VARCHAR(20) NOT NULL,     -- the 사업보고서 filing this text was extracted from --
    -- kept so a later refetch can tell whether a newer annual report has since been filed
    report_nm VARCHAR(200),
    section_text TEXT,                 -- "사업의 내용" chapter text, NULL if extraction failed
    -- (see filing_parser.py -- real DART filings are heterogeneous enough that some won't parse)
    fetched_at DATE NOT NULL
);
