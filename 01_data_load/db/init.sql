CREATE TABLE IF NOT EXISTS securities (
    ticker VARCHAR(20) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    market VARCHAR(10) NOT NULL,    -- KR/US
    exchange VARCHAR(20),           -- 거래소
    sector VARCHAR(100),            -- 업종 대분류
    industry VARCHAR(100),          -- 업종 소분류
    currency VARCHAR(10),           -- 통화
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    listed_date DATE,               -- 상장일자
    delisted_date DATE,             -- 상장폐지일자
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS daily_prices (
    ticker VARCHAR(20) NOT NULL REFERENCES securities(ticker),
    date DATE NOT NULL,         -- 날짜
    close_price NUMERIC(20,6),  -- 일별 종가
    adj_close NUMERIC(20,6),    -- 수정 종가
    market_cap NUMERIC(24,2),   -- 시가 총액
    volume BIGINT,              -- 거래량
    daily_return NUMERIC(12,8), -- 수익률    
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
