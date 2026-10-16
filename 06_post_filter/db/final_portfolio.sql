CREATE TABLE IF NOT EXISTS final_portfolio (
    as_of_date DATE NOT NULL,  -- the day this was computed, same convention as
    -- 05_black_litterman.portfolio_weights.as_of_date
    label VARCHAR(50) NOT NULL DEFAULT 'default',  -- mirrors portfolio_weights.label
    ticker VARCHAR(20) NOT NULL,
    weight NUMERIC NOT NULL,  -- top-N filtered + issuer-group capped + renormalized
    PRIMARY KEY (as_of_date, label, ticker)
);
