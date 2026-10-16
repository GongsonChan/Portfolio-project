CREATE TABLE IF NOT EXISTS portfolio_weights (
    as_of_date DATE NOT NULL,  -- the day this portfolio was computed, not the as_of_date of any
    -- one input (Sigma/Q/Omega each carry their own date -- see 05_black_litterman/DESIGN.md)
    label VARCHAR(50) NOT NULL DEFAULT 'default',  -- mirrors views.label/covariance_matrix.label
    ticker VARCHAR(20) NOT NULL,
    weight NUMERIC NOT NULL,  -- final long-only weight after top-N filtering + renormalization
    PRIMARY KEY (as_of_date, label, ticker)
);
