-- NOTE: 01_data_load/db/init.sql only runs on first container init, so these
-- tables were created manually via psql. Kept here for reference/reproducibility.
CREATE TABLE IF NOT EXISTS garch_estimates (
    ticker VARCHAR(20) NOT NULL REFERENCES securities(ticker),
    date DATE NOT NULL,
    cond_vol NUMERIC,   -- conditional volatility, % scale (returns were *100 before fitting)
    std_resid NUMERIC,  -- standardized residual (return / cond_vol), used as DCC input
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS garch_params (
    ticker VARCHAR(20) NOT NULL REFERENCES securities(ticker),
    fit_date DATE NOT NULL,  -- date the model was fit (most recent date in the fitting window)
    mu NUMERIC,
    omega NUMERIC,
    alpha NUMERIC,
    beta NUMERIC,
    PRIMARY KEY (ticker, fit_date)
);

CREATE TABLE IF NOT EXISTS dcc_params (
    label VARCHAR(50) NOT NULL,      -- which universe_selection label this was fit on
    fit_date DATE NOT NULL,
    a NUMERIC,
    b NUMERIC,
    converged BOOLEAN,
    log_likelihood NUMERIC,
    PRIMARY KEY (label, fit_date)
);

CREATE TABLE IF NOT EXISTS covariance_matrix (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',  -- mirrors universe_selection.label (default vs custom one-off)
    ticker_i VARCHAR(20) NOT NULL,
    ticker_j VARCHAR(20) NOT NULL,
    covariance NUMERIC,
    PRIMARY KEY (as_of_date, label, ticker_i, ticker_j)
);
