CREATE TABLE IF NOT EXISTS var_estimates (
    as_of_date DATE NOT NULL,  -- live daily report date, or a backtest period's start date
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    method VARCHAR(20) NOT NULL,  -- 'parametric' | 'historical'
    confidence NUMERIC NOT NULL,  -- 0.95 | 0.99
    var_value NUMERIC NOT NULL,   -- 1-day VaR, positive = loss magnitude as a fraction of portfolio value
    es_value NUMERIC,             -- Expected Shortfall -- only meaningful for 'historical' (needs the tail sample)
    PRIMARY KEY (as_of_date, label, method, confidence)
);

CREATE TABLE IF NOT EXISTS var_backtest_daily (
    date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    confidence NUMERIC NOT NULL,
    var_value NUMERIC NOT NULL,     -- the VaR in force that day (fixed for the whole holding
    -- period it falls in -- see DESIGN.md)
    realized_return NUMERIC NOT NULL,
    breach BOOLEAN NOT NULL,        -- -realized_return > var_value
    PRIMARY KEY (date, label, confidence)
);

CREATE TABLE IF NOT EXISTS var_backtest_summary (
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    method VARCHAR(20) NOT NULL,
    confidence NUMERIC NOT NULL,
    n_days INT NOT NULL,
    n_breaches INT NOT NULL,
    expected_breach_rate NUMERIC NOT NULL,  -- 1 - confidence
    kupiec_lr_stat NUMERIC,
    kupiec_p_value NUMERIC,
    traffic_light VARCHAR(10) NOT NULL,  -- 'green' | 'yellow' | 'red'
    computed_at DATE NOT NULL,
    PRIMARY KEY (label, method, confidence)
);

CREATE TABLE IF NOT EXISTS component_var (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    ticker VARCHAR(20) NOT NULL,
    confidence NUMERIC NOT NULL,
    component_var NUMERIC NOT NULL,  -- Euler decomposition of parametric VaR (see var_models.py) --
    -- these sum exactly to that day's parametric var_estimates row for the same confidence
    share_of_total NUMERIC NOT NULL,
    PRIMARY KEY (as_of_date, label, ticker, confidence)
);

CREATE TABLE IF NOT EXISTS risk_limit_checks (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    limit_name VARCHAR(100) NOT NULL,
    limit_value NUMERIC NOT NULL,
    observed_value NUMERIC NOT NULL,
    breached BOOLEAN NOT NULL,
    PRIMARY KEY (as_of_date, label, limit_name)
);

CREATE TABLE IF NOT EXISTS basel_capital_requirement (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    n_breaches_250d INT NOT NULL,       -- trailing-250-trading-day exception count (parametric 99%,
    -- from var_backtest_daily) -- Basel's own prescribed window, not the generalized traffic_light()
    multiplier NUMERIC NOT NULL,        -- k, from Basel's official step table (3.00 - 4.00)
    var_10day NUMERIC NOT NULL,         -- today's 1-day parametric 99% VaR, sqrt(10)-scaled
    avg_var_10day NUMERIC NOT NULL,     -- rolling average of var_10day over up to the last 60 days
    n_days_in_average INT NOT NULL,     -- how many prior days actually went into avg_var_10day
    -- (< 60 while daily history is still accumulating -- see DESIGN.md)
    required_capital NUMERIC NOT NULL,  -- max(prior day's var_10day, multiplier * avg_var_10day)
    PRIMARY KEY (as_of_date, label)
);
