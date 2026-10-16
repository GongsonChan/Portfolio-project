CREATE TABLE IF NOT EXISTS backtest_period_returns (
    as_of_date DATE NOT NULL,  -- the OOF fold's rebalance date
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    portfolio_return NUMERIC NOT NULL,  -- sum(weight_i * target_i) -- target is the fold's
    -- already-realized 21-trading-day forward excess return, so no extra price lookup needed
    n_holdings INT NOT NULL,
    PRIMARY KEY (as_of_date, label)
);
