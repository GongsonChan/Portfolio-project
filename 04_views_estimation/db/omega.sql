CREATE TABLE IF NOT EXISTS omega_estimates (
    ticker VARCHAR(20) NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',  -- mirrors views.label (default vs custom one-off)
    computed_at DATE NOT NULL,  -- when this row was (re)computed, not an as_of_date time series --
    -- there's only ever one current Ω per (ticker, label), refreshed in place as new OOF folds
    -- accumulate, same "latest state" pattern as model_registry/dcc_params rather than views'
    -- daily time series.
    omega NUMERIC NOT NULL,       -- Var(target - meta-model pred), shrunk toward the pooled
    -- variance for tickers with few OOF samples (see compute_omega.py)
    n_samples INT NOT NULL,       -- how many trailing-window OOF rows this ticker actually had
    PRIMARY KEY (ticker, label)
);
