-- NOTE: 01_data_load/db/init.sql only runs on first container init, so this
-- table was created manually via psql. Kept here for reference/reproducibility.
CREATE TABLE IF NOT EXISTS universe_selection (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',  -- 'default' = nightly automated run; anything else = a custom one-off
    ticker VARCHAR(20) NOT NULL REFERENCES securities(ticker),
    rank INTEGER NOT NULL,
    score NUMERIC,
    PRIMARY KEY (as_of_date, label, ticker)
);
