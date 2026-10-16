CREATE TABLE IF NOT EXISTS views (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',  -- mirrors universe_selection.label (default vs custom one-off)
    ticker VARCHAR(20) NOT NULL,
    q NUMERIC NOT NULL,  -- final meta-model prediction (21-trading-day excess return), BL's Q
    omega NUMERIC,  -- BL's per-ticker view uncertainty (see omega_estimates/compute_omega.py);
    -- nullable since a ticker can get a Q before it has any OOF history to estimate Ω from (e.g.
    -- a brand-new listing). Copied in from omega_estimates at inference time rather than joined
    -- at BL-consumption time, so a views row is a complete, self-contained (Q, Ω) pair.
    PRIMARY KEY (as_of_date, label, ticker)
);
