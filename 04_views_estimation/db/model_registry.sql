CREATE TABLE IF NOT EXISTS model_registry (
    model_name VARCHAR(50) NOT NULL,   -- e.g. 'lightgbm', 'gru', 'ridge' (meta)
    model_type VARCHAR(20) NOT NULL,   -- 'factor' | 'sequence' | 'meta'
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    trained_at TIMESTAMP NOT NULL,
    oof_corr NUMERIC,                  -- OOF pred-target correlation this config was validated at
    file_path TEXT NOT NULL,
    -- model_type is part of the key because 'ridge' is a factor-model name AND a meta-model
    -- candidate name -- without model_type they'd collide on the same (model_name, label) row.
    PRIMARY KEY (model_name, model_type, label)
);
