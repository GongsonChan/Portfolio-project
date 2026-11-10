CREATE TABLE IF NOT EXISTS business_similarity (
    as_of_date DATE NOT NULL,
    label VARCHAR(50) NOT NULL DEFAULT 'default',
    ticker_i VARCHAR(20) NOT NULL,
    ticker_j VARCHAR(20) NOT NULL,
    similarity NUMERIC NOT NULL,  -- cosine similarity of ko-sroberta-multitask embeddings of
    -- each ticker's "사업의 내용" text (dart_business_description.section_text). Full matrix,
    -- both directions + diagonal (=1.0), same convention as 03_sigma_estimation.covariance_matrix.
    -- Business-model similarity, NOT return correlation -- complements 09_risk_management's
    -- component VaR (see 10/DESIGN.md's "왜 필요한가")
    PRIMARY KEY (as_of_date, label, ticker_i, ticker_j)
);
