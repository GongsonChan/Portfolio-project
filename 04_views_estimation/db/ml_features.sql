-- NOTE: 01_data_load/db/init.sql only runs on first container init, so this
-- table was created manually via psql. Kept here for reference/reproducibility.
CREATE TABLE IF NOT EXISTS ml_features (
    as_of_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL REFERENCES securities(ticker),
    mom_1m NUMERIC, mom_3m NUMERIC, mom_6m NUMERIC, mom_12_1 NUMERIC, reversal_1w NUMERIC,
    vol_1m NUMERIC, vol_3m NUMERIC, vol_1y NUMERIC, mdd NUMERIC, skew_1m NUMERIC, max_ret_1m NUMERIC,
    liquidity NUMERIC, amihud NUMERIC,
    rsi_14 NUMERIC, ma_gap_50 NUMERIC, ma_gap_200 NUMERIC, bb_pctb NUMERIC, volume_surge NUMERIC,
    size NUMERIC, high_proximity NUMERIC,
    beta NUMERIC, idio_vol NUMERIC, cond_vol NUMERIC,
    kospi_ret_1w NUMERIC, kospi_ret_1m NUMERIC, kospi_ret_3m NUMERIC, kospi_vol NUMERIC,
    spx_ret_1w NUMERIC, spx_ret_1m NUMERIC, spx_ret_3m NUMERIC, spx_vol NUMERIC,
    PRIMARY KEY (as_of_date, ticker)
);
-- 횡단면 순위(피처 설계 G) 는 여기 저장 안 함 — 그때그때 사용하는 유니버스 기준으로
-- 학습/추론 시점에 raw 값으로부터 계산해서 씀 (유니버스가 바뀔 수 있어서 고정 저장 안 함).
