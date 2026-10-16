import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline import run_steps

STEPS = [
    ("01_data_load", "scripts/daily_update_krx.py"),
    ("01_data_load", "scripts/daily_update_us.py"),
    ("01_data_load", "scripts/daily_update_index.py"),
    ("03_sigma_estimation", "scripts/update_univariate_garch.py"),
    ("03_sigma_estimation", "scripts/update_dcc.py"),
    ("04_views_estimation", "scripts/daily_update_features.py"),
    ("04_views_estimation", "scripts/run_inference.py"),  # 그날 유니버스에 대해 Q 예측, views 테이블 저장
    ("05_black_litterman", "scripts/run_black_litterman.py"),  # Sigma/Q/Omega 결합 -> 유니버스 전체 비중
    ("06_post_filter", "scripts/run_post_filter.py"),  # 상위 N개 선별 + 발행사/섹터 캡
]

if __name__ == "__main__":
    run_steps(STEPS)
