import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline import run_steps

STEPS = [
    ("01_data_load", "scripts/daily_update_krx.py"),
    ("01_data_load", "scripts/daily_update_us.py"),
    ("01_data_load", "scripts/daily_update_index.py"),
    ("01_data_load", "scripts/clean_implausible_returns.py"),  # 거래정지/분할 등으로 생긴 말도 안 되는 일일수익률 NULL 처리
    ("03_sigma_estimation", "scripts/fit_univariate_garch.py"),  # 전체 재추정, 약 29분
    ("02_filter", "scripts/run_filter.py"),  # 유니버스 재선정
    ("03_sigma_estimation", "scripts/fit_dcc.py"),  # a,b 전체 재추정
    ("04_views_estimation", "scripts/daily_update_features.py"),
    ("04_views_estimation", "scripts/update_all_oof.py"),  # 새로 타겟 계산 가능해진 날짜만 OOF에 추가
    ("04_views_estimation", "scripts/train_final_models.py"),  # 실전 배포 모델 전체 재학습 (~100분+,
    # sequence_training_data.py가 캐싱 없이 매번 가격 데이터부터 시퀀스를 다시 계산하는 게 대부분)
    ("04_views_estimation", "scripts/compute_omega.py"),  # 방금 갱신된 챔피언 메타모델의 OOF로 Ω 재계산
]

if __name__ == "__main__":
    run_steps(STEPS)
