# 수동 갱신 순서 (01 ~ 06)

작업 스케줄러 자동화는 컴퓨터를 상시 켜두지 않는 한 의미가 없어서, 당분간은 아래 순서대로 수동 실행한다. 반드시 이 순서를 지켜야 한다 (뒷 단계가 앞 단계 결과를 그대로 가져다 씀).

## 제일 간단한 방법: 통합 스크립트
아래 단계별 명령어를 매번 하나하나 치는 게 번거로워서, 프로젝트 루트 `scripts/`에 순서를 다 묶어둔 스크립트가 있다. **이거 하나만 실행하면 됨**:
```powershell
.venv\Scripts\python.exe scripts\daily_update.py    # 평일용
.venv\Scripts\python.exe scripts\weekly_update.py    # 주말용(전체 재추정 포함)
```
중간에 한 단계라도 실패하면(exit code 비정상) 그 자리에서 멈추고 뒷단계는 안 돌아감 (잘못된 데이터로 이어지는 거 방지). 아래는 그 통합 스크립트가 내부적으로 뭘 하는지 설명 + 단계별로 따로 돌리고 싶을 때 참고용.

---

**사전 조건**: Docker Desktop 켜져 있어야 함 (`docker ps`로 `smallfunds_db` 확인).

**두 가지 주기가 섞여 있다** — GARCH 모수(omega/alpha/beta), 유니버스 필터(모멘텀/샤프비율 등), DCC 모수(a,b) 셋 다 하루 사이 거의 안 바뀌는데 재추정 비용은 크다. 그래서 **이 셋은 주 1회만 재실행**하고, 나머지 날은 그 결과(모수/유니버스)를 재사용해서 "오늘자 값"만 가볍게 갱신한다 (가격만 매일 진짜로 새로 들어옴).

---

## 평일 (매일) — 가격 + 가벼운 갱신만
```powershell
cd 01_data_load
..\.venv\Scripts\python.exe scripts\daily_update_krx.py     # KOSPI/KOSDAQ 가격 + 메타데이터
..\.venv\Scripts\python.exe scripts\daily_update_us.py       # S&P500+Nasdaq100 가격 + 메타데이터
..\.venv\Scripts\python.exe scripts\daily_update_index.py    # KOSPI/S&P500 지수 (04_views_estimation 매크로 피처용)

cd ..\03_sigma_estimation
..\.venv\Scripts\python.exe scripts\update_univariate_garch.py   # 저장된 모수로 cond_vol/std_resid만 갱신 (fix(), 재추정 없음)
..\.venv\Scripts\python.exe scripts\update_dcc.py                 # 지난 주말 유니버스 + 저장된 a,b로 Σ만 갱신 (재추정 없음)

cd ..\04_views_estimation
..\.venv\Scripts\python.exe scripts\daily_update_features.py     # 그날 피처 패널 갱신
..\.venv\Scripts\python.exe scripts\run_inference.py              # 저장된 최종모델로 그날 유니버스 Q 예측, views 테이블 저장

cd ..\05_black_litterman
..\.venv\Scripts\python.exe scripts\run_black_litterman.py        # Σ/Q/Ω 결합 -> 유니버스 전체 비중 (portfolio_weights)

cd ..\06_post_filter
..\.venv\Scripts\python.exe scripts\run_post_filter.py            # 상위 N개 선별 + 발행사/섹터 캡 (final_portfolio)
```
- `daily_update_us.py`는 미국장 마감 후(한국 기준 아침)에 돌려야 그 전날 미국 데이터가 들어옴.
- 01번 둘 다 "DB에 마지막 날짜 다음날 ~ 오늘"까지 자동으로 갭을 메우는 방식이라 며칠 건너뛰어도 따라잡음.
- **평일엔 `02_filter`를 안 돌림** — 유니버스는 지난 주말 선정분을 그대로 씀 (`update_dcc.py`가 `universe_selection`의 최신 `as_of_date`를 그냥 가져다 씀).
- `update_univariate_garch.py`/`update_dcc.py`는 저장된 모수가 없으면(최초 1회, 신규 상장 등) 자동으로 전체 재추정 폴백.
- 05/06은 매일 돌림 — Σ(03)/Q(04)가 매일 갱신되고, 프로젝트 목표 자체가 "매일 밤 자동 갱신 리포트"라서. Ω(04)/유니버스(02)는 주 1회만 바뀌지만 그래도 매일 재조합할 가치가 있음.

## 주말 (주 1회, 아무 요일이나 상관없음 — 순서만 지키면 됨) — 전체 재추정
```powershell
cd 01_data_load
..\.venv\Scripts\python.exe scripts\daily_update_krx.py
..\.venv\Scripts\python.exe scripts\daily_update_us.py
..\.venv\Scripts\python.exe scripts\daily_update_index.py
..\.venv\Scripts\python.exe scripts\clean_implausible_returns.py   # 거래정지/분할 등으로 생긴 말도 안 되는 일일수익률(|r|>50%) NULL 처리

cd ..\03_sigma_estimation
..\.venv\Scripts\python.exe scripts\fit_univariate_garch.py   # 전체 종목(~3,700개) 모수 전체 재추정, 약 29분

cd ..\02_filter
..\.venv\Scripts\python.exe scripts\run_filter.py              # 유니버스 재선정 (주 1회)

cd ..\03_sigma_estimation
..\.venv\Scripts\python.exe scripts\fit_dcc.py                 # a,b 전체 재추정, 약 50초

cd ..\04_views_estimation
..\.venv\Scripts\python.exe scripts\update_all_oof.py           # 새로 타겟 계산 가능해진 날짜만 팩터/시퀀스/메타 OOF에 추가 (전체 재실행 아님)
..\.venv\Scripts\python.exe scripts\train_final_models.py       # 실전 배포 모델 전체 재학습 (~100분+, 시퀀스 데이터 재계산이 대부분)
..\.venv\Scripts\python.exe scripts\compute_omega.py             # 방금 갱신된 챔피언 메타모델의 OOF로 Ω 재계산
```
- `fit_` 스크립트는 모수 재추정 + 그날 cond_vol/Σ 저장까지 한 번에 다 함 — 같은 날 `update_` 버전을 또 돌릴 필요 없음 (중복, 낭비).
- 순서 반드시 지킬 것: GARCH 전체 재추정 → **필터(유니버스 재선정)** → DCC 전체 재추정 (DCC가 그 주의 새 유니버스 + 최신 표준화잔차를 가져다 씀) → OOF 캐치업 → 모델 재학습 → Ω.

---

## 공통 참고
- GARCH 1단계(`fit_`/`update_univariate_garch.py`)는 유니버스/필터와 무관하게 전체 종목 대상이라, 필터 조건이 바뀌어도 다시 돌릴 필요 없음.
- DCC 쪽(`fit_`/`update_dcc.py`) 유니버스 중 GARCH 데이터 없는 티커는 콘솔에 명시적으로 출력되고 자동 제외됨.
- `--label` 인자로 `run_filter.py`/`fit_dcc.py`/`update_dcc.py` 전부 기본값(`default`) 대신 커스텀 유니버스 지정 가능 (서로 DB에서 안 겹침). 커스텀 유니버스는 이 주간/매일 스케줄과 무관하게 필요할 때 그때그때 수동 실행.
