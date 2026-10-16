# 05_black_litterman 설계

## 목표
03의 Σ(공분산행렬)와 04의 Q/Ω(뷰, 뷰 불확실성)를 Black-Litterman으로 결합해 사후 기대수익률을 구하고, 그걸로 최종 포트폴리오 비중까지 산출한다.

## 라이브러리: PyPortfolioOpt
BL 수식(사전분포 역산, 사후분포 결합)과 평균-분산 최적화를 직접 구현하는 대신 `pypfopt`를 씀 — 둘 다 잘 테스트된 표준 구현이 있고, 우리 Q가 전부 "종목별 절대 뷰"(상대 뷰 없음)라 `BlackLittermanModel(absolute_views=...)` 경로가 그대로 맞아떨어짐 (P행렬을 직접 만들 필요 없음).

## 05와 06의 역할 분리
처음엔 05 안에서 상위 N개 추리기 + 발행사/섹터 비중 캡까지 다 하려고 했는데, 섹터 분류 데이터를 어디서 가져올지 파다가(KRX 공식 API엔 없음, 결국 LLM으로 분류) 범위가 너무 커져서 **분리하기로 함**: 05는 "BL이 전체 유니버스를 어떻게 보는지"까지만 내고 끝. 상위 N개 선별, 발행사/섹터 집중도 캡 같은 **사후 가공은 `06_post_filter`가** 05의 출력을 입력으로 받아서 처리.

## 핵심 설계 결정 (사용자 확정)
- **π (prior)**: 유니버스 시가총액 가중으로 역산. `pypfopt.black_litterman.BlackLittermanModel(pi="market", market_caps=...)`가 내부에서 `δ·Σ·w_mkt` 계산을 해줌.
- **δ (risk aversion)**: 고정값 2.5 (관행값). π 역산과 최종 효율적 프론티어 최적화(`max_quadratic_utility`) 둘 다에 동일하게 씀.
- **τ**: 고정값 0.05 (관행값).
- **최적화 제약**: long-only, 종목별 캡 없음 — `EfficientFrontier`의 기본 `weight_bounds=(0, 1)`을 그대로 씀. (상위 N개 선별·발행사/섹터 캡은 06으로 이동)

## 입력 데이터 — "각자 자기 최신 날짜" 원칙
오늘(04 디버깅) 겪은 교훈을 그대로 적용: 하나의 공유된 as_of_date로 모든 입력을 억지로 맞추지 않음.
- **Σ**: `covariance_matrix`에서 해당 label의 가장 최근 `as_of_date` 스냅샷을 그대로 씀 — DCC가 이미 유니버스 전체를 하나의 일관된 시점으로 계산해서 저장하는 단일 객체이므로(03/DESIGN.md 참고), 여기서 추가로 쪼갤 필요 없음.
- **Q, Ω**: `views`는 이제 티커마다 자기 자신의 최신 cond_vol-complete 날짜로 저장되어 있음(04 오늘 수정). BL은 날짜를 통일하려 하지 않고 **유니버스의 각 티커에 대해 `views`의 최신 행을 그대로 가져옴** (`MAX(as_of_date)` per ticker). `omega_estimates`는 원래부터 날짜 시계열이 아니라 "현재 상태" 테이블이라 티커당 한 행만 있음 — 그대로 조회.
- **시가총액**: `daily_prices.market_cap`에서 각 티커의 최신 값 (01 원본 기준, 마찬가지로 per-ticker).
- Σ의 유니버스와 Q/Ω/시가총액의 유니버스가 어긋날 수 있음(예: Σ엔 있는데 아직 Q가 없는 신규 편입 종목) — 네 집합의 **교집합**만 BL에 넣음. 빠지는 티커가 있으면 몇 개/누구인지 로그로 남김 (04의 "skipping N tickers" 패턴과 동일).

## Σ의 시간 단위 보정 (`HORIZON_DAYS`)
`covariance_matrix`는 GARCH의 **일별(daily)** cond_vol로 지은 Σ인데(03의 `dcc.build_covariance` 참고), Q는 **21거래일** 초과수익률 — 그대로 섞으면 Σ가 Q 대비 ~21배 작아서 최적화의 리스크 패널티가 턱없이 약해지고, Q 높은 종목에 과도하게 쏠림(캡 없이 돌렸을 때 000660/005930이 각각 30%+ 가져가려던 현상의 원인 중 하나). 분산은 시간에 선형으로 늘어난다는 표준 가정(연환산 때 ×252 하는 것과 같은 원리)으로 **Σ에 21을 곱해서** BL에 넘김.

## 처리 흐름 (`run_black_litterman.py`)
1. Σ, Q, Ω, 시가총액을 위 원칙대로 조회 → 교집합 티커로 정렬
2. `market_implied_risk_aversion` 대신 **고정 δ=2.5**를 바로 씀 (확정 사항)
3. `BlackLittermanModel(cov_matrix=Σ, pi="market", market_caps=시가총액, absolute_views=Q, omega=diag(Ω), tau=0.05, risk_aversion=2.5)` → `bl_returns()`, `bl_cov()`로 사후 μ, Σ 추출
4. `EfficientFrontier(μ_bl, Σ_bl, weight_bounds=(0,1))` → `max_quadratic_utility(risk_aversion=2.5)` → 전체 유니버스 비중
5. `portfolio_weights` 테이블에 **유니버스 전체 그대로** 저장 (as_of_date=오늘 실행일, label) — 상위 N개 추리기/캡은 안 함

## DB 스키마
- `portfolio_weights(as_of_date, label, ticker, weight)` — PK (as_of_date, label, ticker). `as_of_date`는 "이 포트폴리오를 계산한 날"이지, 입력 데이터들의 날짜가 아님 (Σ/Q/Ω 각자 자기 날짜를 갖고 있고, 그 입력들의 날짜는 저장하지 않음 — 필요하면 covariance_matrix/views 쪽을 그 시점 기준으로 다시 조회하면 됨). 여기 저장된 비중은 **06_post_filter가 가공하기 전의 원본**.

## 라벨(label) 시스템
03/04와 동일한 패턴 재사용 — `label='default'`는 나이트리 자동 실행, 그 외는 사용자 커스텀 실행(one-off).

## 향후 과제 (지금 범위 아님)
- 상위 N개 선별, 발행사/섹터 집중도 캡은 `06_post_filter` 참고.
- 일일 보고서/LLM 정성 필터링은 06 다음 단계.
- δ를 시장 데이터에서 역산하는 옵션(`market_implied_risk_aversion`)은 지금은 안 씀 — 고정값으로 충분하다고 판단했지만, 나중에 필요하면 함수는 이미 pypfopt에 있음.
