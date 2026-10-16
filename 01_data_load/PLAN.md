# 01. 데이터 로드 - 진행 방향

## 1. 파이썬 환경 + DB 연결 계층
가상환경(프로젝트 공통 `.venv`) 기반으로 `psycopg2`/`SQLAlchemy`를 이용해 Postgres에 붙는 연결 코드 작성.

## 2. 소스별 로더(loader) 작성
- KRX API: 국내 종목 메타데이터 + 가격 데이터 로더
- yfinance: 미국 종목 메타데이터 + 가격 데이터 로더
- "최초 실행 시 과거 이력 전체 백필" 로직과 "매일 하루치만 추가/갱신" 로직을 구분해서 구현

## 3. 스케줄링
로더 스크립트를 자동 실행. 현재는 로컬 Windows 환경이므로 Windows 작업 스케줄러(Task Scheduler) 사용. 이후 서버/클라우드로 이전 시 cron 등으로 교체.

- `SmallFunds_KRX_DailyUpdate`: 매일 21:00 (한국장 마감 후) — `run_daily_update.ps1` → `daily_update_krx.py`
- `SmallFunds_US_DailyUpdate`: 매일 07:30 (미국장 마감 후, 시차 때문에 KRX와 스케줄 분리) — `run_daily_update_us.ps1` → `daily_update_us.py`
- 둘 다 "DB에 마지막으로 들어간 날짜 다음날 ~ 오늘"까지 자동으로 갭을 메우는 방식이라, 컴퓨터를 며칠 꺼놨다 켜도 자동으로 따라잡음
- 전제조건: Docker Desktop이 로그인 시 자동 실행되도록 설정 필요 (Docker Desktop → Settings → General → "Start Docker Desktop when you sign in")

## 현재 상태 (2026-09-23 기준)
- KRX: KOSPI+KOSDAQ 10년치 백필 완료 (2016-09-21 ~)
- US: S&P500 ∪ Nasdaq100 (518종목) 10년치 백필 완료 (2016-09-23 ~)
- 두 시장 모두 같은 `securities`/`daily_prices` 테이블에 `market`/`currency` 컬럼으로 구분되어 적재됨
