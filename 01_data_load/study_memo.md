# Study Memo - 01_data_load

## 0. DB 컨테이너 실행 가이드
Docker Desktop을 먼저 켠 상태에서 `01_data_load` 폴더로 이동 후 실행.

```
docker compose up -d      # 컨테이너 시작 (백그라운드)
docker compose down       # 컨테이너 중지 및 삭제 (데이터는 volume에 남아있음)
docker compose ps         # 상태 확인
docker compose logs -f db # 로그 실시간 확인
```

- `init.sql`은 최초 실행(볼륨이 비어있을 때) 한 번만 적용됨. 스키마를 바꾸고 싶으면 `docker compose down -v`로 볼륨까지 지우고 다시 `up -d` (기존 데이터도 같이 삭제되니 주의).
- 컴퓨터 재부팅 후에는 Docker Desktop만 다시 켜면 `restart: unless-stopped` 설정 덕분에 컨테이너가 자동으로 다시 뜸.

### 테이블 생성 여부 확인
```
docker exec smallfunds_db psql -U <POSTGRES_USER> -d <POSTGRES_DB> -c "\dt" 
```
`<POSTGRES_USER>`, `<POSTGRES_DB>`는 `.env`에 적어둔 값으로 대체. 결과에 `securities`, `daily_prices`가 나오면 정상.

`\dt`는 "테이블 목록"만 보여주는 것("(N rows)"은 테이블 개수), 실제 데이터 건수를 보려면 `SELECT count(*)`를 따로 써야 함.

### 자주 쓰는 조회 예시
```sql
-- 특정 종목 시계열
SELECT * FROM daily_prices WHERE ticker = '005930' ORDER BY date;

-- 특정 날짜 시가총액 상위 10개
SELECT ticker, close_price, market_cap FROM daily_prices
WHERE date = '2026-09-18' ORDER BY market_cap DESC LIMIT 10;

-- 날짜별 데이터 건수 (얼마나 쌓였는지 확인)
SELECT date, count(*) FROM daily_prices GROUP BY date ORDER BY date DESC LIMIT 10;
```
데이터가 계속 쌓이는 테이블(`daily_prices`)은 CSV로 통째로 뽑기보다 이렇게 필요한 만큼만 SQL로 조회하는 게 기본. 눈으로 훑어보고 싶을 땐 `scripts/export_csv.py` 사용.

## SQLAlchemy
파이썬에서 DB(Postgres, MySQL, SQLite 등)를 다루기 쉽게 해주는 라이브러리예요. 원래는 SQL 문을 직접 문자열로 짜서 psycopg2 같은 드라이버에 보내야 하는데, SQLAlchemy는 그 위에 한 겹 씌워서 파이썬스럽게(객체/함수 호출로) DB를 다룰 수 있게 해줘요. 여러 DB 종류를 같은 방식의 코드로 다룰 수 있게 추상화해주는 게 핵심 가치입니다.

## create_engine
SQLAlchemy에서 "이 DB에 접속할 준비가 된 커넥션 관리자"를 만드는 함수예요. db.py에서 DATABASE_URL(유저/비번/주소/DB명이 합쳐진 문자열)을 넣어서 engine을 만들면, 이 engine 객체가 필요할 때마다 실제 연결을 열고 닫는 걸 알아서 관리해줘요. 만들었다고 바로 접속하는 건 아니고, "이렇게 접속할 거야"라는 설계도만 만들어두는 거고, 실제 접속은 engine.connect() 할 때 일어나요.

## yf.download의 end 파라미터는 exclusive
`yf.download(tickers, start=..., end=...)`에서 `end`로 지정한 날짜는 결과에 **포함되지 않음** (그 날짜 직전까지만). 그래서 US 일일 갱신을 저녁에 돌리면 "어제까지"만 받아오게 되는데, 이건 버그가 아니라 오히려 의도에 맞음 — 미국 장은 한국 저녁 시간엔 아직 마감 전이라 어차피 최신 데이터가 없고, 스케줄을 미국장 마감 이후(한국 아침)로 잡으면 이 exclusive 특성이 "정확히 어제(=가장 최근 완료된 거래일)까지" 가져오는 것과 자연스럽게 맞아떨어짐.

## python-dotenv
.env 파일에 적어둔 KEY=VALUE 값들을 읽어서 파이썬의 환경변수로 등록해주는 라이브러리예요. load_dotenv(경로)를 호출하면 그 .env 파일 내용이 마치 원래 시스템에 설정된 환경변수인 것처럼 파이썬 프로세스 안에서 접근 가능해져요. 이게 없으면 .env 파일은 그냥 텍스트 파일일 뿐, 파이썬이 자동으로 읽어주지 않아요.

## os.environ
파이썬 표준 라이브러리 os 모듈에 있는, 현재 프로세스가 볼 수 있는 모든 환경변수를 담고 있는 딕셔너리 같은 객체예요. os.environ["POSTGRES_USER"]처럼 쓰면 그 키의 값을 가져오는데, 키가 없으면 에러(KeyError)가 나요. 그래서 db.py에서는 "반드시 있어야 하는" 값들(POSTGRES_USER, PASSWORD, DB)에 이걸 썼어요 — 없으면 바로 에러 내서 알아채라는 의도예요.

## os.getenv
os.environ이랑 비슷한데, 키가 없을 때 에러 대신 기본값을 줄 수 있어요. os.getenv("DB_HOST", "localhost")는 "DB_HOST라는 환경변수가 있으면 그 값을, 없으면 "localhost"를 써라"는 뜻이에요. .env에 굳이 안 적어놔도 되는(거의 항상 고정인) 값들에 썼습니다.

## inspect
SQLAlchemy가 제공하는 함수인데, engine을 넣으면 "지금 이 DB 안에 어떤 테이블/컬럼/인덱스가 있는지 조사해서 알려주는" 조사관 객체(Inspector)를 돌려줘요. test_connection.py에서 inspector.get_table_names()를 호출해서 "DB 안에 있는 테이블 이름 목록 줘"라고 물어본 거고, 그 결과가 ['securities', 'daily_prices']로 나온 거예요. DB 스키마를 코드로 확인하고 싶을 때 쓰는 도구입니다.
