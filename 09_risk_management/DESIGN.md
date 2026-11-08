# 09_risk_management 설계

## 목표
06의 최종 추천 포트폴리오에 대해 VaR/ES를 산출하고, 그 VaR 모델 자체가 믿을만한지 역사적으로 검증(백테스팅)하고, 한도 초과를 감지해서 일일 리스크 보고서를 자동 생성한다(`daily_update.py` 편입 완료). **스트레스 테스트는 범위에서 뺌 — 시나리오 설계를 나중에 따로 함.**

## 핵심 설계 결정 (사용자 확정)
- **신뢰수준/호라이즌**: 95%·99%, **1일** 기본. 10일은 √10 스케일링으로 파생 (수익률 iid 가정 — 바젤 시장리스크 자본의 표준 관행과 동일 논리, 과장 안 하고 가정 자체를 명시). 바젤 규제자본 승수 계산에 실제로 씀 (아래 "바젤 규제자본 승수" 참고).
- **방법론 2종 병행**: parametric(분산-공분산) + historical simulation. 하나만 쓰지 않는 이유는 실무에서도 모델 리스크를 보려고 여러 방법을 교차 검증하기 때문 — 특히 parametric은 정규분포 가정이라 fat-tail을 과소평가할 수 있는데, historical sim은 분포 가정이 없어서 비교 자체가 의미 있는 진단이 됨.
- **ES(CVaR)**: historical simulation 분포에서 VaR 초과분의 평균. Basel이 FRTB에서 VaR 대신 ES(97.5%)로 자본 기준을 옮긴 추세를 반영해 같이 계산.

## 계산 대상: 06의 최종 포트폴리오
05(BL 원본, 유니버스 전체)가 아니라 **06_post_filter.final_portfolio**(상위 N개 + 발행사/섹터 캡 적용 후) 기준으로 계산 — 실제로 "보유하게 될" 포트폴리오의 리스크를 재야 의미 있음.

## Σ 시간 단위 — 05/07과 다른 선택
05/07은 Q가 21거래일 수익률이라 Σ에 **×21**을 했지만, 여기 VaR는 **1일 호라이즌**이라 **GARCH-DCC가 원래 내는 일별 스케일 Σ를 그대로 씀** (스케일 보정 없음). 이 프로젝트에서 "Σ를 호라이즌에 맞게 스케일링해야 한다"는 교훈을 05에서 직접 겪었기 때문에, 여기서도 반대 방향으로 같은 실수(엉뚱한 스케일 섞어쓰기)를 안 하려고 의식적으로 명시.

## VaR 백테스팅 — 07_backtest 인프라 재사용 + 일별 정밀도
- **재구성 로직 재사용**: "과거 시점 데이터만으로 그 시점의 포트폴리오를 복원"하는 게 07_backtest가 이미 하는 일이라, `07_backtest/src/backtest.py`의 `run_one_fold()`를 확장(비중·trailing 수익률도 함께 반환하도록)해서 가져다 씀 — 같은 BL+필터 로직을 세 번째로 복제하지 않음.
- **일별 실현수익률로 비교**: 07은 21일 간격으로만 리밸런싱하지만(겹침 방지, 07/DESIGN.md 참고), VaR 백테스팅은 통계적 검정력을 위해 샘플이 많아야 함 — 21일에 한 번(~84개)이 아니라, **각 리밸런싱 구간 안에서 실제 보유 비중을 고정한 채 매일의 실현 포트폴리오 수익률**을 따로 계산해서(`가격 데이터에서 직접`) VaR와 비교. 84구간 × 약 21거래일 ≈ 1,700개 일별 관측치 확보 — Kupiec 검정이 의미 있으려면 바젤 권고 기준 자체가 250일 단위라, 84개로는 턴수가 너무 적음.
- **Kupiec POF(Proportion of Failures) 검정**: 실제 손실이 VaR를 넘은 날 수가, 그 신뢰수준에서 "통계적으로 기대되는 빈도"와 유의하게 다른지 우도비 검정.
- **바젤 트래픽라이트**: 최근(또는 전체) 구간의 예외(breach) 횟수를 green(과소추정 아님)/yellow/red(과소추정 의심) 구간으로 분류하는, 규제기관이 실제로 쓰는 판정 기준.

## 한도 설정 + 초과 감지
- **VaR/ES 한도**: parametric 1일 VaR(95%/99%)·ES(99%)가 포트폴리오 가치 대비 일정 비율(기본 각각 5%/8%/10%)을 넘는지 체크.
- **컴포넌트 VaR 집중도 한도**: 06의 발행사/섹터 캡은 "비중" 기준인데, 리스크 관점에선 비중이 작아도 다른 종목들과 상관관계가 높으면 리스크 기여도는 클 수 있음 — 그래서 별도로 **오일러 분해(Euler decomposition)**로 종목별 VaR 기여도를 구해서(파라메트릭 VaR는 비중에 대해 1차 동차함수라, 각 종목 기여분의 합이 정확히 전체 VaR와 같음), 단일 종목이 전체 VaR의 일정 비율(기본 35%)을 넘으면 한도 초과로 플래그.
- `09_risk_management/src/limits.py`의 `DEFAULT_LIMITS` 딕셔너리로 노출 — 02_filter의 `weights`/06의 `--top-n`과 같은 "기본값 있는 파라미터" 원칙, 실제로 한도값은 방법론보다 훨씬 자주 바뀌는 대상이라 상수로 박아두지 않음.
- **일일 리포트**: `scripts/compute_daily_risk_report.py`가 VaR/ES 계산 → 컴포넌트 VaR 저장 → 한도 체크 → 콘솔에 breach 여부까지 출력. `daily_update.py`에 06 다음 단계로 편입.

## (보류) VaR 방법론 비교 — 정규분포 vs 꼬리 보정 (Student-t, Cornish-Fisher)
parametric 99% VaR가 Kupiec 검정상 통계적으로 유의한 under-coverage(p=0.0005)로 나온 걸 교정하려고 Student-t/Cornish-Fisher VaR를 한 번 구현·연결해봤다가 **롤백함**. 이유: 두 방법 다 VaR 숫자 자체는 더 크게(보수적으로) 나왔지만(정규 5.00% → Student-t 5.49% → Cornish-Fisher 6.30%), 바젤 승수 계산(`n_breaches_250d` → k)은 여전히 **parametric 백테스트 결과만 보고** 있어서 — 새 VaR를 끼워 넣어도 승수(k=4, red)가 안 바뀌니 규제자본이 줄어들기는커녕 오히려 커질 수 있음(VaR는 커졌는데 승수는 그대로라서). 이걸 제대로 연결하려면 `run_var_backtest.py`(1,204일 재현)에 Student-t도 같이 백테스트하도록 확장해서, "더 정확한 모델 → 실제로 breach가 줄어듦 → 승수가 내려감"까지 검증해야 의미가 있음 — 꽤 무거운 작업이라 나중으로 미룸. 재개할 때는 `var_models.py`에 `student_t_var`/`student_t_es`/`cornish_fisher_var`/`distribution_moments`/`estimate_t_dof`를 다시 추가하고(git 히스토리 참고), `run_var_backtest.py`의 백테스트 루프에 세 번째 메서드로 끼워 넣는 것부터 시작.

## 바젤 규제자본 승수 (1996 시장리스크 개정안 / 바젤 II 내부모형법)
- VaR 백테스팅 결과(breach 횟수)가 단순히 "모델 신뢰도 참고용"에서 그치지 않고, **실제로 얼마나 더 보수적으로 버퍼를 잡아야 하는지"**까지 이어지도록 바젤의 공식 승수(k) 계산을 구현.
- **k 결정**: 최근 **250 거래일**(바젤이 명시한 윈도우, `kupiec.py`의 일반화된 이항분포 traffic_light와는 다른 계산 — 그건 "통계적으로 유의한지"를 보는 거고, 이건 바젤이 못박은 절대 breach 횟수 구간표를 그대로 따름)의 parametric 99% breach 횟수를 바탕으로 공식 표 적용: 0-4회 k=3.00(green), 5~9회 k=3.40~3.85(yellow, 단계별), 10회 이상 k=4.00(red).
- **10일 VaR**: 바젤 규제자본은 1일이 아니라 **10일 99% VaR** 기준 — `basel_capital.scale_to_10day()`가 √10 스케일링으로 derive (앞서 "신뢰수준/호라이즌" 항목에서 미뤄뒀던 부분을 실제로 구현).
- **필요 자본**: `max(전일 10일-VaR, k × 최근 60일 10일-VaR 평균)` — 바젤 공식 그대로. 60일 평균은 `var_estimates`에 쌓인 과거 parametric 99% VaR로 계산. **처음엔(09가 daily_update.py에 막 편입됐을 때) 과거 데이터가 1일치뿐이라 평균이 그 한 값으로 퇴화해서 숫자가 과장되게 나왔음**(61.26%) — `scripts/backfill_var_history.py`로 해결: `var_backtest_daily`에 이미 있던 점시점-정직 historical replay(1,204일치, `run_var_backtest.py`가 생성)의 최근 60일을 `var_estimates`에 그대로 복사해서 즉시 진짜 60일 평균을 확보(56.51%로 안정화). 이후 `daily_update.py`가 매일 돌면서 쌓이는 라이브 데이터가 자연히 그 자리를 대체해감. `n_days_in_average < 20`이면 콘솔/웹에 "아직 대표성 없음" 경고가 뜨는 로직은 그대로 남겨둠(혹시 모를 재현/초기화 상황 대비).
- `09_risk_management/src/basel_capital.py`에 순수함수로 구현 (`capital_multiplier`, `scale_to_10day`, `required_capital`), `compute_daily_risk_report.py`가 매일 계산·저장.
- **등급별 경고 문구**: `capital_zone(n_breaches_250d)`가 green/yellow/red를 판정하고 `ZONE_MESSAGES`에서 그에 맞는 한글 설명을 가져옴 (red: "리스크 모형 위험 -- 꼬리 리스크를 심각하게 과소추정..."). DB엔 안 저장 — `n_breaches_250d`만 저장해두면 등급/문구는 언제든 그 값에서 순수하게 재계산 가능해서, 중복 저장 대신 콘솔(`compute_daily_risk_report.py`)과 웹(`08_web`) 양쪽에서 같은 값으로부터 각자 계산.

## DB 스키마
- `var_estimates(as_of_date, label, method, confidence, var_value, es_value)` — PK (as_of_date, label, method, confidence). `method` = `parametric`|`historical`. 라이브 일일 리포트와 백테스트 양쪽에서 같은 테이블 사용 (백테스트는 `as_of_date`가 각 리밸런싱 시점).
- `var_backtest_daily(date, label, confidence, var_value, realized_return, breach)` — 일별 실현수익률 vs 그 구간에 적용되던 VaR 비교 기록. `breach` = 실현손실이 VaR 초과했는지.
- `var_backtest_summary(label, confidence, method, n_days, n_breaches, expected_breach_rate, kupiec_lr_stat, kupiec_p_value, traffic_light)` — Kupiec 검정 결과 요약. `weekly_update.py`에 편입(OOF가 매주 갱신되니 같은 주기로 재검증).
- `component_var(as_of_date, label, ticker, confidence, component_var, share_of_total)` — 종목별 VaR 기여도.
- `risk_limit_checks(as_of_date, label, limit_name, limit_value, observed_value, breached)` — 한도별 체크 결과, 위반 여부와 무관하게 매번 전체 기록 (빈 결과가 "다 통과"인지 "실행 자체가 안 됐는지" 구분 안 되는 걸 피하려고).
- `basel_capital_requirement(as_of_date, label, n_breaches_250d, multiplier, var_10day, avg_var_10day, n_days_in_average, required_capital)` — 위 "바젤 규제자본 승수" 계산 결과.

## 알려진 버그/교훈
- **바젤 트래픽라이트 스케일링 실수**: 처음엔 Basel의 n=250 기준 절대 예외 횟수(0-4/5-9/10+)를 샘플 크기 비율로 그냥 선형 스케일했는데, 통계적 변동성은 √n으로 커지지 샘플 수에 비례하지 않아서 더 큰 표본(1,204일)에 그 경계값을 적용하니 "95% VaR가 breach 68건으로 red"라는, **Kupiec 검정(p=0.31, 전혀 유의하지 않음)과 모순되는 결과**가 나옴. 이항분포 누적확률(`binom.cdf`) 기반으로 바꿔서 해결 — n=250, p=0.01에서 Basel 발표 경계값(4/9)을 정확히 재현하는지 검증 후 적용.

## 진행 순서
1. VaR/ES 산출 + 백테스팅(Kupiec+트래픽라이트) — 완료.
2. 한도 설정 + 초과 감지 + 일일 리스크 보고서 — 완료.
3. 스트레스 테스트 — 시나리오 설계를 먼저 하고 나서.

## 개선 과제 (TODO — 하려다 안 한 것들)
- **스트레스 테스트** — 원래 3개 요구사항 중 하나였는데, 시나리오 설계를 나중에 사용자와 같이 하기로 명시적으로 보류함. 아직 시작 안 함.
- **Student-t / Cornish-Fisher VaR를 바젤 승수와 완전히 연동** — 한 번 구현해서 VaR/ES 카드까지 웹에 붙였다가 롤백함(위 "(보류) VaR 방법론 비교" 섹션 참고). VaR 숫자만 바꿔 끼우는 걸로는 바젤 승수(k)가 안 움직여서 의미가 없고, `run_var_backtest.py`의 1,204일 재현에 Student-t 백테스트를 추가해서 "실제로 breach가 줄어 승수가 내려가는지"까지 검증해야 제대로 된 개선이 됨 — 무거운 작업이라 미룸.
