# Diagnostic before/after summary

## Git status

- Feature branch: `feature/diagnostic-routing-output-fix`
- Merged into: `dev`
- Dev merge commit: `fdb6b9b merge diagnostic routing output fixes`
- Server branch after deployment: `dev`
- Server API health check after deployment: `{"status":"ok"}`

## Scope

이번 수정은 테스트 질문을 반복 실행하면서 발견한 다음 문제를 대상으로 했다.

- 정기공시 재무지표 답변이 필요한 행/열보다 많이 출력되는 문제
- 연결/별도, 3개월/누적, 당기/전기 컬럼 선택이 섞이는 문제
- 재무상태표 계정명이 공백 포함 표기일 때 검색 또는 검증에서 빠지는 문제
- 이벤트 질문이 재무지표 단어 때문에 정기공시로 잘못 라우팅되는 문제
- 주요사항/거래소 공시에서 최신 공시 또는 본문 결정표보다 정정신고 표가 먼저 나오는 문제
- 보유공시에서 질문에 명시된 보고자와 다른 보고자의 답변을 내는 문제
- 일반근거/사업설명 답변이 너무 길게 출력되는 문제

## Before and after by question

### 1. 현대자동차 2025년 1분기 연결 매출액

Question:

```text
현대자동차 2025년 1분기 연결 매출액이 얼마야?
```

Before:

- 수익 주석 표와 전체 손익계산서가 함께 출력됐다.
- `매출액`뿐 아니라 `매출원가`, `영업이익`, `주당이익` 등 질문과 무관한 행까지 포함됐다.
- 연결/별도 구분보다 검색 상위 chunk의 본문 전체를 보여주는 경향이 있었다.

After:

- 연결 손익계산서의 `매출액` 행만 출력한다.
- 단일 기간 질문은 `제 58 기 1분기 / 3개월` 컬럼만 남긴다.
- 답변 예시는 `매출액 | 44,407,761`처럼 필요한 값 중심으로 정리된다.

Changed:

- `app/reasoning/periodic_metric_view.py`
- `app/reasoning/periodic_evidence_selector.py`
- `app/generation/answer_generator.py`

### 2. 현대자동차 2025년 1분기 누적 연결 매출액

Question:

```text
현대자동차 2025년 1분기 누적 연결 매출액은?
```

Before:

- 질문에 `누적`이 있어도 3개월/누적 컬럼이 같이 출력되거나 선택이 불명확했다.
- `누적` 같은 단어가 lexical 검색어에 남아 무관한 주석 chunk recall을 높일 수 있었다.

After:

- 질문에 `누적` 또는 `누계`가 있으면 누적 컬럼만 남긴다.
- 질문에 누적 표현이 없으면 기본적으로 3개월 컬럼을 사용한다.

Changed:

- `app/reasoning/query_understanding.py`
- `app/reasoning/periodic_metric_view.py`

### 3. 삼성전자 2025년 1분기 연결 매출액

Question:

```text
삼성전자 2025년 1분기 연결 매출액은?
```

Before:

- `매출액 (주26)`처럼 주석 번호가 붙은 손익계산서 행을 정확한 `매출액` 행으로 인식하지 못했다.
- 부문별 매출 주석 또는 다른 수익 관련 표가 우선될 수 있었다.

After:

- `(주26)` 같은 주석 suffix를 제거하고 행 이름을 비교한다.
- 연결 손익계산서의 정확한 `매출액` 행을 선택한다.

Changed:

- `app/reasoning/periodic_metric_view.py`
- `app/reasoning/periodic_evidence_selector.py`

### 4. NAVER 2025년 1분기 연결 매출액

Question:

```text
NAVER 2025년 1분기 연결 매출액은?
```

Before:

- NAVER 손익계산서에서 매출 성격의 행이 `영업수익 (주5)`으로 표시되어 `매출액` 질문에 매칭되지 않았다.
- 결과가 `확인 필요`로 떨어졌다.

After:

- `영업수익`을 `매출액` alias로 처리한다.
- `영업수익 (주5) | 2,786,783,351,907` 행을 출력한다.

Changed:

- `app/reasoning/periodic_metric_view.py`

### 5. 현대자동차/삼성전자 연결 당기순이익

Question:

```text
현대자동차 2025년 1분기 연결 당기순이익은?
삼성전자 2025년 1분기 연결 당기순이익은?
```

Before:

- `당기순이익`이라는 단어가 현금흐름표나 주석에도 등장해 손익계산서가 아닌 표가 선택될 수 있었다.
- `분기순이익`, `연결분기순이익`, `당기순손익` 같은 실제 표 행 이름을 충분히 커버하지 못했다.

After:

- 손익계산서 섹션을 우선한다.
- `당기순이익` 질문을 `분기순이익`, `연결분기순이익`, `당기순손익` 계열 행과 매칭한다.

Changed:

- `app/reasoning/query_understanding.py`
- `app/reasoning/periodic_metric_view.py`
- `app/reasoning/periodic_evidence_selector.py`

### 6. 삼성전자 2024년 사업보고서 연결 자산총계

Question:

```text
삼성전자 2024년 사업보고서 연결 자산총계는 얼마야?
```

Before:

- DB에는 `자 산 총 계` 행이 있었지만 검색 상위 결과에 잘 올라오지 않았다.
- 검색 보강 후에도 자본변동표 등 다른 source가 같이 출력되어 `answerable=false` 또는 불필요한 대체 source가 섞였다.
- `제 56 (당) 기` 같은 헤더가 citation 검증에서 오탐 처리될 수 있었다.

After:

- 재무상태표 지표 질문은 후보 universe 안에서 exact metric row와 재무제표 section boost가 높은 chunk를 rescue한다.
- `자 산 총 계`를 `자산총계`로 인식한다.
- `제 56 (당) 기` 헤더를 현재 기수로 인식한다.
- 최종 서버 확인 결과:

```text
자산총계 | 514,531,948
```

Changed:

- `app/retrieval/hybrid.py`
- `app/reasoning/router.py`
- `app/reasoning/periodic_metric_view.py`
- `app/reasoning/periodic_evidence_selector.py`
- `app/generation/answer_generator.py`

### 7. 셀트리온 2024년 사업보고서 주요 제품

Question:

```text
셀트리온 2024년 사업보고서 주요 제품은?
```

Before:

- 서술형 정기공시 근거가 24,000자 이상 출력됐다.
- 답변이 너무 길어 평가/검토가 어려웠다.

After:

- 정기공시 source 표시 텍스트를 항목당 600자로 제한한다.
- general evidence 경로도 상위 3개 evidence, 항목당 600자로 제한한다.
- 70개 추가 테스트에서 5,000자 초과 답변은 0개로 줄었다.

Changed:

- `app/generation/answer_generator.py`
- `app/agent/orchestrator.py`

### 8. 한국항공우주 최근 공급계약 금액과 매출액 대비 비율

Question:

```text
한국항공우주 최근 공급계약 금액과 매출액 대비 비율은?
```

Before:

- 질문에 `매출액`이라는 단어가 있어 정기공시 재무지표 질문으로 잘못 라우팅될 수 있었다.
- 공급계약 이벤트 문서보다 재무제표 검색으로 흐를 위험이 있었다.

After:

- 이벤트 키워드가 있으면 재무지표보다 corporate event 라우팅을 우선한다.
- `supply_contract` event type으로 유지된다.

Changed:

- `app/reasoning/query_understanding.py`

### 9. 하나금융지주 자기주식 취득 신탁계약 해지

Question:

```text
하나금융지주 자기주식 취득 신탁계약 해지 내용 알려줘
```

Before:

- `자기주식취득신탁계약해지결정`이 이벤트 사전에 없어 general evidence 검색이 불안정했다.
- 70개 테스트에서는 retrieval 0 또는 `확인 필요`로 떨어졌다.
- 과거 2023년 공시가 먼저 잡히는 문제가 있었다.

After:

- `treasury_share_trust_termination` event type을 추가했다.
- 해당 보고서명을 hard route로 필터링한다.
- `latest_event` 질문에서는 최신 routed event 문서를 우선한다.
- 서버 확인 결과 최신 2025-07-03 하나금융지주 해지 공시의 본문 결정표가 먼저 나온다.

Example after:

```text
계약금액(원) 해지 전 400,000,000,000
계약금액(원) 해지 후 0
해지목적 자기주식 취득 완료에 따른 중도해지
```

Changed:

- `app/reasoning/query_understanding.py`
- `app/reasoning/router.py`
- `app/retrieval/hybrid.py`

### 10. 상각형 조건부자본증권 발행결정 최근 공시 금액

Question:

```text
상각형 조건부자본증권 발행결정 최근 공시 금액은?
```

Before:

- 이벤트 사전에 없어 routing이 약했다.
- 오래된 2023년 정정공시가 먼저 나오거나 timeout/error가 발생했다.
- 최신 문서가 잡혀도 정정신고 표가 본문 결정표보다 먼저 출력됐다.

After:

- `write_down_contingent_capital_security` event type을 추가했다.
- 최신 routed event 문서 우선 rescue를 추가했다.
- 이벤트별 section boost로 `상각형 조건부자본증권 발행결정` 본문 결정표를 정정신고 표보다 우선한다.
- 서버 확인 결과 최신 2026-03-10 신한지주 공시의 본문 결정표가 먼저 나오고 금액이 포함된다.

Example after:

```text
사채의 권면(전자등록)총액 (원) | 400,000,000,000
```

Changed:

- `app/reasoning/query_understanding.py`
- `app/reasoning/router.py`
- `app/retrieval/hybrid.py`

### 11. 삼성전자 국민연금 최근 보유비율

Question:

```text
삼성전자 국민연금 최근 보유비율은?
```

Before:

- 질문에 `국민연금`이 명시되어도 삼성생명보험 등 다른 보고자의 보유 변동 내역을 출력했다.
- answerable이 낮거나 warning이 있어도 사용자에게는 무관한 값이 보였다.

After:

- `HoldingResolution`에 reporter constraint를 보존한다.
- 질문 reporter와 후보 event reporter가 모두 불일치하면 무관 event fallback을 차단한다.
- 서버 확인 결과 국민연금 근거가 없을 때는 다음처럼 출력한다.

```text
확인 필요
확인되지 않은 정보가 있습니다.
```

Changed:

- `app/reasoning/holding_event_resolver.py`
- `app/reasoning/answer_composer.py`

### 12. 에코프로비엠 국민연금 보유 주식수 변동

Question:

```text
에코프로비엠 국민연금 보유 주식수 변동은?
```

Before:

- 질문은 국민연금인데 에코프로 또는 특수관계자 보유 변동이 출력될 수 있었다.

After:

- reporter mismatch fallback 차단으로 무관 보고자 답변을 하지 않는다.
- corpus 안에서 국민연금 근거가 잡히지 않으면 `확인 필요`로 처리한다.

Changed:

- `app/reasoning/holding_event_resolver.py`
- `app/reasoning/answer_composer.py`

## Remaining cases

### 엘앤에프 2024년 사업보고서 연결 부채총계

Question:

```text
엘앤에프 2024년 사업보고서 연결 부채총계는?
```

Current result:

- `확인 필요`

Reason:

- 서버 DB에서 `엘앤에프` 회사/공시가 조인 조회되지 않았다.
- 현재는 코드 문제보다 corpus/universe coverage 문제로 보는 것이 맞다.

### 2차전지 기업 중 2025년 신규시설투자 금액이 큰 곳

Question:

```text
2차전지 기업 중 2025년 신규시설투자 금액이 큰 곳 알려줘
```

Current result:

- `확인 필요`

Reason:

- DB 기준 2차전지 universe 후보는 삼성SDI, 에코프로비엠, LG에너지솔루션으로 확인됐다.
- 2025년 신규시설투자 공시 목록에는 이 후보들의 해당 공시가 확인되지 않았다.
- 향후 안정적으로 처리하려면 sector 기반 후보군 확장과 금액 비교 resolver가 필요하다.

### Event field extraction

Current limitation:

- 주요사항/거래소 공시는 아직 전용 resolver가 아니라 general evidence 경로를 사용한다.
- 최신 문서와 본문 표 우선순위는 개선했지만, 최종 답변은 여전히 표 근거 중심이다.

Future work:

- `facility_investment`, `supply_contract`, `treasury_share_trust_termination`, `write_down_contingent_capital_security` 등에 대해 event-specific field resolver를 추가한다.
- 예: 계약금액, 자기자본 대비 비율, 해지목적, 사채 총액 같은 필드를 구조화해 한두 문장으로 출력.

## Files changed

- `app/agent/orchestrator.py`
- `app/generation/answer_generator.py`
- `app/reasoning/answer_composer.py`
- `app/reasoning/holding_event_resolver.py`
- `app/reasoning/periodic_evidence_selector.py`
- `app/reasoning/periodic_metric_view.py`
- `app/reasoning/query_understanding.py`
- `app/reasoning/router.py`
- `app/retrieval/hybrid.py`
- `docs/periodic_metric_selection_fix.md`
- `tests/test_answer_composer.py`
- `tests/test_answer_generator.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_orchestrator.py`
- `tests/test_periodic_evidence_selector.py`
- `tests/test_periodic_metric_view.py`
- `tests/test_query_understanding.py`

## Verification

Local full test:

```text
863 passed, 1 skipped
```

Server after dev deployment:

```text
85 passed
healthz: {"status":"ok"}
```

70-question diagnostic after fixes:

```text
total: 70
errors: 1
answerable_false_or_check_needed: 6
retrieval0: 4
long_gt5000: 0
```

Important interpretation:

- Remaining `확인 필요` cases are either corpus coverage issues, reporter evidence not found, or ranking/comparison resolver gaps.
- Previously dangerous cases where the system answered with unrelated evidence were changed to either correct routed evidence or explicit `확인 필요`.
