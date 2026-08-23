# Periodic metric selection fix

## Background

정기공시 재무지표 질문에서 답변에 필요한 표 행만 선택하지 못하는 문제가 있었다.

- 연결 매출액 질문에서 수익 주석 또는 전체 손익계산서가 과하게 출력됨
- 삼성전자 `매출액 (주26)`처럼 주석 번호가 붙은 행을 정확한 `매출액` 행으로 인식하지 못함
- 2025년 1분기와 2024년 1분기 비교 질문에서 2024년 보고서를 우선 선택하거나 `확인 필요`로 떨어짐
- `당기순이익` 질문에서 손익계산서가 아니라 현금흐름표/주석의 `당기순이익(손실)` 표를 선택함
- 주요사항/거래소/일반근거 답변이 Top-K 표 전체를 덤프해 지나치게 길어짐
- `자 산 총 계`처럼 띄어쓰기된 재무상태표 행이 실제 DB에는 있어도 검색 상위권에 들지 못해 `확인 필요`로 떨어짐

## Changed Files

- `app/reasoning/periodic_metric_view.py`
  - 재무지표 표에서 질문 지표에 해당하는 행만 남기는 공통 projection 로직을 보강했다.
  - `매출액` 질문에서 NAVER처럼 손익계산서 row가 `영업수익`으로 표기되는 경우도 같은 매출 지표로 처리한다.
  - 단일 기간 질문에서는 선택된 보고서의 현재 기수/명시 연도에 해당하는 열만 남긴다.
  - 분기 질문은 기본적으로 `3개월` 열을 남기고, 질문에 `누적` 또는 `누계`가 있을 때만 누적 열을 남긴다.
  - 전년동기대비, 기간 비교, 추이 질문에서는 비교 열을 유지한다.
  - `(주26)` 같은 주석 번호 suffix를 제거하고 행 이름을 비교한다.
  - `당기순이익`을 `분기순이익`, `연결분기순이익`, `당기순손익` 등 손익계산서 실제 행 이름과 매칭한다.
  - `자 산 총 계`, `부 채 총 계`, `자 본 총 계`처럼 공백이 포함된 재무상태표 행을 각각 `자산총계`, `부채총계`, `자본총계`와 매칭한다.
  - `제 56 (당) 기`처럼 `(당)`/`(전)` 표시가 들어간 헤더도 현재 회계 기수로 인식한다.

- `app/reasoning/router.py`
  - section boost 비교 시 괄호/문장부호를 제거한 비교도 함께 수행한다.
  - 예: chunk section `"(첨부)연 결 재 무 제 표"`가 boost key `"첨부연결재무제표"`와 매칭된다.

- `app/retrieval/hybrid.py`
  - `자산총계`, `부채총계`, `자본총계` 질문에서는 lexical/vector 상위 결과에 실제 재무상태표 chunk가 빠지는 경우를 보강한다.
  - 이미 회사/연도/정기공시로 좁혀진 candidate universe 안에서 exact metric term과 재무제표 section boost가 모두 높은 chunk만 rescue한다.
  - 일반 질문에는 적용하지 않고, 재무상태표 3개 지표에만 적용한다.

- `app/reasoning/periodic_evidence_selector.py`
  - 연결/별도 기준과 손익계산서 섹션을 우선하는 evidence selection을 보강했다.
  - `매출액과`, `매출액을`처럼 조사가 붙은 지표 표현을 같은 지표로 정규화한다.
  - 기간 비교 질문에서는 단일 명시 기간 필터를 강제하지 않고, 비교 연도 중 최신 보고서를 우선한다.
  - exact metric row가 확인된 source는 eligibility 단계에서 탈락하지 않도록 했다.
  - 재무지표 단일 행을 선택한 경우 불필요한 기간 모호성 경고를 제거한다.
  - `자산총계`, `부채총계`, `자본총계`도 단일 재무제표 행 선택 대상으로 포함한다.
  - `자산 총계`처럼 공백 때문에 남은 focus term이 요청 metric과 같으면 불필요한 추가 조건으로 보지 않는다.

- `app/reasoning/query_understanding.py`
  - `당기순이익` 검색어에 `분기순이익`, `연결분기순이익`, `당기순손익`, `분기순손익`을 추가해 실제 손익계산서 chunk recall을 높였다.
  - `자산총계`, `부채총계`, `자본총계` 검색어에 `자산 총 계`처럼 공시 표에 띄어쓰기된 계정명을 추가해 재무상태표 recall을 높였다.
  - `누적`, `누계`, `3개월` 같은 기간 열 선택 단어를 lexical 검색어에서 제거해 주석의 `누적비지배지분` 같은 무관 chunk recall을 줄였다.
  - `공급계약 금액과 매출액 대비 비율`처럼 이벤트 질문에 재무지표 단어가 포함된 경우에도 기업 이벤트 라우팅을 우선하도록 했다.

- `app/reasoning/answer_composer.py`
  - 답변 렌더링 단계에서 사용할 수 있도록 정기공시 request에 `comparison` 정보를 포함했다.
  - 보유공시에서 질문에 보고자/reporter가 명시됐는데 후보 event의 reporter가 모두 불일치하면 무관한 event를 fallback으로 출력하지 않는다.

- `app/reasoning/holding_event_resolver.py`
  - composer가 reporter 제약 여부를 판단할 수 있도록 `HoldingResolution`에 `reporter_constraint`를 보존한다.

- `app/generation/answer_generator.py`
  - 정기공시 답변 표를 렌더링할 때 metric, period, comparison 정보를 함께 넘겨 행/열 projection을 적용한다.
  - projection된 표 header/separator 줄은 독립적인 fact claim이 아니므로 citation scope 검증에서 제거하지 않도록 했다.
  - 재무상태표 header에서 `제 56 (당) 기`, 빈 계정명 칸 등이 등장해도 표 구조로 인식해 citation 검증에서 오탐 제거되지 않도록 했다.
  - 정기공시 서술형 근거가 과도하게 길어지는 경우 표시 텍스트를 항목당 600자로 제한해 제품/사업 설명 질문에서 원문 전체를 덤프하지 않도록 했다.

- `app/agent/orchestrator.py`
  - resolver가 없는 general evidence 경로의 답변 근거를 상위 3개로 제한했다.
  - 근거 본문은 항목당 600자로 제한하고 `[truncated]` 표시를 남긴다.

- Tests
  - `tests/test_periodic_metric_view.py`
  - `tests/test_periodic_evidence_selector.py`
- `tests/test_query_understanding.py`
- `tests/test_orchestrator.py`
- `tests/test_hybrid_retrieval.py`
- `tests/test_answer_generator.py`
- `tests/test_answer_composer.py`

## Verified Questions

서버 API에서 다음 케이스를 확인했다.

- `현대자동차 2025년 1분기 연결 매출액`
  - 연결 손익계산서 `매출액` 행만 선택하고, 단일 기간 질문에서는 2025년 1분기 `3개월` 열만 표시
- `현대자동차 2025년 1분기 별도 매출액`
  - 별도 손익계산서 `매출액` 행만 선택
- `현대자동차 2025년 1분기 매출액과 2024년 1분기 매출액 비교`
  - 2025년 분기보고서의 비교 열을 사용
- `삼성전자 2025년 1분기 연결 매출액`
  - 부문 주석이 아니라 연결 손익계산서 `매출액 (주26)` 행 선택
- `현대자동차 2025년 1분기 연결 당기순이익`
  - 현금흐름표가 아니라 연결 손익계산서 `연결분기순이익` 행 선택
- `삼성전자 2025년 1분기 연결 당기순이익`
  - 연결 손익계산서 `분기순이익` 행 선택
- `NAVER 2025년 1분기 연결 매출액`
  - 손익계산서의 `영업수익 (주5)` 행을 매출액으로 선택
- `삼성전자 2024년 사업보고서 연결 자산총계`
  - `(첨부)연결재무제표`의 `자 산 총 계` 행을 선택하도록 보강 중

## Test Results

Local:

```text
93 passed, 1 warning, 16 subtests passed
94 passed, 1 warning, 16 subtests passed
32 passed, 1 warning
78 passed, 1 warning, 16 subtests passed
79 passed, 1 warning, 16 subtests passed
80 passed, 1 warning, 16 subtests passed
851 passed, 1 skipped, 1 warning, 500 subtests passed
81 passed, 1 warning, 16 subtests passed
852 passed, 1 skipped, 1 warning, 500 subtests passed
55 passed, 1 warning, 16 subtests passed
854 passed, 1 skipped, 1 warning, 500 subtests passed
858 passed, 1 skipped
859 passed, 1 skipped
860 passed, 1 skipped
860 passed, 1 skipped
```

Additional diagnostic run:

- `local-test/questions_extra70.jsonl` 중 49개를 우선 실행했다.
- 발견한 즉시 수정 대상:
  - `X027 셀트리온 2024년 사업보고서 주요 제품은?`: 서술형 정기공시 근거가 24,000자 이상 출력됨.
  - `X035 한국항공우주 최근 공급계약 금액과 매출액 대비 비율은?`: `매출액` 단어 때문에 공급계약 질문이 정기공시 재무지표로 잘못 라우팅됨.

Server:

```text
93 passed, 16 subtests passed
94 passed, 16 subtests passed
```

Diagnostic 36 after fixes:

```text
total: 36
answerable: 35
unanswerable: 1
errors: 0
retrieval_count=0: D036
answers over 5000 chars: 0
```

`D036` is `삼성전자 2028년 연결 매출액`, which is outside the provided corpus period. The expected behavior is `확인 필요`.

## Remaining Risks

- General evidence answers are now bounded, but they still summarize by showing selected evidence text rather than extracting every requested field into a concise sentence.
- Ranking questions such as "가장 큰 공시" may still need a dedicated comparison/calculation resolver for robust numeric sorting.
- Major/exchange disclosure types still use the general evidence path. A future improvement should add event-specific resolvers for convertible bonds, treasury stock, facility investment, supply contracts, merger, and spin-off fields.
