# Periodic metric selection fix

## Background

정기공시 재무지표 질문에서 답변에 필요한 표 행만 선택하지 못하는 문제가 있었다.

- 연결 매출액 질문에서 수익 주석 또는 전체 손익계산서가 과하게 출력됨
- 삼성전자 `매출액 (주26)`처럼 주석 번호가 붙은 행을 정확한 `매출액` 행으로 인식하지 못함
- 2025년 1분기와 2024년 1분기 비교 질문에서 2024년 보고서를 우선 선택하거나 `확인 필요`로 떨어짐
- `당기순이익` 질문에서 손익계산서가 아니라 현금흐름표/주석의 `당기순이익(손실)` 표를 선택함
- 주요사항/거래소/일반근거 답변이 Top-K 표 전체를 덤프해 지나치게 길어짐

## Changed Files

- `app/reasoning/periodic_metric_view.py`
  - 재무지표 표에서 질문 지표에 해당하는 행만 남기는 공통 projection 로직을 보강했다.
  - 단일 기간 질문에서는 선택된 보고서의 현재 기수/명시 연도에 해당하는 열만 남긴다.
  - 분기 질문은 기본적으로 `3개월` 열을 남기고, 질문에 `누적` 또는 `누계`가 있을 때만 누적 열을 남긴다.
  - 전년동기대비, 기간 비교, 추이 질문에서는 비교 열을 유지한다.
  - `(주26)` 같은 주석 번호 suffix를 제거하고 행 이름을 비교한다.
  - `당기순이익`을 `분기순이익`, `연결분기순이익`, `당기순손익` 등 손익계산서 실제 행 이름과 매칭한다.

- `app/reasoning/periodic_evidence_selector.py`
  - 연결/별도 기준과 손익계산서 섹션을 우선하는 evidence selection을 보강했다.
  - `매출액과`, `매출액을`처럼 조사가 붙은 지표 표현을 같은 지표로 정규화한다.
  - 기간 비교 질문에서는 단일 명시 기간 필터를 강제하지 않고, 비교 연도 중 최신 보고서를 우선한다.
  - exact metric row가 확인된 source는 eligibility 단계에서 탈락하지 않도록 했다.
  - 재무지표 단일 행을 선택한 경우 불필요한 기간 모호성 경고를 제거한다.

- `app/reasoning/query_understanding.py`
  - `당기순이익` 검색어에 `분기순이익`, `연결분기순이익`, `당기순손익`, `분기순손익`을 추가해 실제 손익계산서 chunk recall을 높였다.
  - `누적`, `누계`, `3개월` 같은 기간 열 선택 단어를 lexical 검색어에서 제거해 주석의 `누적비지배지분` 같은 무관 chunk recall을 줄였다.

- `app/reasoning/answer_composer.py`
  - 답변 렌더링 단계에서 사용할 수 있도록 정기공시 request에 `comparison` 정보를 포함했다.

- `app/generation/answer_generator.py`
  - 정기공시 답변 표를 렌더링할 때 metric, period, comparison 정보를 함께 넘겨 행/열 projection을 적용한다.
  - projection된 표 header/separator 줄은 독립적인 fact claim이 아니므로 citation scope 검증에서 제거하지 않도록 했다.

- `app/agent/orchestrator.py`
  - resolver가 없는 general evidence 경로의 답변 근거를 상위 3개로 제한했다.
  - 근거 본문은 항목당 1,000자로 제한하고 `[truncated]` 표시를 남긴다.

- Tests
  - `tests/test_periodic_metric_view.py`
  - `tests/test_periodic_evidence_selector.py`
  - `tests/test_query_understanding.py`
  - `tests/test_orchestrator.py`

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
```

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
