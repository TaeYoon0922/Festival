# Festival — DART 공시 QA Agent

미래에셋증권 AI Festival 제출용 프로젝트입니다. DART 전자공시 원문을 구조 보존 방식으로
파싱·청킹해 PostgreSQL 16 + pgvector에 적재하고, hybrid retrieval과 결정적(deterministic)
추론으로 근거가 붙은 답변을 만든 뒤, HyperCLOVA X가 그 근거 위에서 한국어 답변을 쓰고
사후 검증을 통과한 것만 HTTP API로 제공합니다.

## 0. 평가용 API End-point

제10회 2026 미래에셋증권 AI Festival · 공시 Agent 제출 엔드포인트입니다.

```
http://101.79.20.171:8000/answer
```

| 항목 | 값 |
|---|---|
| End-point URL | `http://101.79.20.171:8000/answer` |
| 헬스체크 | `http://101.79.20.171:8000/healthz` |
| 표준 포트(80) | `http://101.79.20.171/answer` · `http://101.79.20.171/healthz` — 동일 서비스 |
| 메서드 · 경로 | `GET /answer` (경로 고정) |
| 쿼리 파라미터 | `question_id`, `question` |
| 인증 헤더 | 없음 |
| 응답 Content-Type | `application/json` |
| 응답 필드 | `question_id` · `question` · `retrieved_context` · `think_trace` · `answer` (**모두 string**) |
| 문자 인코딩 | UTF-8 |
| 접근 제한 | 없음 (전체 개방) |
| 가동 기간 | 2026-09-07 ~ 2026-09-20 상시 (systemd `Restart=always`) |

80번 포트는 `nat PREROUTING REDIRECT --dport 80 --to-port 8000`으로 8000번에 연결되어 있고,
그 규칙을 서비스 유닛의 `ExecStartPost`가 매 기동마다 다시 세우므로 재시작·재부팅 후에도
두 포트가 함께 살아납니다. 8000번은 계속 열려 있으므로 제출한 URL 그대로 사용하면 됩니다.

```bash
curl -sG "http://101.79.20.171:8000/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=삼성전자의 2025년 연결기준 매출액은 얼마인가?"
```

요청·응답 스키마 전문은 [`docs/EVALUATION_API.md`](docs/EVALUATION_API.md)에 있습니다.

## 1. Project Overview

| 항목 | 값 |
|---|---|
| 대상 기업 | 국내 주요 산업 대표 상장기업 **70개사** |
| 공시 기간 | **2023-01-01 ~ 2026-03-31** (정기공시는 FY2023 ~ 2026년 1분기) |
| 공시 문서 | **4,204건** / 원본 파일 4,619개 (XML 4,616 + HTML 3) |
| 청크 | 1,363,336개 (text 229,725 / table 1,071,368 / projection 62,243) |
| 저장소 | PostgreSQL 16 + pgvector (HNSW, cosine) |
| 임베딩 | 로컬 BGE-M3 추론(`bge_m3_local`), `BAAI/bge-m3`, 1024차원 |
| 검색 | lexical + vector → RRF → deterministic rerank |
| 생성 | 결정적 answer generator를 기준선으로, HyperCLOVA X(HCX-005, `temperature=0`)가 작성·재작성 |
| API | FastAPI + uvicorn, `GET /healthz`, `GET /answer` |
| 응답 시간 | 중앙 5.2초 · 평균 7.6초 · 최대 22.0초 (제한 300초 대비 7.3%) |

**데이터 제약** — 답변 근거는 **주최 측이 제공한 코퍼스만** 사용합니다. 외부 웹 검색, 외부
데이터셋, 모델의 사전 지식은 근거로 쓰지 않습니다. 언어 모델에게 전달되는 것은 검색된 공시
발췌뿐이고, 모델이 쓴 문장은 **그 발췌에 실제로 있는 숫자·인용·기업명만 남았는지 사후에
자리수 단위로 대조**합니다([7. HyperCLOVA X Safety Strategy](#7-hyperclova-x-safety-strategy)).
대조를 통과하지 못한 응답은 통째로 폐기되므로 코퍼스 밖 정보가 답변에 남을 경로가 없습니다.

핵심 설계 원칙은 **근거가 유창함보다 우선한다**는 것입니다. 결정적 파이프라인이 먼저 인용이
붙은 답변을 완성해 정답의 기준선을 만들고, 언어 모델은 그 위에서만 문장을 씁니다. 검증에
실패하면 언제나 결정적 답변이 그대로 나갑니다.

## 2. Key Features

- **구조 보존 청킹** — 문서 계층과 표 구조를 유지한 채 chunk를 생성하고, 결정적 chunk ID와
  문서·section·table provenance를 함께 보존합니다.
- **Hybrid retrieval** — 메타데이터로 후보 범위를 좁힌 뒤 lexical Top-N과 vector Top-N을
  RRF로 융합하고, 결정적 rerank로 최종 Top-10을 만듭니다.
- **공시 유형별 resolver** — holding event / periodic fact를 각각 구조화된 사실로 복원합니다.
- **인용 강제** — 모든 사실 문장은 chunk와 `source_refs`까지 추적되는 인용을 갖고, 인용
  블록에는 `doc_id`·`chunk_id`와 함께 **공시명과 접수일**이 함께 표시됩니다.
- **HCX 안전 계층 (두 방식)** — 서술형 답변은 모델이 공시 발췌를 읽고 쓴 뒤 사후 대조하고,
  수치가 확정된 답변은 값을 placeholder로 가린 채 문장만 다듬게 합니다. 확정 수치를 문장으로
  이미 갖춘 답변은 아예 모델을 거치지 않습니다.
- **Fail-closed 검증** — 근거에 없는 숫자, 배수가 바뀐 숫자, 없는 인용, 코퍼스 밖 기업명,
  투자 표현, 비교 단정, placeholder 무결성 위반은 전부 폐기 후 결정적 답변으로 fallback입니다.

## 3. Architecture

```text
Question
  |
  v
Query Understanding            company / period / metric / route 해석 → QueryPlan
  |                            + HCX 질의 해석 검증 (모든 질의, plan은 불변)
  v
Planner / Metadata Filter      corp_code · 공시유형 · 기간으로 후보 범위 확정
  |
  v
Hybrid Retrieval               lexical Top-50 + vector Top-50 → RRF(k=60)
                               → deterministic rerank → Top-10
  |
  v
Resolver / Fact / Calculator   TaskRouter → holding_event_resolver
/ Correction / Timeline          | periodic_fact_resolver
                                 | (그 외: resolver 없이 evidence 경로)
  |
  v
Evidence Builder               chunk를 근거 그룹으로 묶고 provenance 유지
  |
  v
Deterministic Answer Generator AnswerComposer → CitationAwareAnswerGenerator
                               (인용이 붙은 확정 답변. 여기까지가 정답의 기준선)
  |
  +-- states_figure = true ---> 그대로 서빙. HCX 미호출
  |                            (확정 수치를 이미 문장으로 갖춘 답변)
  |
  +-- states_figure = false --> AnswerSynthesizer (HCX-005)
                                 공시 발췌를 읽고 답변 작성
                                 → 사후 대조 실패 시
                               AnswerNarrator (HCX-005)
                                 값 마스킹 후 문장만 재작성
                                 → 실패 시
                               AnswerLead (HCX-005)
                                 도입 한 줄만 작성
  |
  v
Validator / Fallback           숫자 근거 대조 · 배수 조작 · 인용 유효성
                               · 미제공 기업명 · 금지 표현 · 비교 단정
                               · placeholder 무결성 · 인용 재부착
                               모든 실패는 결정적 답변으로 복귀
  |
  v
Serialization (API 경계)       retrieved_context · think_trace를 문자열로 렌더링
  |
  v
API Response                   question_id · question · retrieved_context
                               · think_trace · answer  (5개 필드 모두 string)
```

## 4. Corpus & Structural Chunking

원문은 국내 주요 상장기업의 DART 공시입니다. 정기공시(periodic) 1,054건, 주요사항(major)
598건, 거래소(exchange) 1,469건, 지분공시(holding) 1,083건으로 총 4,204건입니다.

청킹은 고정 길이가 아니라 문서 구조를 따릅니다.

- 본문: `section → paragraph → sentence → chunk`
- 표: 병합 셀을 논리 그리드로 복원한 뒤 `header → row group`
- 공시 유형별 독립 전략 (periodic / major·exchange / holding)
- 검색 문맥: 원문 `content`와 별도로 기업·공시·section 경로를 담은 `retrieval_text`
- 추적성: 결정적 chunk ID, 문서/section/table metadata, 앞뒤 chunk 링크

표 행은 문자 길이 때문에 분할하지 않습니다. 작은 이벤트·key-value 표는 통째로 유지하고,
큰 표만 행 묶음으로 나누되 모든 묶음에 동일한 header와 표 context를 반복합니다.

코퍼스와 청킹 결과는 동결되어 있습니다. 파서·청커 재실행이나 임베딩 재생성은 이 저장소의
정상 운영 절차가 아닙니다. 재현 절차는 [14. Development / Data Processing Appendix](#14-development--data-processing-appendix)에 남겨 두었습니다.

## 5. Retrieval Pipeline

`app/retrieval/hybrid.py`의 `HybridQueryExecutor`가 lexical과 vector 검색을 하나의 라우팅된
후보 집합 위에서 실행합니다. 운영 파라미터는 `app/api/settings.py`의 `ApiSettings` 기본값이며,
내부 회귀 세트를 통과한 설정과 동일합니다.

| 파라미터 | 기본값 |
|---|---|
| `top_k` (최종) | 10 |
| `lexical_top_n` | 50 |
| `vector_top_n` | 50 |
| `rrf_k` | 60 |
| `lexical_weight` / `vector_weight` | 1.0 / 1.0 |
| `fusion_weight` / `deterministic_weight` | 0.60 / 0.40 |
| `rerank_mode` | `legacy` |
| `rerank_window_size` | 2 |

벡터 검색이 실패하면 lexical 결과로 계속 진행합니다(`fallback_on_vector_error`).

임베딩은 **제출 서버에서 로컬 BGE-M3 추론(`bge_m3_local`, `BAAI/bge-m3`, 1024차원)** 으로
얻습니다. 적재된 청크 임베딩과 질의 임베딩이 같은 모델·버전이어야 하므로 운영 설정도
이 값으로 고정되어 있습니다([9. Environment Variables](#9-environment-variables)).

CLOVA Studio의 OpenAI 호환 `/v1/openai/embeddings` 경로(`clova_studio`)도 구현되어 있고
`FESTIVAL_EMBEDDING_PROVIDER`로 선택합니다. 이 경로에서는 입력이 길어 `40003`이 반환되면
안전하게 분할해 평균 풀링한 뒤 정규화합니다. 임베딩 모델은 답변을 생성하지 않으므로
LLM 제한(HyperCLOVA X) 대상이 아닙니다.

벡터 커버리지는 코퍼스 전체가 아니라 부분입니다. 후보 청크에 임베딩이 없으면 하이브리드가
아니라 lexical 단독으로 동작하며, 그 사실을 숨기지 않고 `think_trace.warnings`에
`vector_coverage_absent:provider=…,candidates=…,embedded=…,ratio=…`로 남깁니다.

## 6. Agent / Reasoning Pipeline

`AgentOrchestrator`(`app/agent/orchestrator.py`)가 검색 결과를 읽기 전용으로 소비합니다.
retrieval 결과의 순서·점수·후보 payload를 변경하지 않는다는 불변식을 매 요청 검사합니다.

`TaskRouter`가 QueryPlan과 고정된 lexical 신호만으로 경로를 정합니다.

| task_type | resolver |
|---|---|
| `holding_event` | `holding_event_resolver` |
| `periodic_fact` | `periodic_fact_resolver` |
| `corporate_event` | 없음 (evidence 경로) |
| `general_evidence` | 없음 (evidence 경로) |
| `unknown` | 없음, 답변 불가 처리 |

이후 `EvidenceBuilder`가 근거 그룹을 만들고, `AnswerComposer`가 구조화된 answer draft를,
`CitationAwareAnswerGenerator`가 인용이 붙은 최종 결정적 답변을 만듭니다. **이 답변이 정답의
기준선이며, 이후 단계는 이 기준선을 넘어설 수 없습니다.**

## 7. HyperCLOVA X Safety Strategy

이 에이전트가 쓰는 언어 모델은 **HyperCLOVA X(HCX-005) 하나뿐**이며 `temperature=0`으로
호출합니다. 다른 LLM은 답변 생성 경로 어디에도 없습니다.

HCX가 붙는 위치는 **도움이 되는 곳에 붙이고 해가 되는 곳에서 뺀다**는 한 가지 기준으로
정해집니다. 판단 기준은 결정적 생성기가 남기는 `states_figure` 한 값입니다.

| 결정적 답변의 성격 | HCX 관여 |
|---|---|
| 확정 수치를 이미 문장으로 진술함 (`states_figure=true`) | **호출하지 않음.** 그대로 서빙 |
| 수치가 없거나 서술이 필요함 (`states_figure=false`) | 7.2 → 7.3 → 7.4 순으로 시도 |
| 모든 질의 (성격 무관) | 7.1 질의 해석 검증은 항상 수행 |

수치 답변에서 모델을 빼는 이유는 단순합니다. 숫자는 모델이 **조용히** 망칠 수 있는 유일한
대상이고 — 없던 단위, 큰 쪽과 작은 쪽이 뒤바뀐 비교 — 그 답변은 이미 사람이 읽을 수 있는
한 문장이라 모델이 더할 것이 없기 때문입니다.

### 7.1 질의 해석 검증 (모든 질의)

`SemanticQueryFallback.verify`(`app/reasoning/semantic_query_fallback.py`)가 결정적 규칙이
이미 해석에 성공한 질의까지 포함해 **모든 질의를 HCX에 보냅니다.** 다만 모델은 QueryPlan을
바꿀 수 없습니다. 결정적으로 RESOLVED된 해석이 그대로 실행되고, 모델이 무엇으로 읽었는지는
`think_trace.query_validation`에만 기록됩니다.

규칙 기반 해석이 실패한 경우에만 `interpret`이 호출되고, 그때는 결과가 재검증을 거쳐
plan에 반영됩니다.

### 7.2 답변 작성 — 사후 검증 (서술형 질의)

`AnswerSynthesizer`(`app/generation/answer_synthesis.py`)는 일반적인 RAG입니다. 검색된 공시
발췌를 번호를 붙여 모델에 주고, 모델이 그 근거만으로 한국어 답변을 쓰고 문장마다 `[n]`을
답니다. **모델이 실제 수치를 봅니다.** 대신 쓰고 난 뒤에 전부 대조합니다.

| 단계 | 내용 |
|---|---|
| 입력 | 서빙된 chunk 최대 8개, 각 1,800자까지 |
| 요청 | 발췌에 있는 숫자·날짜·이름만 **그대로 복사**할 것, 반올림·환산·재배수 금지, 없는 단위 부여 금지, 비교·증감·순위·손익 단정 금지 |
| 검증 | 아래 표. 하나라도 걸리면 응답 전체 폐기 |
| 실패 | 7.3으로 진행 (`answer_synthesis.status`에 거부 사유 기록) |

| 검사 | status |
|---|---|
| 빈 응답 · 1,500자 초과 | `empty` · `too_long` |
| 서빙되지 않은 chunk를 가리키는 인용 | `unknown_citation` |
| 사실을 쓰면서 인용을 달지 않음 | `no_citation` |
| 근거의 숫자에 배수(억·조 등)를 새로 붙임 | `rescaled_number:<값>` |
| 근거에 없는 숫자 | `unsupported_number:<값>` |
| 투자 권유·전망 표현 | `evaluative_wording:<단어>` |
| 어느 쪽이 크다는 단정 | `comparison_verdict:<단어>` |
| 코퍼스에 없는 기업명 | `unsupplied_company` |

숫자 검사는 의미 판단이 아니라 **자릿수 대조**입니다. 근거 텍스트에 그 숫자열이 없으면
문장이 무엇을 주장하든 근거에서 나온 값이 아닙니다. 배수가 붙은 경우는 배수를 적용한 값만
주장으로 인정하고, 차이·합·비율처럼 근거 값들로부터 산술적으로 유도되는 값은
`_accounted_for`가 확인해 허용합니다.

마크다운 울타리처럼 내용이 아닌 흠은 거부 대신 제거합니다.

스위치는 `FESTIVAL_HCX_SYNTHESIS_ENABLED`(기본 `true`)입니다.

### 7.3 답변 나레이션 — 값 마스킹 (7.2가 거부된 경우)

`AnswerNarrator`(`app/generation/answer_narration.py`)는 반대 방향의 안전 장치입니다.
**모델은 어떤 수치도 보지 못합니다.** 인용 마커·날짜·숫자는 호출 전에 전부 자릿수 없는
토큰으로 치환되고, 호출 후 원본 문자 그대로 복원됩니다.

| 단계 | 내용 |
|---|---|
| 입력 | 결정적 답변 본문. 인용 블록(`doc_id`·`chunk_id`)은 **전송하지 않음** |
| 마스킹 | 인용 마커·날짜·숫자 → `__FESTIVAL_NUMBER_A__` 등 자릿수 없는 토큰 |
| 요청 | 토큰을 그대로·같은 순서로 유지한 채 산문으로 다시 쓸 것 |
| 검증 | 토큰 무결성(누락·중복·순서) · 새 숫자 · 새 인용 마커 · 금지 표현 · 미제공 기업명 · 길이 |
| 복원 | `restore_literals`로 원본 값 복귀 후 인용 블록 재부착 |
| 실패 | 결정적 답변을 그대로 서빙 (`answer_narration.status`에 거부 사유 기록) |

거부 사유는 `empty` · `markdown_fence` · `citation_marker` · placeholder 무결성 사유 ·
`expanded`(길이 초과) · `digit`(새 숫자) · `evaluative_wording` · `unsupplied_company`입니다.

모델이 바꿀 수 있는 것은 **표현뿐**입니다. 수치·날짜·인용·출처는 바꿀 수 없습니다.

스위치는 `FESTIVAL_HCX_NARRATION_ENABLED`(기본 `true`)로 분리돼 있어, 이 계층만 끄고
synthesis·verbalizer·semantic fallback·opening line은 유지할 수 있습니다.

라이브 성공률 측정:

```bash
python scripts/measure_answer_narration.py --base-url http://HOST:PORT --show
python scripts/ask_comparison_set.py --base-url http://HOST:PORT
```

### 7.4 도입 문장 (7.2·7.3이 모두 거부된 경우)

`AnswerLead`가 답변에 없는 **도입 한 줄만** 씁니다. 본문·표·인용 마커는 전혀 건드리지
않으며, 마스킹과 검증은 7.3과 같습니다. 나레이션이 성공한 경우에는 두 번 말하게 되므로
실행하지 않습니다.

### 7.5 Compact claim verbalizer (지분공시 단일 이벤트)

**호출 조건** — 다음을 모두 만족할 때만 호출합니다.

1. `FESTIVAL_HCX_ENABLED`가 참이고 API 키가 설정되어 있음
2. 결정적 답변이 `answerable`
3. compact verified claim을 만들 수 있음
4. **그 claim이 단일 이벤트임**

다중 이벤트 holding claim은 HCX를 호출하지 않고 결정적 답변을 그대로 서빙합니다
(`skipped_multi_event_compact_claim`). 라이브 실험에서 단일 이벤트는 안정적으로 복원된 반면
다중 이벤트는 필드 순서가 섞이거나 기업명이 빠지는 등 재현적으로 실패했기 때문입니다.

**호출 방식**

- 인용은 모델에 전달하지 않습니다. 호출 전에 분리하고, 성공 후 이벤트 소유권에 따라
  결정적으로 다시 붙입니다. 모델이 `[1]`을 만들어낼 경로 자체가 없습니다.
- 검증된 값(DATE / NUMBER / TEXT)은 숫자가 없는 placeholder로 가려 보냅니다. 값이 왕복을
  통과하거나 실패할 뿐, 조용히 바뀔 수 없습니다.
- `temperature=0`, OpenAI 호환 chat completions 계약(`model`·`messages`·`temperature`·`max_tokens`).

**검증** — 다음 중 하나라도 걸리면 결정적 답변으로 돌아갑니다. 부분 복구나 추정 수정은
하지 않습니다.

| 검사 | fallback status |
|---|---|
| placeholder 누락·중복·순서변경·미지 토큰 | `fallback_placeholder_integrity_failed` |
| 모델이 만든 인용 마커 | `fallback_validation_failed` |
| placeholder 밖 숫자 생성 | `fallback_validation_failed` |
| 구조화 TEXT 값 재생성 | `fallback_validation_failed` |
| 단위 중복 (`7.12%` 뒤 `%` 추가 등) | `fallback_redundant_unit_suffix` |
| 투자 권유·전망 표현 | `fallback_validation_failed` |
| 추론·결론 문장 추가 | `fallback_validation_failed` |
| 사실 검증기 실패 | `fallback_validation_failed` |
| 인용 재부착 실패 | `fallback_citation_attachment_failed` |
| 타임아웃 / HTTP 오류 / 응답 파싱 실패 | `fallback_timeout` · `fallback_http_error` · `fallback_invalid_response` |

## 8. API

평가용 End-point와 요청·응답 스키마는 [`docs/EVALUATION_API.md`](docs/EVALUATION_API.md)에
정리했습니다.


FastAPI 애플리케이션은 `app/api/app.py`이며, 진입점은 `python -m app.api`입니다.
DB 연결은 요청 시점에 열립니다. 따라서 **DB가 죽어 있어도 서버는 기동하고**, 요청 단위로
503을 반환합니다.

### GET /healthz

```bash
curl -s http://localhost:8000/healthz
```

```json
{"status": "ok"}
```

파이프라인을 구성하지 않으므로 DB·임베딩·HCX 상태와 무관하게 응답합니다.

### GET /answer

| 쿼리 파라미터 | 필수 | 설명 |
|---|---|---|
| `question_id` | 예 | 호출자가 지정하는 식별자. 공백만 있으면 422 |
| `question` | 예 | 한국어 질문. 공백만 있으면 422 |

```bash
curl -sG http://localhost:8000/answer \
  --data-urlencode 'question_id=HX01' \
  --data-urlencode 'question=하이브 에스엠 변동 후 주식 수'
```

### 응답 top-level schema

응답 최상위는 **정확히 다음 5개 필드**입니다. 추가 필드는 넣지 않습니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `question_id` | string | 요청 값 그대로 반환 |
| `question` | string | 요청 값 그대로 반환 |
| `retrieved_context` | string | 서빙된 Top-K chunk를 구분선으로 이어 붙인 텍스트 |
| `think_trace` | string | 실행 요약을 `key: value` 줄로 이어 붙인 텍스트 (아래 참조) |
| `answer` | string | 최종 답변. 어떤 경우에도 빈 문자열이 아닙니다 |

`answer` 끝에는 근거 공시 블록이 붙습니다. 평가 기준이 "모든 답변에는 근거 공시를 표시할
것"을 요구하므로, 식별자만이 아니라 **사람이 확인할 수 있는 공시명과 접수일**을 함께 씁니다.

```text
[1] doc_id: periodic_20260310002820
    chunk_id: periodic_20260310002820:ch_70f7b399fe050756e113
    공시: 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10
```

**다섯 필드의 값은 모두 string입니다.** 주최측 공지("모든 필드의 값은 문자열(string)타입입니다")를
따르며, `retrieved_context` 안의 구분 방식은 참가팀 재량이라는 안내에 맞춰 아래 형식을 씁니다.
직렬화는 `app/api/serialization.py` 한 곳에서만 일어나고, 파이프라인 내부는 계속 구조화된
형태로 동작합니다.

`retrieved_context` — chunk마다 다음 블록, 블록 사이는 `-` 60자 구분선:

```text
[1] 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10
doc_id: periodic_20260310002820
chunk_id: periodic_20260310002820:ch_70f7b399fe050756e113
chunk_type: table | bm25_score: 12.34 | corp_code: 00126380 | 기준기간: 2025년 12월
섹션: III. 재무에 관한 사항 > 2-2. 연결 손익계산서
내용:
| 열 1 | 제 57 기 |
| 매출액 (주30) | 333,605,938 |
source_refs: [{"row":3,"field":"매출액"}]
provenance: {"table_id":"t7"}
```

`retrieval_text`만 따로 싣지 않습니다. 그 고유 정보인 기업명·공시명·섹션 경로가 이미 블록
머리글이고, chunk 원문은 `내용`에 그대로 들어갑니다. 검색 결과가 없으면
`검색된 공시 근거가 없습니다.`입니다.

chunk 하나의 `내용`은 1,500자에서 끊고 `…(이하 생략)`을 남깁니다. 서빙되는 chunk에는 손익
계산서 한 장이 통째로 들어오기도 해서 열 건이면 4.8만 자에 달했는데, 필드 전체가 임의의
지점에서 잘리면 뒤쪽 chunk가 통째로 사라집니다. chunk마다 경계를 두면 열 건이 모두 남고
잘린 지점이 눈에 보입니다.

`think_trace` — 한 줄에 한 항목. 리스트는 `stages`가 ` > `, `warnings`가 `; `로 이어지고,
하위 컴포넌트 요약은 compact JSON 한 줄입니다:

```text
task_type: periodic_fact
route: periodic_fact_resolver
stages: query_understanding > task_router > answer_generator
retrieval_count: 10
answerable: true
warnings: annual_report_source_preferred; periodic_metric_row_preferred
hcx_status: skipped_no_compact_verified_claim
answer_synthesis: {"status":"skipped_stated_figure","used":false}
query_validation: {"status":"resolved","retrieval_allowed":true,"hcx_status":"success"}
```

#### think_trace는 chain-of-thought가 아닙니다

`think_trace`는 **어떤 컴포넌트가 실행되었고 무엇으로 결론 났는지에 대한 실행 요약**입니다.
모델의 내부 사고 과정이나 중간 추론 텍스트는 담기지 않으며, 컴포넌트 이름·개수·상태값만
포함합니다.

| 필드 | 설명 |
|---|---|
| `task_type` | 라우팅된 태스크 유형 |
| `route` | 사용된 resolver (없으면 `general_evidence`) |
| `stages` | 실행된 컴포넌트 이름 목록 |
| `retrieval_count` | 검색 결과 수 |
| `selected_evidence_count` | 선택된 근거 수 |
| `answerable` | 근거로 답변 가능 여부 |
| `warnings` | 결정적 생성기가 남긴 경고 |
| `hcx_status` | compact claim verbalizer 결과 (`success` / `skipped_*` / `fallback_*` / `disabled` / `not_configured`) |
| `answer_synthesis` | 7.2 답변 작성 결과와 거부 사유 |
| `answer_narration` | 7.3 나레이션 결과와 거부 사유 |
| `answer_lead` | 7.4 도입 문장 결과 |
| `query_validation` | 결정적 해석 결과와 7.1 HCX 검증 결과 |

### 오류 응답

| 상황 | 코드 | 본문 |
|---|---|---|
| `question` / `question_id` 누락·공백 | 422 | FastAPI 검증 오류 |
| DB 접속 불가 | 503 | `{"reason": "database_unavailable", "message": ...}` |
| 임베딩 서비스 불가 | 503 | `{"reason": "embedding_unavailable", "message": ...}` |
| 그 외 실패 | 503 | `{"reason": "internal_error", "message": ...}` |

접속 문자열·자격증명·traceback은 응답에 포함되지 않습니다.

## 9. Environment Variables

값은 이 문서에 기재하지 않습니다. `.env`는 `.gitignore`와 `.dockerignore` 양쪽에서 제외됩니다.

### PostgreSQL

| 변수 | 필수 | 기본값 |
|---|---|---|
| `DATABASE_URL` | 택1 | 없음. 설정 시 아래 `PG*`보다 우선 |
| `PGHOST` | 택1 | 없음 |
| `PGPORT` | 아니오 | 없음 |
| `PGDATABASE` | 택1 | 없음 |
| `PGUSER` | 택1 | 없음 |
| `PGPASSWORD` | 택1 | 없음 |
| `PGSSLMODE` | 아니오 | 없음 (적재 스크립트에서 사용) |

### 임베딩

**운영 설정은 로컬 BGE-M3 추론(`bge_m3_local`)입니다.** 청크 임베딩과 질의 임베딩이
같은 모델·같은 버전이어야 하며, `chunk_embeddings` 테이블의 기본키가
`(chunk_id, embedding_model, embedding_version)`이므로 이 값이 어긋나면 벡터 검색이
아무것도 찾지 못합니다.

| 변수 | 운영값 | 비고 |
|---|---|---|
| `FESTIVAL_EMBEDDING_PROVIDER` | `bge_m3_local` | 기본값은 `hash`(테스트용) |
| `FESTIVAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | 적재된 임베딩과 일치해야 함 |
| `FESTIVAL_EMBEDDING_VERSION` | `6892b95fed65c899a30896eb40d619ae284d0455` | 모델 리비전. 위와 같음 |
| `FESTIVAL_EMBEDDING_DIMENSIONS` | `1024` | DB 인덱스와 일치해야 함 |
| `FESTIVAL_EMBEDDING_MAX_LENGTH` | `2048` | 8192는 8GB GPU에서 VRAM을 초과함 |
| `FESTIVAL_EMBEDDING_DEVICE` | `cpu` | 서빙은 질의 1건당 1회라 CPU로 충분 |
| `FESTIVAL_EMBEDDING_CUDA_OOM_RETRY` | `true` | GPU 적재 경로에서 배치 자동 축소 |
| `FESTIVAL_EMBEDDING_BATCH_SIZE` | `32` | 적재 경로에서만 사용 |
| `FESTIVAL_EMBEDDING_MIN_BATCH_SIZE` | `1` | |

로컬 추론에는 `requirements-embedding.txt`(FlagEmbedding)가 필요합니다.

<details>
<summary>HTTP 임베딩 경로 (CLOVA Studio) — 적재 대안</summary>

OpenAI 호환 엔드포인트로도 적재할 수 있습니다. 다만 계정 rate limit(HTTP 429,
코드 42901)이 있어 배치 16 이상이나 병렬 실행에서 대량 실패합니다. 안전한 조합은
**단일 프로세스 · 배치 8**이며 약 6.6 청크/초입니다.

| 변수 | 기본값 | 비고 |
|---|---|---|
| `FESTIVAL_EMBEDDING_PROVIDER` | `hash` | HTTP 경로는 `clova_studio` |
| `FESTIVAL_EMBEDDING_API_URL` | 없음 | OpenAI 호환 `/v1/openai/embeddings` |
| `FESTIVAL_EMBEDDING_API_KEY` | 없음 | **비밀값** |
| `FESTIVAL_EMBEDDING_API_KEY_HEADER` | `Authorization` | |
| `FESTIVAL_EMBEDDING_API_KEY_PREFIX` | `Bearer` | |
| `FESTIVAL_EMBEDDING_TIMEOUT_SECONDS` | `60` | |
| `FESTIVAL_EMBEDDING_LONG_TEXT_SEGMENT_CHARS` | `1800` | 40003 분할 폴백 |

**HTTP 경로와 로컬 경로는 `embedding_version`이 다르므로 섞이지 않습니다.**
서빙 설정이 적재에 쓴 값과 같아야 합니다.

</details>

### HyperCLOVA X

| 변수 | 기본값 | 비고 |
|---|---|---|
| `FESTIVAL_HCX_ENABLED` | `true` | 끄면 항상 결정적 답변 |
| `FESTIVAL_HCX_API_URL` | `https://clovastudio.stream.ntruss.com/v1/openai/chat/completions` | |
| `FESTIVAL_HCX_API_KEY` | 없음 | **비밀값**. 없으면 `not_configured`로 강등 |
| `FESTIVAL_HCX_API_KEY_HEADER` | `Authorization` | |
| `FESTIVAL_HCX_API_KEY_PREFIX` | `Bearer` | |
| `FESTIVAL_HCX_MODEL` | `HCX-005` | |
| `FESTIVAL_HCX_TIMEOUT_SECONDS` | `15.0` | |
| `FESTIVAL_HCX_MAX_TOKENS` | `1024` | |
| `FESTIVAL_HCX_TEMPERATURE` | `0.0` | |
| `FESTIVAL_HCX_SYNTHESIS_ENABLED` | `true` | 7.2 답변 작성만 개별로 끄는 스위치 |
| `FESTIVAL_HCX_NARRATION_ENABLED` | `true` | 7.3 나레이션만 개별로 끄는 스위치 |
| `FESTIVAL_ANSWER_DIRECT_ENABLED` | `true` | 결정적 직답 문장 생성 스위치 |

### API / 실행

| 변수 | 기본값 |
|---|---|
| `FESTIVAL_API_HOST` | `0.0.0.0` |
| `FESTIVAL_API_PORT` | `8000` |
| `FESTIVAL_API_WORKERS` | `1` |
| `FESTIVAL_API_LOG_LEVEL` | `info` |
| `FESTIVAL_API_TOP_K` | `10` |
| `FESTIVAL_API_LEXICAL_TOP_N` | `50` |
| `FESTIVAL_API_VECTOR_TOP_N` | `50` |
| `FESTIVAL_API_RRF_K` | `60` |
| `FESTIVAL_API_LEXICAL_WEIGHT` | `1.0` |
| `FESTIVAL_API_VECTOR_WEIGHT` | `1.0` |
| `FESTIVAL_API_FUSION_WEIGHT` | `0.60` |
| `FESTIVAL_API_DETERMINISTIC_WEIGHT` | `0.40` |
| `FESTIVAL_API_RERANK_MODE` | `legacy` |
| `FESTIVAL_API_RERANK_WINDOW_SIZE` | `2` |
| `FESTIVAL_API_DIAGNOSTIC_TOP_N` | 없음 |
| `FESTIVAL_API_DB_CONNECT_TIMEOUT_SECONDS` | `10` |

### 코퍼스 경로

| 변수 | 기본값 |
|---|---|
| `DISCLOSURE_DATA_DIR` | `data/corpus` (이미지 내부에서는 `/app/data/corpus`) |

## 10. Docker Deployment

이미지는 API 서버를 실행합니다. 검색 데이터는 이미지가 아니라 PostgreSQL에 있으므로 원문
raw 디렉터리를 마운트할 필요가 없습니다. 이미지에는 `data/corpus`의 메타데이터
(`manifest.jsonl`, `universe.csv` 등)만 포함되며 `data/corpus/raw`는 제외됩니다.

```bash
docker build -t festival-api .

docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  festival-api
```

```bash
curl -s http://localhost:8000/healthz

curl -sG http://localhost:8000/answer \
  --data-urlencode 'question_id=HX01' \
  --data-urlencode 'question=하이브 에스엠 변동 후 주식 수' | jq
```

`Dockerfile`의 `CMD`는 `python -m app.api`이며 8000번 포트를 노출합니다. 코퍼스 연결 점검용
진입점은 `docker run --rm festival-api python -m app`으로 여전히 사용할 수 있습니다.

### 빌드 컨텍스트

`.dockerignore`가 다음을 제외합니다: `.git`, `.gitignore`, `.env`, `.env.*`, `.venv`,
`venv`, `__pycache__`, `*.py[cod]`, `*.zip`, `data/corpus/raw`, `data/processed`,
`data/exports`, `data/db_export`, `reports`, `__MACOSX/`, `.DS_Store`.

빌드 산출물과 평가 산출물, 그리고 모든 `.env` 계열 파일이 컨텍스트에 들어가지 않습니다.
로컬에서 COPY export를 돌려 생성한 `data/db_export`도 제외되므로, 제출 이미지를 빌드하기 전에
따로 비울 필요가 없습니다. 이미지에는 `app/`, `scripts/`, `data/corpus`의 메타데이터만
복사됩니다.

### Docker smoke 검증 (내부 기록)

| 항목 | 결과 |
|---|---|
| 이미지 빌드 | 성공 |
| `/answer` production 경로 | 성공 |
| `bge-m3` 임베딩 1024차원 | 성공 |
| PostgreSQL retrieval | 성공 |
| HCX 검증 / fallback | 성공 |
| 빌드 컨텍스트 | 14.16 GB → 6.26 MB |

## 11. PostgreSQL / pgvector

대상은 **PostgreSQL 16 + pgvector**입니다. SQL은 단계별로 분리되어 있습니다.

| 파일 | 역할 |
|---|---|
| `db/001_schema.sql` | 기본 DDL |
| `db/002_indexes.sql` | base load 이후 인덱스 |
| `db/003_validation.sql` | DB 측 무결성 검증 |
| `db/004_vector_search.sql` | `vector` 확장, 임베딩 테이블, HNSW cosine 인덱스 |
| `db/005_table_chunk_provenance_backfill.sql` | table chunk provenance 백필 |

임베딩 테이블은 차원을 제약으로 강제하고(`vector_dims(embedding) = embedding_dimensions`),
1024차원 코사인 HNSW 인덱스를 만듭니다. 접속 정보는 코드가 아니라 `DATABASE_URL` 또는 표준
`PG*` 환경변수로 전달합니다.

적재 절차는 [14. Development / Data Processing Appendix](#14-development--data-processing-appendix)를 참고하십시오.

## 12. Evaluation

아래는 **내부 회귀 세트(internal regression set)** 결과입니다. 공식 대회 평가 결과가 아니며,
개발 중 회귀를 잡기 위해 동결해 둔 자체 질의 60건(Gold40 + Holding20)에 대한 측정치입니다.

### 12.1 Gold60 (2026-08-21 동결 산출물)

| 지표 | 값 |
|---|---|
| `question_count` | 60 |
| Recall@1 | 0.483333 |
| Recall@5 | 0.783333 |
| Recall@10 | 0.900000 |
| `answerable_rate` | 1.000000 |
| `gold_doc_citation_rate` | 0.916667 |
| `gold_chunk_citation_rate` | 0.883333 |
| `all_evidence_terms_rate` | 0.950000 |
| `end_to_end_success_rate` | 0.900000 |
| 성공 | 54 / 60 |

남은 6건은 전부 **retrieval miss**입니다. 즉 reasoning·인용·답변 가능성 판정에서의 실패는
0건이며, 정답 근거가 Top-10 안에 들어오지 못한 경우입니다.

산출물은 `reports/evaluation/gold60/2026-08-21-agent-90pct/`에 원본 그대로 보존되어 있고,
`run_manifest.json`과 `SHA256SUMS`로 실행 파라미터·코드 커밋·체크섬을 함께 기록합니다.

```bash
python scripts/preserve_evaluation_artifact.py --verify
```

재평가(PostgreSQL·임베딩 자격증명 필요):

```bash
python scripts/evaluate_postgres_agent_gold60.py \
  --output-dir data/processed/postgres_agent_gold60
```

현재 코드로 같은 세트를 다시 돌리면 `end_to_end_success_rate`는 **0.816667**입니다. 위 표는
동결 시점의 산출물이고 이 값은 그 이후 검색·해석 변경이 누적된 현재 상태입니다.
`FESTIVAL_ANSWER_DIRECT_ENABLED`를 켠 경우와 끈 경우가 동일하게 0.816667로 나오므로,
차이는 답변 생성 계층이 아니라 그 앞단에서 발생합니다.

### 12.2 응답 시간 — 두 기업 비교 30문항

서빙 중인 엔드포인트에 실제로 요청해 측정한 값입니다(`scripts/ask_comparison_set.py`).

| 항목 | 값 |
|---|---|
| 최소 · 중앙 · 평균 | 0.0초 · 5.2초 · 7.6초 |
| p90 · 최대 | 13.8초 · 22.0초 |
| 10초 이하 | 19 / 30 |
| 20초 이하 | 29 / 30 |
| 주최측 제한(300초) 대비 최대치 | 7.3% |

가장 느린 22.0초도 제한의 1/13입니다. 0.0초는 코퍼스 범위 밖으로 판정해 검색 이전에 끝난
경우이고, 가장 느린 구간은 HCX 호출이 두 번 일어난 경우입니다. 언어 모델이 관여하지 않는
수치 답변은 대체로 2~4초에 끝났습니다.

## 13. Safety & Fallback

- **답변은 항상 존재합니다.** HCX가 꺼져 있든, 건너뛰든, 거부되든, 결정적 답변이 서빙됩니다.
  빈 문자열이 반환되는 경로는 없습니다.
- **근거 없는 답변은 만들지 않습니다.** 근거가 부족하면 `answerable=false`로 표시하고
  확인이 필요하다는 결정적 문구를 냅니다. 이 경우 HCX는 호출조차 하지 않습니다.
- **부분 복구를 하지 않습니다.** 검증에 실패한 모델 출력은 고쳐 쓰지 않고 폐기합니다.
  잘못 추정한 값 하나가 결정적 답변 전체보다 위험하기 때문입니다.
- **숫자는 판단이 아니라 대조로 지킵니다.** 모델이 쓴 모든 숫자는 서빙된 공시 발췌 안에서
  자릿수 그대로 찾을 수 있어야 하고, 배수가 붙었다면 배수를 적용한 값이 근거에 있거나
  근거 값들로부터 산술적으로 유도되어야 합니다. 그렇지 않으면 응답 전체가 폐기됩니다.
- **비교 단정은 모델에게 맡기지 않습니다.** 어느 쪽이 크다는 문장이 모델 출력에 나타나면
  거부합니다. 비교 결론은 두 값의 단위가 공시 원문에서 확인될 때에만, 결정적 계층이 붙입니다.
- **비밀값을 노출하지 않습니다.** 오류 응답에 DSN·자격증명·traceback이 포함되지 않으며,
  API 키는 설정 객체의 `repr`에서도 제외됩니다.
- **투자 자문을 생성하지 않습니다.** 매수·매도·목표주가 등의 표현이 모델 출력에 새로
  등장하면 그 응답은 거부됩니다.

## 14. Development / Data Processing Appendix

아래는 코퍼스 구축 단계의 기록입니다. **제출용 API 운영에는 필요하지 않습니다.** 검색 데이터는
이미 PostgreSQL에 적재되어 있으며, 파서·청커 재실행과 임베딩 재생성은 정상 운영 절차가
아닙니다.

### 저장소 구조

```text
.
├── app/
│   ├── api/          # FastAPI 계층 (app, pipeline, schemas, settings, serialization)
│   ├── parsing/      # DART XML 파싱 · 구조 청킹 · 검증
│   ├── retrieval/    # 임베딩 · lexical/vector 검색 · hybrid 융합
│   ├── reasoning/    # query understanding · resolver · evidence · composer
│   ├── agent/        # task router · orchestrator · 평가
│   ├── generation/   # 결정적 생성 · 검증 · HCX verbalizer
│   └── exporting/    # PostgreSQL COPY export
├── db/               # 스키마 · 인덱스 · 검증 · 벡터 SQL
├── data/corpus/      # 메타데이터와 로컬 raw 데이터 연결
├── reports/          # 보존된 평가 산출물
├── scripts/          # 파이프라인 · 평가 · 진단 스크립트
└── tests/
```

### 데이터 연결

대용량 원문은 Git에 올리지 않습니다. 기본 경로는 `data/corpus`이며 `data/corpus/raw`가
로컬에 압축 해제된 공시 원문을 가리킵니다.

```bash
export DISCLOSURE_DATA_DIR=/absolute/path/to/corpus
python -m app          # 코퍼스 연결 점검
python -m unittest discover
```

### 파싱 파일럿 (20건)

Vector DB나 외부 LLM 없이 `raw XML → Section / Table → Chunk`만 수행합니다. 정정공시를 제외한
정기·거래소·주요사항·지분공시를 각각 5건씩 선정하며, 정기공시 표본에는 삼성전자 사업보고서가
포함됩니다.

```bash
python scripts/parse_sample.py
python scripts/validate_sample.py
```

결과는 `data/processed/sample_20`에 생성됩니다 (`selection.json`, `summary.json`,
`documents/*.json`). 재생성 가능하므로 Git에서 제외합니다.

고정 길이 청커와 구조 청커를 동일한 20건에서 비교하려면:

```bash
python scripts/compare_chunking_pilot.py
```

결과는 `data/processed/chunking_pilot_20`에 생성됩니다.

### 전체 재처리 전 freeze gate

저장된 20건 pilot에 대한 길이·제외 표 audit, Legacy/Structural BM25 비교, 유형별 정정공시
3건씩의 추가 검증만 수행합니다. 전체 4,204개 문서를 재처리하지 않습니다.

```bash
python scripts/run_final_chunking_validation.py
```

결과는 `data/processed/chunking_final_validation`에 생성됩니다.

### 전체 코퍼스 파싱

파일럿 검증 후 전체 4,204개 문서의 XML 4,616개와 대체 HTML 3개를 처리합니다. 정정공시와
정기공시 첨부 XML도 포함합니다.

```bash
python scripts/parse_all.py --workers 4
python scripts/validate_all.py
```

산출물: `source_plan.jsonl`, `index.jsonl`, `summary.json`,
`documents/<유형>/<문서 ID>/*.json.gz`.

### PostgreSQL 16 COPY export 및 적재

동결된 Structural v2.1 결과를 재청킹하지 않고 COPY 파일로 변환합니다.

```powershell
.\.venv\Scripts\python.exe scripts\export_db_release.py
.\.venv\Scripts\python.exe scripts\validate_db_export.py
.\.venv\Scripts\python.exe scripts\load_postgres.py --dry-run
```

release export 위치는 `data/db_export`입니다. 대용량 CSV는 Git에서 제외되며
`manifest.json`, `export_report.json`, `validation_report.json`에 count와 무결성 결과가
남습니다.

실제 적재에는 `psql`이 필요합니다.

```powershell
$env:DATABASE_URL='postgresql://user:password@host:5432/database'
.\.venv\Scripts\python.exe scripts\load_postgres.py --apply-schema --validate
.\.venv\Scripts\python.exe scripts\load_postgres.py --apply-indexes --validate
```

각 테이블은 임시 staging table로 COPY한 뒤 `ON CONFLICT DO NOTHING`으로 반영되며, 완료 상태를
`.import_state.json`에 기록해 중간 실패 후 재시작할 수 있습니다.

### 임베딩 (선택)

로컬 BGE-M3 추론이 필요한 경우에만 설치합니다. HTTP 임베딩 경로에는 필요하지 않습니다.

```bash
pip install -r requirements-embedding.txt
```

CLOVA Studio 설정과 40003 분할 폴백 동작은 `docs/embedding_pilot.md`에 정리되어 있습니다.

### 테스트

테스트 의존성은 `requirements-dev.txt`에 분리되어 있습니다. `Dockerfile`은 `requirements.txt`만
복사하므로 제출 이미지에는 포함되지 않습니다.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

전체 스위트는 PostgreSQL·임베딩·HCX 자격증명 없이 통과합니다.
