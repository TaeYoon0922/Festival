# Festival — DART 공시 QA Agent

미래에셋증권 AI Festival 제출용 프로젝트입니다. DART 전자공시 원문을 구조 보존 방식으로
파싱·청킹해 PostgreSQL 16 + pgvector에 적재하고, hybrid retrieval과 결정적(deterministic)
추론으로 근거가 붙은 답변을 생성한 뒤, HyperCLOVA X로 그 답변을 안전하게 자연어로 다듬어
HTTP API로 제공합니다.

## 1. Project Overview

| 항목 | 값 |
|---|---|
| 대상 기업 | 국내 주요 산업 대표 상장기업 **70개사** |
| 공시 기간 | **2023-01-01 ~ 2026-03-31** (정기공시는 FY2023 ~ 2026년 1분기) |
| 공시 문서 | **4,204건** / 원본 파일 4,619개 (XML 4,616 + HTML 3) |
| 청크 | 1,363,336개 (text 229,725 / table 1,071,368 / projection 62,243) |
| 저장소 | PostgreSQL 16 + pgvector (HNSW, cosine) |
| 임베딩 | CLOVA Studio OpenAI 호환 엔드포인트, `bge-m3`, 1024차원 |
| 검색 | lexical + vector → RRF → deterministic rerank |
| 생성 | 결정적 answer generator + HyperCLOVA X(HCX-005) verbalizer |
| API | FastAPI + uvicorn, `GET /healthz`, `GET /answer` |

**데이터 제약** — 답변 근거는 **주최 측이 제공한 코퍼스만** 사용합니다. 외부 웹 검색, 외부
데이터셋, 모델의 사전 지식은 근거로 쓰지 않습니다. 언어 모델에도 검색된 원문을 전달하지 않고
이미 검증된 사실만 전달하므로([7. HyperCLOVA X Safety Strategy](#7-hyperclova-x-safety-strategy))
코퍼스 밖 정보가 답변에 유입될 경로가 구조적으로 없습니다.

핵심 설계 원칙은 **근거가 유창함보다 우선한다**는 것입니다. 사실·수치·인용은 전부 결정적
파이프라인이 만들고, 언어 모델은 이미 검증된 문장을 다듬는 역할만 맡습니다. 검증에 실패하면
언제나 결정적 답변이 그대로 나갑니다.

## 2. Key Features

- **구조 보존 청킹** — 문서 계층과 표 구조를 유지한 채 chunk를 생성하고, 결정적 chunk ID와
  문서·section·table provenance를 함께 보존합니다.
- **Hybrid retrieval** — 메타데이터로 후보 범위를 좁힌 뒤 lexical Top-N과 vector Top-N을
  RRF로 융합하고, 결정적 rerank로 최종 Top-10을 만듭니다.
- **공시 유형별 resolver** — holding event / periodic fact를 각각 구조화된 사실로 복원합니다.
- **인용 강제** — 모든 사실 문장은 chunk와 `source_refs`까지 추적되는 인용을 갖습니다.
- **HCX 안전 계층** — 인용을 분리하고 검증된 값을 placeholder로 가린 채 호출하며, 통과하지
  못한 응답은 폐기하고 결정적 답변으로 되돌립니다.
- **Fail-closed 검증** — placeholder 무결성, 숫자 생성, 단위 중복, 구조화 텍스트 누출,
  투자 표현, 추론 문장, 인용 재부착까지 모두 실패 시 fallback입니다.

## 3. Architecture

```text
Question
  |
  v
Query Understanding            company / period / metric / route 해석 → QueryPlan
  |
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
  v
Safe HyperCLOVA X Verbalizer   단일 이벤트 compact claim만 대상
                               인용 분리 → 값 placeholder 마스킹 → HCX-005 호출
  |
  v
Validator / Fallback           placeholder 무결성 · 숫자 · 단위 · 텍스트 누출
                               · 금지 표현 · 추론 문장 · 인용 재부착 · 최종 검증
                               실패 시 결정적 답변으로 복귀
  |
  v
API Response                   question_id · question · retrieved_context
                               · think_trace · answer
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

벡터 검색이 실패하면 lexical 결과로 계속 진행합니다(`fallback_on_vector_error`). 임베딩은
CLOVA Studio의 OpenAI 호환 `/v1/openai/embeddings` 엔드포인트로 `bge-m3` 1024차원을 얻으며,
입력이 길어 `40003`이 반환되면 안전하게 분할해 평균 풀링한 뒤 정규화합니다.

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

HCX는 답을 만들지 않습니다. 이미 검증된 사실을 자연스러운 한국어로 옮기는 변환기입니다.

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
| `retrieved_context` | array | 서빙된 Top-K chunk. rank, chunk_id, doc_id, bm25_score, chunk_type, section_path, report_nm, corp_code, corp_name, rcept_dt, period, content, retrieval_text, source_refs, provenance |
| `think_trace` | object | 실행 요약 (아래 참조) |
| `answer` | string | 최종 답변. 어떤 경우에도 빈 문자열이 아닙니다 |

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
| `hcx_status` | HCX 사용 결과 (`success` / `skipped_*` / `fallback_*` / `disabled` / `not_configured`) |

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

### 임베딩 (CLOVA Studio)

| 변수 | 기본값 | 비고 |
|---|---|---|
| `FESTIVAL_EMBEDDING_PROVIDER` | `hash` | 운영에서는 `clova_studio` 필요 |
| `FESTIVAL_EMBEDDING_MODEL` | `festival-hash-embedding` | 운영값 `bge-m3` |
| `FESTIVAL_EMBEDDING_VERSION` | `v1` | 적재된 임베딩 버전과 일치해야 함 |
| `FESTIVAL_EMBEDDING_DIMENSIONS` | `1024` | DB 인덱스와 일치해야 함 |
| `FESTIVAL_EMBEDDING_API_URL` | 없음 | OpenAI 호환 `/v1/openai/embeddings` |
| `FESTIVAL_EMBEDDING_API_KEY` | 없음 | **비밀값** |
| `FESTIVAL_EMBEDDING_API_KEY_HEADER` | `Authorization` | |
| `FESTIVAL_EMBEDDING_API_KEY_PREFIX` | `Bearer` | |
| `FESTIVAL_EMBEDDING_TIMEOUT_SECONDS` | `60` | |
| `FESTIVAL_EMBEDDING_BATCH_SIZE` | `32` | |
| `FESTIVAL_EMBEDDING_MAX_LENGTH` | `8192` | |
| `FESTIVAL_EMBEDDING_MIN_BATCH_SIZE` | `1` | |
| `FESTIVAL_EMBEDDING_LONG_TEXT_SEGMENT_CHARS` | `1800` | 40003 분할 폴백 |
| `FESTIVAL_EMBEDDING_DEVICE` | `cpu` | 로컬 추론 경로 |
| `FESTIVAL_EMBEDDING_CUDA_OOM_RETRY` | `true` | 로컬 추론 경로 |

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

## 13. Safety & Fallback

- **답변은 항상 존재합니다.** HCX가 꺼져 있든, 건너뛰든, 거부되든, 결정적 답변이 서빙됩니다.
  빈 문자열이 반환되는 경로는 없습니다.
- **근거 없는 답변은 만들지 않습니다.** 근거가 부족하면 `answerable=false`로 표시하고
  확인이 필요하다는 결정적 문구를 냅니다. 이 경우 HCX는 호출조차 하지 않습니다.
- **부분 복구를 하지 않습니다.** 검증에 실패한 모델 출력은 고쳐 쓰지 않고 폐기합니다.
  잘못 추정한 값 하나가 결정적 답변 전체보다 위험하기 때문입니다.
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
│   ├── api/          # FastAPI 계층 (app, pipeline, schemas, settings)
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
