# qa-tool — 공시 Q&A 작성 도구

팀원이 **공시를 직접 보며** 질문·기대답변 카드를 만들 때 쓰는 **작성 전용 도구**입니다.

| 구분 | 설명 |
|---|---|
| **용도** | Q&A 초안 YAML/JSON 생성, 필드 체크리스트 |
| **범위** | `qa-tool/` 디렉터리만 — `app/`, Independent Eval Gold, `GET /answer`와 **무관** |
| **근거** | `data/corpus/universe.csv` · manifest 검색 (코퍼스 밖 데이터 사용 안 함) |
| **Git base** | 공용 기준 `origin/taeyoon` — `feat/qa-tool-curation` 등 별도 브랜치에서 작업 |

대회 제출·서버 API·에이전트 코드를 바꾸지 않습니다. 라우팅·파이프라인 메모(`negative_example`, `related_pipeline`)만 카드에 적을 수 있습니다.

---

## 1. CLI (권장)

레포 루트에서:

```bash
python qa-tool/draft.py "현대자동차 2025년 1분기 연결 매출액은?"

python qa-tool/draft.py --prefix HY "현대자동차 2025년 사업보고서 연결 매출액 구성은?"

python qa-tool/draft.py --prefix SP --batch qa-tool/examples/sector-probes-batch.txt --out qa-tool/output/

python qa-tool/draft.py --format json "고려아연 신규시설투자 투자종료일은?"
```

생성물은 `qa-tool/output/` (`output/*.yaml`은 gitignore).

---

## 2. 웹 UI

```bash
cd qa-tool/web
python -m http.server 8765
```

`http://127.0.0.1:8765/`

---

## 3. 출력 필드

| 필드 | 의미 |
|---|---|
| `question_id` | 고유 ID |
| `question` | 질문 원문 |
| `listed_name` / `corp_name` | universe.csv (DART 법인명 = 폴더 키) |
| `doc_group` | periodic / exchange / holding / major |
| `report_nm · period` | 보고서·기간 힌트 |
| `basis` | 연결 / 별도 / 해당없음 |
| `task_type (기대)` | 라우트 힌트 (참고) |
| `expected_behavior` | `normal_answer` / `clarification` / `answerable_false` |
| `must_include_in_answer` | 답에 반드시 포함할 표·숫자 |
| `must_NOT_invent` | 추론·코퍼스 밖 금지 |
| `evidence` | doc_id, section_path (공시 확인 후) |
| `negative_example` | 흔한 오답·함정 |
| `related_pipeline (taeyoon)` | **참고용** — 현재 `app/` P0-D·comparison 동작 메모 |
| `notes` | 정정본·보완 메모 |

**Clarification Gate** (수동 카드 — `examples/clarification-date-template.yaml`):

| 필드 | 의미 |
|---|---|
| `clarification_trigger` | rcept_dt 0건, corpus 밖, date_basis 모호 |
| `clarification_prompt` | 규칙 기반 재질문 (manifest 후보만) |
| `candidate_docs` | 제시 가능 doc_id·rcept_dt |
| `follow_up_question` / `follow_up_expected_doc` | 재확인 후 정답 |

**taeyoon 파이프라인 참고** (카드 작성 시):

- `app/reasoning/query_validation.py` — P0-D clarification·fail-closed
- `app/reasoning/query_understanding.py` — `comparison_frame`, comparison 파싱
- `think_trace.query_understanding.clarification_required` — clarification route
- derived compute(증가율·%p·집계) — **대부분 미구현** → partial / answerable=false 카드 병행

숫자·doc_id·기대답 본문은 공시 확인 후 사람이 채웁니다.

---

## 4. doc_group 판별 (도구 기준)

| doc_group | 공시 유형 |
|---|---|
| periodic | 사업·반기·분기 — 매출/손익/재무, **구성**, segment |
| exchange | 단일판매·공급, 시설투자 |
| holding | 대량보유상황 |
| major | 유상증자, CB, **자기주식·신탁 해지** |

**주의:** `자기주식취득신탁계약**해지**` → **major** (exchange `계약해지` substring 함정).

---

## 5. 예시 (`examples/`)

| 파일 | 내용 |
|---|---|
| `hy-revenue-breakdown.yaml` | 매출 **구성** vs 총액 |
| `hm-treasury-trust-termination.yaml` | 신탁 해지 + 소각 (major) |
| `hm04-inventory-yoy-comparison.yaml` | 연결 **재고자산** 당·전기 (comparative) |
| `clarification-date-template.yaml` | 날짜 clarification 빈 템플릿 |
| `hm07-clarification-date.yaml` | corpus 밖 2026-06-08 → 재질문 |
| `hm07-supply-contract-ok.yaml` | follow-up 2025-05-28 양성 |
| `sector-probes-batch.txt` | 20섹터 복합 질문 일괄 입력 |
| `sector-probes/_generate.py` | 상세 SP 카드 재생성 (PROBES 데이터 수정 후 실행) |
| `peer-compare-batch.txt` | 동일 산업 2사 peer 비교 질문 30개 일괄 입력 |
| `peer-compare/_generate.py` | peer 비교 PC 카드 재생성 |

### taeyoon 실제 `/answer` 파이프라인 (5단계 요약 아님)

카드의 `related_pipeline (taeyoon)` 메모는 아래 순서를 기준으로 합니다.

```text
질문
  → query_understanding (QueryPlan)
  → query_validation (P0-D)     RESOLVED 아니면 retrieval 차단 · clarification
  → [hcx_semantic_fallback]     최대 1회 · 재검증
  → hybrid retrieval            lexical + vector → RRF → rerank
  → [correction_expansion]      P0-A · retrieval 안
  → [corporate_event_expansion] P0-B · retrieval 안
  → multi_document_planner      P0-C · retrieval 이후 additive
  → orchestrator                task_router → resolver → answer_composer
  → answer_generator
  → answerability_guard         P0-D
  → hcx_verbalizer              실패 시 결정적 답변 유지
```

peer-compare 일괄 실행:

```bash
# 로컬 — DB 없이 P0-D·플래너·라우트만 (기본)
python scripts/run_peer_compare_batch.py --mode routing

# 서버 — Postgres·임베딩 연결 후 전체 E2E
FESTIVAL_HCX_ENABLED=false python scripts/run_peer_compare_batch.py --mode e2e
```

결과: `qa-tool/output/peer-compare-probe-full.json` (routing) 또는 `peer-compare-e2e.json` (e2e).

---

## 6. universe JSON

```bash
python qa-tool/draft.py --export-universe qa-tool/web/universe.json
```

---

## 7. 테스트

```bash
python -m pytest tests/test_qa_tool.py -q
```

에이전트 테스트와 분리된 **작성 도구 전용** 테스트입니다.
