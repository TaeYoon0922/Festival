# 평가용 API 명세

제10회 2026 미래에셋증권 AI Festival · 공시 Agent

---

## End-point

```
http://101.79.20.171:8000
```

| 항목 | 값 |
|---|---|
| 프로토콜 | HTTP |
| 인증 | 없음 |
| 문자 인코딩 | UTF-8 |
| 가동 기간 | 2026-09-07 ~ 2026-09-20 상시 |
| 운영 방식 | systemd(`festival-api.service`), `Restart=always`, 부팅 시 자동 기동 |

---

## `GET /healthz`

상태 확인용입니다. 파이프라인을 구성하지 않으므로 DB·임베딩·HCX 상태와 무관하게 응답합니다.

```bash
curl -s http://101.79.20.171:8000/healthz
```

```json
{"status": "ok"}
```

---

## `GET /answer`

### 요청

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question_id` | string | 예 | 호출자가 지정하는 식별자. 공백만 있으면 422 |
| `question` | string | 예 | 한국어 질문. 공백만 있으면 422 |

```bash
curl -sG "http://101.79.20.171:8000/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=삼성전자의 2025년 연결기준 매출액은 얼마인가?"
```

```python
import requests

response = requests.get(
    "http://101.79.20.171:8000/answer",
    params={"question_id": "Q-001", "question": "삼성전자의 2025년 연결기준 매출액은 얼마인가?"},
    timeout=180,
)
result = response.json()
```

### 응답 — 최상위는 정확히 5개 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `question_id` | string | 요청 값 그대로 |
| `question` | string | 요청 값 그대로 |
| `retrieved_context` | string | 답변 생성에 참고한 검색 문서 |
| `think_trace` | string | 실행 요약 |
| `answer` | string | 최종 답변. **어떤 경우에도 빈 문자열이 아닙니다** |

**다섯 필드의 값은 모두 string입니다.** 공지의 "모든 필드의 값은 문자열(string)타입입니다"를
따르며, `retrieved_context`의 구분 방식은 참가팀 재량이라는 안내에 맞춰 아래 형식을 씁니다.

### `retrieved_context` — chunk당 한 블록, 블록 사이는 `-` 60자 구분선

한 블록에 담기는 값입니다.

| 필드 | 표기 | 설명 |
|---|---|---|
| `rank` | integer | 서빙 순위 (1부터) |
| `chunk_id` | string | 결정적 청크 식별자 |
| `doc_id` | string | 공시 문서 식별자 |
| `bm25_score` | number | 어휘 검색 점수 |
| `chunk_type` | string | `text` · `table` · `table_projection` |
| `section_path` | array[string] | 문서 내 섹션 경로 |
| `report_nm` | string | 공시명 |
| `corp_code` | string | 기업 고유번호 |
| `corp_name` | string | 기업명 |
| `rcept_dt` | string | 접수일 `YYYY-MM-DD` |
| `period` | object | `base_year` · `base_month` |
| `content` | string | 청크 원문 |
| `retrieval_text` | string | 검색용 문맥(기업·공시·섹션 경로 포함) |
| `source_refs` | array | 표 행·필드 단위 출처 |
| `provenance` | object | 문서·섹션·표 추적 정보 |

### `think_trace` — 한 줄에 한 항목 (`key: value`)

**chain-of-thought가 아닙니다.** 어떤 컴포넌트가 실행되었고 무엇으로 결론 났는지에 대한
실행 요약이며, 모델의 내부 추론 텍스트는 포함되지 않습니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `task_type` | string \| null | 라우팅된 태스크 유형 |
| `route` | string | 사용된 resolver 또는 종료 사유 |
| `stages` | array[string] | 실행된 컴포넌트 이름 목록 (실행 순서) |
| `retrieval_count` | integer | 검색 결과 수 |
| `selected_evidence_count` | integer | 선택된 근거 수 |
| `answerable` | boolean | 근거로 답변 가능 여부 |
| `warnings` | array[string] | 생성 과정에서 남긴 경고 |
| `hcx_status` | string | HyperCLOVA X 사용 결과 |

조건부로 나타나는 필드입니다. 해당 컴포넌트가 실행되지 않으면 키 자체가 없습니다.

| 필드 | 나타나는 경우 |
|---|---|
| `correction` | 정정 그래프가 문서를 보강한 경우 |
| `corporate_event` | 이벤트 그래프가 문서를 보강한 경우 |
| `multi_document_planner` | 다중문서 완전성 계층이 개입한 경우 |
| `comparison_evidence` | 비교 질의를 기업별로 검색한 경우 |
| `query_understanding` · `query_validation` | 질의 검증 경로 |
| `answerability` | 답변 가능성 판정이 실행된 경우 |
| `clarification` | 역질문 판정이 실행된 경우 |
| `answer_narration` | 답변이 다시 쓰기 대상 길이였던 경우 (거의 모든 응답) |

`answer_narration.status`가 `success`면 HyperCLOVA X가 다시 쓴 답변이고, 그 외의 값이면
결정적 답변이 그대로 서빙된 것입니다. 어느 쪽이든 수치·날짜·인용은 동일합니다 — 모델은
값을 볼 수 없고, 토큰이 하나라도 어긋나면 다시 쓰기가 폐기됩니다.

### 응답 예시

```json
{
  "question_id": "Q-001",
  "question": "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
  "retrieved_context": "[1] 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10\ndoc_id: periodic_20260310002820\nchunk_id: periodic_20260310002820:ch_70f7b399fe050756e113\nchunk_type: table | bm25_score: 12.34 | corp_code: 00126380 | 기준기간: 2025년 12월\n섹션: III. 재무에 관한 사항 > 2-2. 연결 손익계산서\n내용:\n| 열 1 | 제 57 기 |\n| --- | --- |\n| 매출액 (주30) | 333,605,938 |",
  "think_trace": "task_type: periodic_fact\nroute: periodic_fact_resolver\nstages: query_understanding > query_validation > task_router > evidence_builder > periodic_fact_resolver > periodic_evidence_selector > answer_composer > answer_generator > answerability_guard\nretrieval_count: 10\nselected_evidence_count: 10\nanswerable: true\nwarnings: annual_report_source_preferred\nhcx_status: skipped_no_compact_verified_claim",
  "answer": "삼성전자의 연결기준 매출액 관련 공시 근거입니다.\n\n정기공시 근거 1\n근거 1 보고 기간: 2025년\n근거 1 보고서: 사업보고서 (2025.12)\n근거 1 재무제표 기준: 연결\n근거 1 단위: 해당 표에 단위 표기가 없어 공시 원문의 수치를 그대로 표시했습니다.\n확인된 사업 또는 공시 내용:\n1.\n내용: | 매출액 (주30) | 333,605,938 | [1]\n\n인용\n[1]\ndoc_id: periodic_20260310002820\nchunk_id: periodic_20260310002820:ch_70f7b399fe050756e113\n공시: 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10"
}
```

---

## 오류 응답

접속 문자열·자격증명·traceback은 **어떤 경우에도 응답에 포함되지 않습니다.**

| 상황 | 코드 | 본문 |
|---|---|---|
| `question` / `question_id` 누락 또는 공백 | 422 | FastAPI 검증 오류 |
| 데이터베이스 접속 불가 | 503 | `{"reason": "database_unavailable", "message": "…"}` |
| 임베딩 서비스 불가 | 503 | `{"reason": "embedding_unavailable", "message": "…"}` |
| 그 외 실패 | 503 | `{"reason": "internal_error", "message": "…"}` |

---

## 동작 특성

### 응답 시간

| 구간 | 소요 |
|---|---|
| 일반 질의 | 10~20초 |
| 서버 기동 직후 첫 질의 | 최대 60초 (BGE-M3 모델 지연 로딩) |

**타임아웃은 180초 이상을 권장합니다.**

기동 직후 예열은 systemd가 수행합니다(`ExecStartPost`). 서비스가 재시작되면
`scripts/warmup_api.py`가 포트를 기다렸다가 질문 하나를 보내 모델을 적재하므로, 재시작
직후에 들어온 첫 평가 요청도 일반 구간에 들어옵니다. 예열이 실패해도 서비스 기동은
막지 않습니다.

### 답변이 항상 존재합니다

근거를 찾지 못한 경우에도 빈 문자열이 아니라 **무엇을 확인하지 못했는지**를 답합니다.

| 상황 | `route` | 답변 성격 |
|---|---|---|
| 질문이 모호함 | `clarification` | 필요한 정보를 되묻습니다 |
| 지원하지 않는 요청 유형 | `unsupported` | 지원 범위를 고지합니다 |
| 코퍼스 범위 밖 | `out_of_scope` | 제공 데이터 범위를 고지합니다 |
| 근거 부족 | `insufficient_evidence` | 확인하지 못했음을 고지합니다 |
| 개인정보 포함·요구 | `sensitive_input` | 답변하지 않음을 고지합니다 |

### 개인정보 입력 처리

주민등록번호·연락처·이메일 등 개인 식별자가 포함되었거나, 공시가 공개하지 않는 개인
연락처·주소를 요구하는 질문은 **검색과 모델 호출 이전에** 종료합니다. 이 경우
`retrieved_context`는 비어 있으며, 이는 근거를 숨긴 것이 아니라 코퍼스를 조회하지 않았기
때문입니다.

공시가 법적으로 공개하는 정보 — 대표이사, 최대주주, 보고자 — 에 대한 질문은 정상
처리됩니다.

### 근거 표시

모든 사실 문장은 `[n]` 형태의 인용을 가지며, 각 인용은 `answer` 말미의 인용 목록에서
`doc_id`와 `chunk_id`로 식별됩니다. 그 식별자는 `retrieved_context`의 항목과 일치합니다.

---

## 운영

```bash
systemctl status festival-api      # 상태
systemctl restart festival-api     # 재시작
journalctl -u festival-api -n 50   # 로그
```

설정은 `/srv/festival/app/.env`와 유닛 파일
`/etc/systemd/system/festival-api.service`에서 읽습니다.
