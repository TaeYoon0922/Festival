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
| 표준 포트(80) | `http://101.79.20.171` — 같은 서비스에 연결됨 |
| 인증 | 없음 |
| 접근 제한 | 없음 (전체 개방) |
| 문자 인코딩 | UTF-8 |
| 가동 기간 | 2026-09-07 ~ 2026-09-20 상시 |
| 운영 방식 | systemd(`festival-api.service`), `Restart=always`, 부팅 시 자동 기동 |

80번 포트는 `nat PREROUTING REDIRECT --dport 80 --to-port 8000`으로 8000번에 연결되어 있고,
그 규칙을 서비스 유닛의 `ExecStartPost`가 매 기동마다 다시 세웁니다. 재시작·재부팅 후에도
두 포트가 함께 살아나며, **제출한 `:8000` URL은 그대로 유효합니다.**

```bash
curl -s http://101.79.20.171/healthz        # 80
curl -s http://101.79.20.171:8000/healthz   # 8000
```

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

한 블록의 형식입니다.

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

| 줄 | 담기는 값 |
|---|---|
| 머리글 | `rank` · `corp_name` · `report_nm` · `rcept_dt` |
| `doc_id` | 공시 문서 식별자 |
| `chunk_id` | 결정적 청크 식별자 |
| 식별 줄 | `chunk_type`(`text`·`table`·`table_projection`) · `bm25_score` · `corp_code` · `period` |
| `섹션` | 문서 내 섹션 경로 (` > `로 연결) |
| `내용` | 청크 원문 |
| `source_refs` | 표 행·필드 단위 출처 (compact JSON) |
| `provenance` | 문서·섹션·표 추적 정보 (compact JSON) |

`retrieval_text`는 따로 싣지 않습니다. 그 고유 정보인 기업명·공시명·섹션 경로가 이미 블록
머리글이고, 청크 원문은 `내용`에 그대로 들어갑니다.

`내용`은 청크당 1,500자에서 줄 경계로 끊고 `…(이하 생략)`을 남깁니다. 서빙되는 청크에는
손익계산서 한 장이 통째로 들어오기도 해 열 건이면 4.8만 자에 달했는데, 필드 전체가 임의의
지점에서 잘리면 뒤쪽 청크가 통째로 사라집니다. 청크마다 경계를 두면 열 건이 모두 남고
잘린 지점이 눈에 보입니다.

검색 결과가 없으면 `검색된 공시 근거가 없습니다.`입니다.

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
| `answer_synthesis` | HyperCLOVA X가 답변을 작성한 경우 |
| `answer_narration` | HyperCLOVA X가 답변을 다시 쓴 경우 |
| `answer_lead` | 도입 문장만 작성한 경우 |

#### HyperCLOVA X가 관여하는 지점

이 에이전트가 쓰는 언어 모델은 **HyperCLOVA X(HCX-005) 하나뿐**이며 `temperature=0`으로
호출합니다. 붙는 위치는 결정적 답변이 이미 확정 수치를 진술하는지에 따라 갈립니다.

| 결정적 답변 | 동작 | `status` |
|---|---|---|
| 확정 수치를 이미 문장으로 진술함 | **모델 미호출.** 그대로 서빙 | `skipped_stated_figure` |
| 그 외 (서술형) | 모델이 공시 발췌를 읽고 답변 작성 | `answer_synthesis` |
| ↳ 검증에 걸린 경우 | 값을 가린 채 문장만 재작성 | `answer_narration` |
| ↳ 그것도 걸린 경우 | 도입 한 줄만 작성 | `answer_lead` |
| 모든 질의 | 질의 해석 확인 (계획은 불변) | `query_validation.hcx_status` |

`answer_synthesis`에서는 모델이 실제 수치를 봅니다. 대신 응답을 받은 뒤 **모델이 쓴 모든
숫자를 서빙된 발췌 안에서 자릿수 그대로 대조**하고, 없는 숫자·배수가 바뀐 숫자·없는 인용·
코퍼스 밖 기업명·투자 표현·비교 단정이 하나라도 있으면 응답 전체를 폐기합니다.

`answer_narration`에서는 수치·날짜·인용이 호출 전에 전부 자릿수 없는 토큰으로 치환되므로
**모델이 값을 보지 못하며**, 토큰이 하나라도 어긋나면 다시 쓰기가 폐기됩니다.

어느 경로가 쓰였든, 그리고 모두 거부되었든, **서빙된 수치·날짜·인용은 결정적 계층이 만든
값과 동일합니다.**

### 응답 예시

```json
{
  "question_id": "Q-001",
  "question": "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
  "retrieved_context": "[1] 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10\ndoc_id: periodic_20260310002820\nchunk_id: periodic_20260310002820:ch_70f7b399fe050756e113\nchunk_type: table | bm25_score: 12.34 | corp_code: 00126380 | 기준기간: 2025년 12월\n섹션: III. 재무에 관한 사항 > 2-2. 연결 손익계산서\n내용:\n| 열 1 | 제 57 기 |\n| --- | --- |\n| 매출액 (주30) | 333,605,938 |",
  "think_trace": "task_type: periodic_fact\nroute: periodic_fact_resolver\nstages: query_understanding > query_validation > task_router > evidence_builder > periodic_fact_resolver > periodic_evidence_selector > answer_composer > answer_generator > answerability_guard\nretrieval_count: 10\nselected_evidence_count: 10\nanswerable: true\nwarnings: annual_report_source_preferred\nhcx_status: skipped_no_compact_verified_claim\nanswer_synthesis: {\"status\":\"skipped_stated_figure\",\"used\":false}\nquery_validation: {\"status\":\"resolved\",\"retrieval_allowed\":true,\"hcx_status\":\"success\"}",
  "answer": "삼성전자의 2025년 연결기준 매출액은 333,605,938입니다. (공시 원문에 단위 표기 없음) [1]\n\n인용\n[1]\ndoc_id: periodic_20260310002820\nchunk_id: periodic_20260310002820:ch_70f7b399fe050756e113\n공시: 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10"
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

두 기업 비교 질문 30문항을 서빙 중인 엔드포인트에 한 건씩 보내 측정한 값입니다.

| 구간 | 소요 |
|---|---|
| 중앙 · 평균 | 5.2초 · 7.6초 |
| p90 · 최대 | 13.8초 · 22.0초 |
| 모델 호출이 없는 수치 질의 | 2~4초 |
| 서버 기동 직후 첫 질의 | 최대 60초 (BGE-M3 모델 지연 로딩) |

가장 느린 문항도 주최측 제한 300초의 7.3%입니다. **타임아웃은 180초 이상을 권장합니다.**

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

식별자만으로는 사람이 원문을 확인할 수 없으므로, 각 인용에 **공시명과 접수일** 줄을 함께
붙입니다.

```text
[1]
doc_id: periodic_20260310002820
chunk_id: periodic_20260310002820:ch_70f7b399fe050756e113
공시: 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10
```

---

## 운영

```bash
systemctl status festival-api      # 상태
systemctl restart festival-api     # 재시작
journalctl -u festival-api -n 50   # 로그
```

설정은 `/srv/festival/app/.env`와 유닛 파일
`/etc/systemd/system/festival-api.service`에서 읽습니다.
