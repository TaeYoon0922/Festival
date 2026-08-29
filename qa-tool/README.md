# qa-tool — 공시 Q&A 작성 도구

팀원이 **공시를 직접 보며** 질문·기대답변 카드를 만들 때 쓰는 **작성 전용 도구**입니다.

| 구분 | 설명 |
|---|---|
| **용도** | Q&A 초안 YAML/JSON 생성, 필드 체크리스트 |
| **범위** | `qa-tool/` 디렉터리만 — `app/`, Gold60, `GET /answer`와 **무관** |
| **근거** | `data/corpus/universe.csv` · manifest 검색 안내 (코퍼스 밖 데이터 사용 안 함) |

대회 제출·서버 API·에이전트 라우팅 코드를 바꾸지 않습니다. 라우팅 버그 메모(`negative_example`)만 카드에 적어 둘 수 있습니다.

---

## 1. CLI (권장)

레포 루트에서:

```bash
# 단일 질문 → stdout
python qa-tool/draft.py "현대자동차 2025년 1분기 연결 매출액은?"

# question_id 접두사
python qa-tool/draft.py --prefix HY "현대자동차 2025년 사업보고서 연결 매출액 구성은?"

# 여러 질문 (한 줄에 하나) → qa-tool/output/ 에 저장
python qa-tool/draft.py --prefix B --batch questions.txt --out qa-tool/output/

# stdin
type questions.txt | python qa-tool/draft.py --batch - --out qa-tool/output/

# JSON 형식
python qa-tool/draft.py --format json "고려아연 신규시설투자 투자종료일은?"
```

생성물은 기본적으로 `qa-tool/output/`에 두세요. (`output/*.yaml`은 gitignore — 커밋하지 않음)

---

## 2. 웹 UI (브라우저)

```bash
cd qa-tool/web
python -m http.server 8765
```

브라우저에서 `http://127.0.0.1:8765/` — 질문 입력 시 YAML 초안·필드 표를 바로 확인합니다.

---

## 3. 출력 필드

| 필드 | 의미 |
|---|---|
| `question_id` | 고유 ID (접두사·회사 약어 자동) |
| `question` | 질문 원문 |
| `listed_name` / `corp_name` | universe.csv 매칭 (DART 법인명 = 폴더 키) |
| `doc_group` | periodic / exchange / holding / major |
| `report_nm · period` | 보고서 종류·기간 힌트 |
| `basis` | 연결 / 별도 / 해당없음 |
| `task_type (기대)` | 에이전트 라우트 힌트 (참고용) |
| `must_include_in_answer` | 답에 반드시 포함할 표·숫자 |
| `must_NOT_invent` | 추론·코퍼스 밖 금지 |
| `evidence` | doc_id, section_path (공시 확인 후 채움) |
| `negative_example` | 흔한 오답·알려진 라우팅 함정 |
| `notes` | 정정본·clarification |
| `corpus_note` | universe 70개사 밖 회사일 때 경고 |

**숫자·doc_id·기대답 본문**은 공시를 연 뒤 사람이 채웁니다. 이 도구는 **양식 초안**만 만듭니다.

---

## 4. doc_group 판별 (도구 기준)

| doc_group | 공시 유형 | 볼 내용 |
|---|---|---|
| periodic | 사업·반기·분기 | 매출/손익/재무, 매출 **구성**, 부문, R&D |
| exchange | 단일판매·공급, 시설투자 | 계약금액, 투자종료일, 자기자본대비 |
| holding | 대량보유상황 | 보고자, 변동 전·후 주식수·비율 |
| major | 유상증자, CB, 합병, **자기주식·신탁** | 처분/취득/신탁계약 해지, 소각 예정 등 |

**주의:** `자기주식취득신탁계약**해지**`는 **major**입니다. `계약해지` 부분문자열 때문에 exchange로 오분류하기 쉬운 함정 질문입니다 (에이전트 라우팅 수정은 별도 작업).

---

## 5. 예시

`examples/` 디렉터리:

- `hy-revenue-breakdown.yaml` — 매출액 **구성** (총액 한 줄과 구분)
- `hm-treasury-trust-termination.yaml` — 신탁계약 해지 + 소각 (major)

---

## 6. universe JSON 갱신

70개사 목록을 웹 UI에 반영할 때:

```bash
python qa-tool/draft.py --export-universe qa-tool/web/universe.json
```

---

## 7. 테스트

```bash
python -m pytest tests/test_qa_tool.py -q
```

에이전트 테스트와 분리된 **작성 도구 전용** 테스트입니다.
