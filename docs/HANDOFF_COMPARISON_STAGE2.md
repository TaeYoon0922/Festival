# 인수인계 — 기업 비교 2단계(조건부 순위)

작성 2026-09-04 · 제출 마감 2026-09-06 24:00

이 문서는 다음 작업자가 **처음부터 다시 조사하지 않고** 이어받도록 쓴 것입니다.
오늘 확인한 사실, 이미 막아 둔 함정, 시도했다가 되돌린 것을 모두 적었습니다.

---

## 0. 지금 상태

**브랜치** — 제출 브랜치는 `dev`입니다. 작업은 `feat/comparison-sidebyside`에서 했고
서버 `dev`에 cherry-pick으로 반영돼 있습니다.

| 커밋 | 내용 |
|---|---|
| `31fc839` | 라우팅이 후보를 전멸시킨 경우의 필터 완화 (OF4 수정) |
| `0287ce5` | 기업별 검색 + 인터리브 병합 (비교 1단계) |
| `9ec17b2` | `think_trace`에 `comparison_evidence` 필드 |
| `da758ca` | 각 다리를 질문의 연도로 스코프 |
| `12292d7` | 시스템 구조도 |
| `df17534` `30c27fa` `fce1a10` | 기술제안서 + 실측 반영 |

**공식 예시형 6문항 현재 결과** (`scripts/measure_official_examples.py`)

| ID | 유형 | 상태 |
|---|---|---|
| OF1 | 검색 Closed | 답변. 표 덤프, 단위 없음 (§4 참조) |
| OF2 | 검색 Open | 답변. 노이즈 근거 2건 |
| OF3 | 비교 Closed | 답변. 양사 2025년 수치가 1·2위. **순위는 미진술** ← 이 작업의 대상 |
| OF4 | 비교 Open | 답변. 전환사채 공시 6건 |
| OF5 | 복합 Closed | 정상 (근거 기반 부정 답변) |
| OF6 | 복합 Open | `answerable=False`. 벡터 부분 커버리지 여파 (§5) |

---

## 1. 해야 할 일 — 2단계 조건부 순위

OF3는 지금 근거를 나열하지만 **"더 큰 기업은?"에 답하지 않습니다.**

```
현재    [LG]  2025년 당기 중 총 10.5조원 [1]
        [SDI] 2025년 상반기 중 1조 8,430억원 [2]

목표    LG에너지솔루션이 10.5조원으로 삼성SDI(…)보다 큽니다 [1][2]
        (단, 아래 게이트를 모두 통과할 때만)
```

### 설계 — 기준 판정은 텍스트가 아니라 메타데이터로

**오늘 확인한 가장 중요한 사실입니다.**

```
LG 사업보고서(2025.12)   "2025년 당기 중 총 10.5조원"   ← 당기 = 연간
LG 반기보고서(2025.06)   "2025년 당기 중 총 5.8조원"    ← 당기 = 반년
```

**"당기"는 보고서에 상대적이라 텍스트로 판정하면 틀립니다.** `report_nm`이 곧 기준입니다.

```
사업보고서  →  연간
반기보고서  →  반기
분기보고서  →  분기
```

`report_nm`은 구조화된 메타데이터라 파싱이 필요 없습니다.

### 구현 순서

```
① 기업별로 같은 보고서 종류 선택
   연도만 물은 질문이면 사업보고서를 우선.
   한 기업이라도 그 종류에서 근거를 못 찾으면 → 1단계 출력으로 강등

② 그 근거 문장에서 금액 파싱
   "총 10.5조원" → 10_500_000_000_000
   한글 단위(조·억·만·천)는 유한하고 명확

③ 순위 게이트 — 전부 충족해야만 순위를 진술
   · 모든 기업이 같은 보고서 종류에서 나옴
   · 모든 금액이 깨끗하게 파싱됨
   · 합계 표지("총", "누적")가 있음  ← 부문별 금액과 구분하는 핵심
   · 연도가 동일

   하나라도 어긋나면 → 1단계 출력 그대로 (실패의 최악이 현재 동작)
```

### ③의 "합계 표지"가 왜 필수인가

```
"당사는 2025년 상반기 중 1조 8,430억원을 … 사용하였으며,
 각 부문별 투자금액은 에너지솔루션 부문 1조 7,994억원, 전자재료 부문 436억원입니다."
```

**한 문장에 금액이 셋입니다.** 합계 표지 없이 첫 숫자를 잡으면 우연히 맞고, 부문별을 잡으면
틀립니다. 표지가 없으면 declines해야 합니다.

### 파일 배치

```
app/reasoning/comparison_evidence.py   기존. 1단계가 여기 있음
                                       └ 여기에 순위 로직을 추가하거나
                                          comparison_ranking.py 를 신설
tests/test_comparison_evidence.py      기존 19개. 여기에 추가
```

`app/reasoning/company_comparison.py`(계약금액 레인)는 **건드리지 마십시오.** FINAL FREEZE이고
테스트 1,127줄이 붙어 있습니다.

---

## 2. 절대 하지 말 것 — 오늘 시도했다가 되돌린 것들

### ① `_comparison_from_query`에 `comparison_frame == "cross_company"` 추가

**테스트 17개가 명시적으로 금지합니다.**

```
test_a_frame_does_not_make_a_question_an_executable_comparison
test_a_frame_does_not_make_a_comparison_answerable
test_a_cross_company_frame_blocks_role_reinterpretation
```

`_comparison_frame`의 독스트링이 이유를 적어놨습니다 — 프레임은 **방화벽 신호**일 뿐 실행
권한이 아닙니다. 두 기업을 발행회사/보고자로 재해석하는 것을 막는 장치이고, 그건 계속
막아야 합니다. 1단계는 그래서 **방화벽을 뚫지 않고 옆에 문을 냈습니다**(검증 차단 지점에서
분기).

### ② 비교 계층 신규 구현

`dev`에 이미 `app/reasoning/company_comparison.py`(721줄) + 테스트 1,127줄이 있습니다.
계약금액 한정이지만 구조가 잘 짜여 있고, `_comparison_execution`의 **랭크 인터리브**가
한 기업 독식 문제를 이미 해결합니다. 새로 만들지 말고 그 패턴을 따르십시오.

### ③ 정기공시 답변의 자연어 문장화

**코퍼스에 단위가 없어서 불가능합니다.** 세 경로를 모두 확인했습니다.

| 경로 | 결과 |
|---|---|
| 청크 자신의 메타데이터 | 정기공시 표 922,431개 중 391,943개만 보유 (42.5%) |
| 같은 표의 다른 청크 | 해당 표가 청크 하나뿐인 경우가 많음 |
| 청크가 속한 섹션 본문 | 단위 없는 607,501개 중 회수 13,947개 (2.3%) |

삼성전자 매출액 청크(`periodic_20260310002820:ch_70f7b399fe050756e113`)는 값·연결/별도·기간은
있으나 단위가 없습니다. 되찾으려면 파서 개선 후 재파싱이 필요하고, 재파싱은 결정적 chunk ID를
바꿔 1,363,336개 임베딩을 전부 무효화합니다.

**단, 서술형 금액은 다릅니다.** "1조 8,430억원"처럼 단위가 문장 안에 있어서 2단계가 가능합니다.

### ④ 필터 완화를 strict-zero 밖으로 넓히기

`31fc839`가 구현한 완화는 **후보가 0일 때만** 동작합니다. FREEZE_LOG P1-B가 완화 사다리를
전부 측정하고 "company나 doc_group을 완화하면 recall·precision·latency가 동시에 나빠진다"고
기록했습니다. 트리거를 넓히지 마십시오.

---

## 3. 반드시 지킬 것

### 테스트는 HCX를 끄고 돌립니다

```bash
FESTIVAL_HCX_ENABLED=false python -m pytest tests/ -q
```

`.env`를 소스한 셸에서 그냥 돌리면 **실제 HCX API가 호출됩니다.** 비용이 들고, 같은 테스트가
실행마다 달라지며, `test_natural_language_pipeline_keeps_the_public_schema_and_selected_context`가
결정적 답변 대신 문장화된 답변을 받아 실패합니다. 코드 회귀가 아닙니다.

**기준선: `2711 passed, 14 skipped`** (2026-09-04 서버 dev 기준)

### 연도 필터는 `base_year`입니다

```sql
if years: conditions.append("d.base_year = ANY(%s)")
```

접수일이 아닙니다. **2025 사업보고서는 2026년 3월에 접수됩니다.** `rcept_dt`로 걸면 정답
문서를 버립니다.

그리고 `backend_filters`는 `period.is_fiscal and years`일 때만 연도 조건을 만듭니다.
`period_type='reference_year'`("2025년에")는 `is_fiscal`이 아니라 필터가 안 걸립니다.
1단계의 `evidence_subplan`이 이걸 다리 안에서만 보정합니다.

### `think_trace`는 선언된 모델입니다

`app/api/schemas.py`의 `ThinkTrace`에 필드를 추가하지 않으면 **직렬화에서 조용히 버려집니다.**
`exclude_if=lambda value: value is None`을 붙여 다른 응답의 키 집합을 바꾸지 마십시오.

### 동결 컴포넌트 재개봉 절차

`docs/FREEZE_LOG.md`에 정책이 있습니다. 재개봉하려면 실패 재현 · 위반된 불변식 특정 ·
회귀 테스트 추가 · 동결 컴포넌트 **밖의** 해법이 불가능한 사유 기록이 필요합니다.
"성능 개선"이나 "읽기 불편함"은 사유가 되지 않습니다.

---

## 4. 검증 방법

```bash
# 서버
cd /srv/festival/app && source .venv/bin/activate
set -a; source .env; set +a

FESTIVAL_HCX_ENABLED=false python -m pytest tests/ -q 2>&1 | tail -3

pkill -9 -f "app.api"; sleep 3
FESTIVAL_API_PORT=8020 FESTIVAL_HCX_ENABLED=false \
FESTIVAL_EMBEDDING_PROVIDER=bge_m3_local FESTIVAL_EMBEDDING_MODEL=BAAI/bge-m3 \
FESTIVAL_EMBEDDING_VERSION=6892b95fed65c899a30896eb40d619ae284d0455 \
FESTIVAL_EMBEDDING_DIMENSIONS=1024 FESTIVAL_EMBEDDING_MAX_LENGTH=2048 \
FESTIVAL_EMBEDDING_DEVICE=cpu \
  nohup python -m app.api > /tmp/api-8020.log 2>&1 &
sleep 40

python scripts/measure_official_examples.py --port 8020
```

OF3 단독 확인:

```bash
curl -sG http://127.0.0.1:8020/answer \
  --data-urlencode 'question_id=OF3' \
  --data-urlencode 'question=LG에너지솔루션과 삼성SDI 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin); t = d['think_trace']
print(json.dumps(t.get('comparison_evidence'), ensure_ascii=False))
for r in d['retrieved_context']:
    print(r['rank'], r.get('corp_name'), r.get('rcept_dt'), r.get('report_nm'))
print(d['answer'][:800])
"
```

**게이트가 declines할 때 1단계 출력이 그대로 나오는지**를 반드시 확인하십시오. 그게 이
설계의 안전 속성입니다.

---

## 5. 별도로 진행 중인 것 — 임베딩

로컬 RTX 4060에서 전체 코퍼스를 임베딩 중입니다.

```
provider   bge_m3_local · BAAI/bge-m3 · 6892b95fed65c899a30896eb40d619ae284d0455
설정       batch 32 · MAX_LENGTH 2048 · 길이순 정렬 입력(local_sorted.txt)
연결       SSH 터널 (localhost:55432 → 192.168.2.6:5432)
속도       약 48 청크/초
```

진행 확인:

```bash
psql -At -c "
SELECT COUNT(*), ROUND(100.0*COUNT(*)/1363336,1)
FROM chunk_embeddings WHERE embedding_model='BAAI/bge-m3';"
```

**부분 커버리지는 무커버리지보다 나쁩니다.** 임베딩된 청크만 lexical·vector 양쪽에서 점수를
받아 RRF에서 부당한 이득을 얻습니다(FREEZE_LOG에 같은 현상이 기록돼 있습니다). OF6가
`answerable=True → False`로 회귀한 원인이 이것으로 보이며, **100% 도달 후 재측정하면 회복될
가능성이 큽니다.**

완료되면 6문항을 다시 재고, `docs/TECHNICAL_PROPOSAL.md` §6.2의 수치를 갱신하십시오.

### 임베딩 설정 관련 주의

```
CLOVA API   429(코드 42901) rate limit. 배치 16 이상 또는 병렬 시 대량 실패
            → 배치 8 단일 프로세스가 유일하게 안전. 약 6.6/s
로컬 GPU    MAX_LENGTH를 8192로 두면 VRAM 초과 → WDDM이 시스템 RAM으로 넘겨
            10배 느려짐(죽지 않아서 알아채기 어려움). 2048이 적정
입력 정렬   char_count 순으로 정렬해야 패딩 낭비가 사라짐
            (정렬 전 1.08/s → 정렬 후 수십 배)
```

---

## 6. 남은 다른 작업

| 항목 | 내용 |
|---|---|
| README 정합성 | 임베딩 설정이 `clova_studio`로 적혀 있으나 실제는 `bge_m3_local` |
| 제안서 수치 갱신 | 임베딩 완료 후 §6.2 재측정값 반영 |
| 제출 3종 점검 | 소스코드 · 기술제안서 · 평가용 API 서버 정보(End-point + 명세) |
| OF3 문구 | 2단계를 안 할 경우, "기준 기간이 달라 우열은 판단하지 않았습니다"를 명시하는 편이 평가지표 '정보한계 대응'에 유리 |

---

## 7. 이 시스템의 일관된 원칙

새 기능을 넣을 때 이 기준으로 판단하십시오.

> **확신 있는 오답보다 정직한 한계 고지가 낫다.**

- 근거가 없으면 답을 만들지 않습니다
- 추출이 보장되지 않으면 실행하지 않습니다
- 검증에 실패하면 결정적 답변으로 되돌아갑니다
- 모델은 원문도, 숫자도, 인용도 보지 않습니다

2단계의 게이트가 엄격한 이유도 이것입니다. **순위를 못 매기는 것은 실패가 아니고, 틀린
순위를 말하는 것이 실패입니다.**
