# TEAM_GUIDE — Festival 팀 운영 가이드

> **서버는 공용 실행 환경이고, 개발은 각자 로컬 브랜치에서 한다.
> 서버는 언제나 `dev`만 본다. 근거는 주최측 corpus뿐이고 외부 데이터를 붙이지 않는다.
> DB·청킹·임베딩은 건드리지 말고, 질문 실패 원인을 먼저 분류한 뒤 최소 범위만 수정한다.
> 수정 후 pytest + 기존 Gold60 회귀 테스트를 통과한 코드만 서버에 반영한다.**

아키텍처·API 스펙 전체는 [README.md](README.md)를 보세요.

---

## 0. 절대 하지 말 것 (읽고 시작)

| 금지 | 이유 |
|---|---|
| **OpenDART API·뉴스·웹 검색 등 외부 데이터 연결** | **대회 규칙 위반**입니다. 근거는 주최측 corpus뿐 → [1번](#1-대회-규칙--외부-데이터-금지) |
| **`GET /answer` 응답 구조 변경** | 제출 계약입니다. 내부는 바꿔도 외부 인터페이스는 고정 → [2번](#2-api-계약-freeze) |
| **NCP 리소스 삭제·반납, Docker volume 삭제** | 코드 사고보다 복구가 훨씬 어렵습니다 → [3번](#3-인프라-보호--지우면-못-되돌리는-것) |
| **확인 없이 서버 정지 / DB 서버까지 같이 정지** | 남의 Gold60이 중간에 끊깁니다. DB 서버를 끄면 팀 전체가 막힙니다 → [8번](#8-서버-종료--재기동--하루-작업-마감-절차) |
| **서버 공용 `.venv`에 개인이 `pip install`** | 남의 실행 환경을 조용히 바꿉니다 → [3번](#3-인프라-보호--지우면-못-되돌리는-것) |
| **서버에서 `dev` 아닌 브랜치 checkout** | `/srv/festival/app`은 **하나의 working tree**다. 내가 바꾸면 남이 쓰던 서버 코드도 같이 바뀐다 → [6번](#6-서버는-언제나-dev만-본다-제일-중요) |
| **서버에서 `git reset --hard` / `git clean -fd`** | 남이 검증 중인 상태와 로컬 산출물을 통째로 날립니다 → [6번](#6-서버는-언제나-dev만-본다-제일-중요) |
| **재적재 / 재임베딩 / 재청킹 / reparse** | 서버 DB는 검증 완료 상태다. corpus·chunk·embedding·schema 전부 이미 올라가 있다 |
| **DB `DROP` / `TRUNCATE` / provenance backfill / 전체 재적재** | 복구 비용이 가장 큰 작업. 일반 query·resolver·presentation 튜닝에는 DB를 만질 이유가 없다 |
| **서버에서 직접 코드 수정** | 서버는 실행 환경이다. 서버 편집분은 다음 `git pull`에서 사라지거나 충돌을 만든다 |
| **공용 브랜치(`dev`, `main`)에서 바로 개발** | 여러 명이 같은 브랜치를 만지면 회귀 원인 추적이 불가능해진다 |
| **기존 tag 이동 / force push** | 태그는 복구용 앵커다. 한 번 옮기면 되돌아갈 지점이 사라진다 |
| **`.env`, `.env.backup`, API key, DB password 커밋** | 단톡·이슈·PR 본문에도 붙여넣지 않는다 |
| **root 비밀번호 / GitHub PAT를 단톡으로 전달** | 각자 SSH 공개키를 등록한다 → [5번](#5-서버-접속과-보안) |
| **개발 포트(8010·8011…)를 인터넷에 개방** | ACG에 열지 말고 SSH 터널을 쓴다 → [5번](#5-서버-접속과-보안) |
| **Gold60 특정 문항·회사·값에 맞춘 튜닝** | Gold60은 시험문제가 아니라 회귀 테스트다 → [12번](#12-수정했으면-반드시-세-가지-확인) |
| **특정 회사명·날짜·값·question_id 하드코딩** | "파마리서치면 이렇게" 식 수정 금지. 항상 다른 회사·질문에도 적용되는 일반 규칙이어야 한다 |
| **검증된 factual context 삭제로 답변 줄이기** | 이미 한 번 `requested_fields`만 남겼다가 Gold60 evidence term이 크게 떨어졌다. 값은 유지하고 **표현만** 압축한다 |
| **presentation 단계에서 event 임의 병합/삭제** | 국민연금공단 / 국민연금기금처럼 비슷해 보여도 resolver가 별도 event로 준 것은 합치지 않는다. 그건 별도의 entity-normalization 작업이다 |
| **답이 틀렸다고 HCX prompt부터 수정** | 공시 선택·라우팅·resolver는 전부 규칙/DB 기반이다. HCX는 제한적 문장화(verbalization)만 한다 |

---

## 1. 대회 규칙 — 외부 데이터 금지

**이건 기술 안전 문제가 아니라 실격 사유입니다.** 답이 안 나온다고 밖에서 채우면 안 됩니다.

- **답변 근거로 쓸 수 있는 것은 주최측이 제공한 공시 corpus뿐입니다.**
- **OpenDART API, 뉴스, 위키, 검색엔진을 runtime fallback으로 붙이지 않습니다.**
  "일단 답이라도 나오게" 하려고 붙이는 것이 가장 흔한 위반 경로입니다.
- **정답이 corpus에 없으면 외부에서 채우거나 추측하지 않습니다.**
- **corpus 범위 밖 질문은 `answerable=false` / 근거 부족으로 처리합니다.** 그게 정답 동작입니다.
- **LLM이 기억하는 일반지식으로 공시 사실을 보충하지 않습니다.**
  HCX가 아는 것 같아 보여도 그건 근거가 아닙니다.

corpus 범위 (README 기준):

| 항목 | 값 |
|---|---|
| 기간 | 2023-01-01 ~ 2026-03-31 |
| 회사 | 70개 |
| 문서 | 4,204건 |

이 범위 밖을 묻는 질문에 답이 안 나오는 것은 **버그가 아닙니다.**
"모르겠다 / 근거가 부족하다"로 나가는 것이 규칙에 맞는 동작입니다.

### 헷갈리기 쉬운 구분 — 외부 *서비스* vs 외부 *데이터*

금지 대상은 **근거(evidence)를 외부에서 가져오는 것**입니다.

| | 허용 여부 |
|---|---|
| CLOVA Studio 임베딩 (`bge-m3`) — 우리 corpus를 벡터화 | **허용.** 데이터는 우리 corpus입니다 |
| HyperCLOVA X — 이미 검증된 답을 문장으로 다듬기 | **허용.** 새 사실을 추가하지 않는 verbalizer입니다 |
| OpenDART API로 공시를 새로 조회 | **금지.** 외부에서 근거를 가져옵니다 |
| 뉴스·웹 검색으로 수치 보완 | **금지** |
| HCX에게 "네가 아는 대로 채워줘" | **금지.** 모델 기억은 근거가 아닙니다 |

새 외부 호출을 코드에 넣기 전에 **반드시 팀에 먼저 물어보세요.**

---

## 2. API 계약 Freeze

제출 인터페이스입니다. **내부 구조는 얼마든지 바꿔도, 외부 계약은 고정입니다.**

`GET /answer`의 최상위 필드는 **정확히 이 5개**이고 순서·이름·개수를 바꾸지 않습니다:

```
question_id
question
retrieved_context
think_trace
answer
```

- **필드를 추가하지 않습니다.** 디버깅용 값이 필요하면 `think_trace` 안에 넣으세요.
- **필드를 빼거나 이름을 바꾸지 않습니다.**
- **`GET /healthz`를 유지합니다.**
- **presentation을 개선한다고 API schema를 건드리지 않습니다.**
  답변 문자열(`answer`)의 내용을 바꾸는 것과 schema를 바꾸는 것은 전혀 다른 일입니다.

계약을 지키는 테스트가 이미 있습니다 — `tests/test_answer_api.py`의
`test_returns_exactly_the_five_contract_fields`. 필드를 하나라도 더하거나 빼면 여기서 바로 실패합니다.
**이 테스트가 실패했는데 테스트를 고치는 건 계약을 깨는 것입니다.**
schema를 바꾸는 PR은 팀 합의 없이 머지하지 않습니다.

---

## 3. 인프라 보호 — 지우면 못 되돌리는 것

코드 사고는 태그로 복구됩니다. **클라우드 리소스와 volume 삭제는 복구가 안 됩니다.**

### NCP 콘솔 — 삭제·반납 금지

- **서버 반납/삭제 금지**
- **Block Storage 삭제 금지**
- **Public IP 반납 금지**
- **NAT Gateway 삭제 금지**
- **VPC / Subnet / ACG 삭제 금지**

> 비용 절약이 필요하면 **'정지'까지만** 합니다.
> **삭제·반납은 관리자 확인 없이 하지 않습니다.** 되돌릴 방법이 없습니다.

### Docker / DB volume

- **`docker system prune --volumes` 금지** — DB volume까지 같이 날아갑니다.
- **`docker compose down -v` 금지** — `-v`가 volume 삭제입니다.
- **`festival-postgres` 컨테이너 삭제 금지**
- **`/srv/festival-data/postgres-data` 삭제·이동 금지**

디스크가 부족해 정리가 필요하면 `--volumes` 없이, 그리고 **혼자 판단하지 말고 먼저 물어보세요.**
DB를 직접 안 건드려도 volume 하나 지우면 거기서 끝입니다.

### 서버 공용 `.venv`

`/srv/festival/app/.venv`는 **모두가 함께 쓰는 하나의 환경**입니다.

- **개인이 `pip install` / `pip uninstall` 하지 않습니다.**
  내 코드가 돌게 만든 그 설치가 남의 실행 결과를 조용히 바꿉니다.
- 새 dependency가 필요하면 **해당 requirements 파일을 PR로 수정** → 팀 합의 → 서버에 설치.
  실행에 필요하면 `requirements.txt`, 테스트에만 쓰면 `requirements-dev.txt`입니다.
- 로컬에서 자유롭게 실험하는 건 괜찮습니다. 서버 공용 환경만 보호합니다.

---

## 4. 처음 한 번 — 로컬 세팅

**로컬은 코드와 단위 테스트를 위한 곳입니다.** 실제 PostgreSQL을 쓰는 통합 질문 검증은 서버에서 합니다.

```bash
git clone https://github.com/TaeYoon0922/Festival.git
cd Festival

python --version                  # 3.12 이상 (제출 이미지는 python:3.12-slim)

python -m venv .venv
source .venv/bin/activate         # Windows: .\.venv\Scripts\activate

pip install -r requirements-dev.txt   # 실행 의존성 + pytest를 한 번에

git checkout dev
git pull
git checkout -b feature/<이름>-<주제>
```

세팅이 됐는지 확인 — **DB 없이 전부 통과해야 정상입니다**:

```bash
python -m pytest -q
```

### 로컬에 가져오지 않는 것

- **`.env`를 로컬로 복사하지 않습니다.** 서버에 있는 것을 서버에서만 씁니다.
- **DB 덤프·corpus 원본을 로컬로 내려받지 않습니다.** 수십 GB이고, 어차피 서버 DB를 씁니다.
- `requirements-embedding.txt`는 로컬 BGE-M3 추론용 **옵션**입니다. 설치할 필요 없습니다.

### requirements 파일이 세 개인 이유

| 파일 | 언제 |
|---|---|
| `requirements.txt` | 실행에 필요한 것. **`Dockerfile`이 이것만 복사합니다** |
| `requirements-dev.txt` | 위 + `pytest`. **개발자는 이걸 설치합니다** |
| `requirements-embedding.txt` | 로컬 BGE-M3 추론용 옵션. 안 써도 됩니다 |

새 dependency를 추가할 때 **어느 파일에 넣을지 먼저 생각하세요.**
테스트에만 쓰는 걸 `requirements.txt`에 넣으면 제출 이미지가 그만큼 커집니다.

단위 테스트는 DB·임베딩·HCX 없이 전부 통과합니다. DB가 필요한 건
`/answer` 실검증과 Gold60 회귀, **두 가지뿐**이고 둘 다 서버에서 합니다.

---

## 5. 서버 접속과 보안

서버는 두 대입니다. **평소 쓰는 건 앱 서버 하나뿐입니다.**

| 서버 | 접속 | 용도 |
|---|---|---|
| **앱 서버** | `ssh root@<APP_SERVER_IP>` | 코드 실행, API 기동, Gold60 평가. **여기만 씁니다** |
| DB 서버 | `ssh root@<DB_SERVER_IP>` | PostgreSQL + pgvector. **들어갈 일이 거의 없습니다** |

> **실제 IP는 이 문서에 적지 않습니다.** 이 저장소는 public이라 커밋하면 그대로 공개됩니다.
> 주소는 팀 내부(단톡·DM)에서 따로 전달받으세요. 각자 로컬에서 `~/.ssh/config`에
> 별칭으로 저장해 두면 편합니다.

DB 서버는 사설 IP라 노트북에서 바로 붙지 않습니다. 앱 서버를 거쳐야 합니다.
그리고 애초에 **들어갈 일이 없습니다** — 0번 표의 DB 금지 항목을 다시 보세요.
query·resolver·presentation 튜닝은 전부 앱 서버에서 끝납니다.

작업 시작 시퀀스 (앱 서버):

```bash
ssh root@<APP_SERVER_IP>
cd /srv/festival/app
source .venv/bin/activate
set -a
source .env
set +a
```

- `.env`는 **서버에 이미 있습니다.** 새로 만들거나 덮어쓰지 마세요.
- `.env` 내용을 화면 캡처·채팅·문서에 복붙하지 마세요.

### 접속 계정 — 비밀번호를 돌리지 않습니다

- **root 비밀번호나 GitHub PAT를 단톡으로 보내지 않습니다.** 한 번 뿌린 자격증명은 회수가 안 됩니다.
- 각자 **SSH 공개키**(`~/.ssh/id_ed25519.pub`)를 만들어 관리자에게 전달하고,
  서버 `~/.ssh/authorized_keys`에 등록하는 방식으로 붙습니다.
- 키가 다 등록되면 서버의 비밀번호 로그인은 꺼 두는 편이 안전합니다
  (`sshd_config`의 `PasswordAuthentication no`).

키가 없다면:

```bash
ssh-keygen -t ed25519 -C "<이름>"
cat ~/.ssh/id_ed25519.pub          # 이 한 줄만 전달. 개인키(.pub 없는 파일)는 절대 전달 금지
```

### 방화벽(NCP ACG) — SSH는 `/32`로만 엽니다

- 접속할 사람의 **현재 공인 IP만** SSH 22번에 `/32`로 추가합니다.
- **`0.0.0.0/0`으로 열지 않습니다.** 편하다고 열어두면 그 순간부터 자동 스캐닝 대상이 됩니다.
- 자기 공인 IP 확인: `curl -s ifconfig.me`
- 카페·집·핫스팟을 옮기면 IP가 바뀝니다. 그때마다 추가하고, **안 쓰는 규칙은 지웁니다.**
- **ACG 자체를 삭제하지 않습니다** ([3번](#3-인프라-보호--지우면-못-되돌리는-것)).

### 개발 포트는 인터넷에 열지 않습니다 — SSH 터널을 씁니다

8010·8011 같은 개발 API 포트를 ACG에 여는 것은 **인증 없는 API를 공개하는 것**입니다.

- 서버 안에서 `curl localhost:8010`으로 테스트하는 건 괜찮습니다.
- 노트북 브라우저에서 보고 싶으면 터널을 쓰세요:

```bash
ssh -L 8010:127.0.0.1:8010 root@<APP_SERVER_IP>
```

터널을 연 뒤에는 **노트북의** `http://localhost:8010`으로 접근됩니다. ACG는 그대로 닫아 둡니다.

---

## 6. 서버는 언제나 `dev`만 본다 (제일 중요)

`/srv/festival/app`은 **한 개의 git working tree**입니다. 사람마다 폴더가 따로 있는 게 아닙니다.

> **A가 서버에서 `git checkout feature/a`를 하는 순간,
> B가 쓰던 서버 코드도 `feature/a`가 됩니다.**
> B는 아무것도 안 했는데 자기 질문 결과가 바뀝니다. 그리고 왜 바뀌었는지 모릅니다.

이건 이론이 아니라 이 구조에서 반드시 일어나는 일입니다. 그래서 규칙은 하나입니다.

### 서버에서 `git checkout`은 `dev`에만 합니다

**남의 feature branch는 물론이고, 내 feature branch도 서버에서 checkout하지 않습니다.**

기본 흐름은 이 순서를 벗어나지 않습니다:

```
로컬 feature branch  →  push  →  PR  →  dev merge  →  서버에서 dev pull  →  API 재시작
```

### 서버에서 쓰는 git 명령은 이 네 개뿐입니다

```bash
cd /srv/festival/app
git status --short                 # 비어 있어야 합니다
git branch --show-current          # dev 여야 합니다
git fetch origin
git pull --ff-only origin dev
```

- **`git status --short`가 깨끗하지 않으면 pull하지 말고 멈추세요.** 누군가 뭔가 하고 있는 것입니다.
  단톡에 먼저 물어보세요. 남의 검증 중간에 바닥을 빼는 셈이 됩니다.
- **`--ff-only`를 씁니다.** 서버에서 머지 커밋이 만들어지는 걸 막습니다.
  이게 거부되면 `dev`가 앞서 있다는 뜻이니 그것도 멈추고 확인할 신호입니다.
- **`git reset --hard`, `git clean -fd`는 서버에서 임의로 쓰지 않습니다.**
  남이 검증 중인 상태와 로컬 산출물(평가 결과, 로그)을 통째로 날립니다.
  정말 필요하면 단톡에 먼저 알리세요.

### 예외 — merge 전에 실제 DB 테스트가 꼭 필요할 때

단위 테스트로는 못 잡고 실제 PostgreSQL이 있어야만 확인되는 변경이 있습니다.
그럴 때만, **한 번에 한 명씩 순차적으로** 합니다.

1. 단톡에 **"서버 잡습니다 — `<브랜치명>`"** 이라고 먼저 알립니다.
2. 작업합니다.
3. 끝나면 **반드시 되돌립니다.** 이게 빠지면 다음 사람이 남의 코드로 테스트하게 됩니다.

```bash
cd /srv/festival/app
git status --short              # 깨끗한지 먼저 확인
git fetch origin
git checkout <내-브랜치>        # 잡았다고 알린 뒤에만
git pull --ff-only origin <내-브랜치>
# → API 재시작 (7-C), 테스트

# 끝나면 즉시
git checkout dev
git pull --ff-only origin dev
# → API 재시작
```

4. 단톡에 **"서버 놓습니다"** 라고 알립니다.

### 나중에 — `git worktree`로 사람별 실행 폴더

동시 검증이 잦아지면 순번 대기가 병목이 됩니다. 그때는 working tree를 분리하면 됩니다.

```bash
# (아직 도입 전 — 팀 합의 후 한 번에 만듭니다)
cd /srv/festival/app
git worktree add /srv/festival/wt-<이름> <브랜치>

cd /srv/festival/wt-<이름>
source /srv/festival/app/.venv/bin/activate      # venv는 공유
set -a; source /srv/festival/app/.env; set +a    # .env도 공유 (복사하지 않습니다)
FESTIVAL_API_PORT=80<번호> FESTIVAL_HCX_ENABLED=false python -m app.api
```

`/srv/festival/app` 본체는 **계속 `dev`로 고정**해 두고, 개인 검증은 각자 worktree에서 합니다.
지금은 아직 안 만들어져 있으니, 필요해지면 먼저 이야기하세요.

---

## 7. 서버에서 API 켜기

### 먼저 "지금 뭐가 떠 있는지" 확인

순서가 제일 중요합니다. **띄우기 전에 확인부터.**
Docker로 떠 있는데 `.venv`로 또 띄우면 포트가 겹치고, 반대로 옛 컨테이너가 계속 응답합니다.

```bash
docker ps                                  # 컨테이너가 떠 있나
ps -ef | grep "app.api" | grep -v grep     # host python으로 떠 있나
ss -tlnp | grep -E "8000|801"              # 어느 포트를 누가 잡고 있나
```

### 포트는 단톡에 점유표를 두고 씁니다

```
8000  (제출/데모 예약 — 개인 사용 금지)
8010  태윤 — query routing
8011  A    — periodic
8012  B    — holding
```

테스트가 끝나면 **자기 프로세스만** 종료하고 단톡에 `8011 반환`이라고 남깁니다.
남의 포트에 뜬 프로세스를 임의로 죽이지 마세요.

### 개발 중에는 `.venv`, 최종 확정 후에 Docker

| 상황 | 무엇으로 | 포트 |
|---|---|---|
| **팀 개발·튜닝 중 (지금)** | **host `.venv`** | 8010, 8011 … 각자 |
| 제출/데모 최종본 | Docker | 8000 |

**지금 Docker 이미지는 최신 tuning 코드가 아닐 수 있습니다.** 개발 중에 Docker로 테스트하면
며칠 전 코드를 보게 됩니다. 반드시 `.venv`로 테스트하세요.

### A. host `.venv`로 띄우기 — 개발 중엔 이걸 씁니다

**HCX는 꺼 두고 시작합니다.**

```bash
cd /srv/festival/app
source .venv/bin/activate
set -a; source .env; set +a

FESTIVAL_API_PORT=8010 \
FESTIVAL_HCX_ENABLED=false \
python -m app.api
```

`FESTIVAL_HCX_ENABLED`의 기본값은 `true`입니다. 그냥 띄우면 서버 `.env` 설정에 따라
HCX가 켜진 채로 돕니다. 여러 명이 질문을 마구 던지는 상황에서 그러면:

- **API 호출 비용**이 사람 수만큼 붙고
- **rate limit**에 걸려 남의 테스트까지 막히고
- **같은 질문에 다른 문장**이 나와서 "고쳐진 건지 아닌지" 판단이 안 됩니다.

query·resolver·presentation 튜닝은 **HCX와 무관합니다.** 꺼도 검증에 아무 지장이 없습니다.
끈 상태에서는 `think_trace.hcx_status`가 `disabled`로 나옵니다 — 정상입니다.

**HCX 문장화 자체를 테스트할 때만** 명시적으로 켭니다:

```bash
FESTIVAL_API_PORT=8010 FESTIVAL_HCX_ENABLED=true python -m app.api
```

> **Gold60은 예외입니다.** Gold60은 검증된 baseline과 **동일한 조건**에서 돌려야 비교가 성립합니다.
> `FESTIVAL_HCX_*`나 `FESTIVAL_API_*`를 앞에 붙이지 말고, 서버 `.env` 그대로 실행하세요 → [12-(3)](#3-gold60-회귀-앱-서버에서-한-번에-한-명만)

터미널을 놓아 주려면 백그라운드로:

```bash
FESTIVAL_API_PORT=8010 FESTIVAL_HCX_ENABLED=false \
  nohup python -m app.api > /tmp/api-8010.log 2>&1 &
tail -f /tmp/api-8010.log
```

기본 포트 8000은 제출/데모용으로 비워 둡니다.
**어떤 포트도 ACG에 열지 않습니다** ([5번](#5-서버-접속과-보안) 참조).

### B. Docker로 띄우기 — 최종 버전 확정 후

```bash
cd /srv/festival/app
docker build -t festival-api .

docker run -d --name festival-api \
  --env-file .env \
  -p 8000:8000 \
  festival-api

curl -s http://localhost:8000/healthz
```

- `Dockerfile`의 `CMD`는 `python -m app.api`, `EXPOSE 8000`입니다.
- 검색 데이터는 이미지가 아니라 **PostgreSQL에 있습니다.** raw 코퍼스를 마운트할 필요가 없습니다.
- `.env`는 `.dockerignore`에 있어 **이미지에 안 들어갑니다.** 반드시 `--env-file .env`로 넘기세요.
- **`festival-api` 컨테이너만 다룹니다.** `festival-postgres`는 건드리지 않습니다
  ([3번](#3-인프라-보호--지우면-못-되돌리는-것)).

### C. 코드 갱신 후에는 반드시 재시작

`git pull`만 하고 예전 프로세스를 그대로 두면 **옛 코드가 계속 응답합니다.**

```bash
# .venv로 떠 있던 경우
ps -ef | grep "app.api" | grep -v grep
kill <PID>
FESTIVAL_API_PORT=8010 FESTIVAL_HCX_ENABLED=false python -m app.api
```

```bash
# Docker로 떠 있던 경우 — 이미지 재빌드까지 해야 반영됩니다
docker stop festival-api && docker rm festival-api
docker build -t festival-api .
docker run -d --name festival-api --env-file .env -p 8000:8000 festival-api
```

> **`docker restart`만으로는 코드가 안 바뀝니다.** 코드가 이미지 안에 복사돼 있어서
> `docker build`를 다시 해야 합니다. 이것 때문에 "분명 고쳤는데 왜 그대로지?"가 나옵니다.

재시작 확인은 `/healthz`가 아니라 **실제로 고친 질문**으로 하세요. `/healthz`는 파이프라인을
구성하지 않으므로 옛 코드에서도 `ok`를 돌려줍니다.

---

## 8. 서버 종료 / 재기동 — 하루 작업 마감 절차

> **서버를 끄기 전에 다른 사람이 Gold60/API를 쓰는지 확인하고, DB 컨테이너나 볼륨은 건드리지 않는다.
> NCP에서는 "정지"만 누르고 "반납/삭제"는 절대 누르지 않는다.
> 다시 켤 때 둘 다 꺼져 있었다면 DB 서버를 먼저 켠다.**

**NCP에서 바로 `정지`부터 누르지 않습니다.** 순서가 있습니다.

```
종료:  앱 작업 정리 → API/평가 프로세스 종료 → Git 상태 확인
       → DB 상태 확인 → sync → 앱 서버 정지 → DB 서버 정지

재기동: DB 서버 시작 → PostgreSQL 확인 → 앱 서버 시작
       → Git/환경 확인 → API 실행
```

### 왜 이 절차가 필요한지 — 실제로 이랬습니다

2026-08-22에 처음 종료해 보니 서버 상태가 이랬습니다:

```
branch      tuning/...          ← dev가 아님
untracked   진단 JSON 12개       ← git clean -fd 였으면 전부 소멸
host API    3개 실행 중
Docker API  2개 실행 중
```

여기서 `정지` 버튼만 눌렀다면 다음 날 **낯선 브랜치에서 시작**하게 됩니다.
그리고 `git clean -fd`로 "정리"했다면 진단 결과 12개가 그대로 사라졌을 것입니다.
**`reset --hard` / `clean -fd` 금지 규칙이 실제로 필요한 이유가 이날 확인됐습니다.**

### 종료 전 표준 상태 (shutdown gate)

아래가 되면 꺼도 됩니다. **여기까지 확인하고 `sync → 앱 서버 정지 → DB 서버 정지`로 갑니다.**

```
App server shutdown-ready:

  Git branch      dev
  Git status      clean
  Host API        none
  Docker API      none
  Evaluation      none

DB server:

  festival-postgres   Up
```

### 1. 다른 사람이 쓰고 있는지 먼저 확인

단톡에 알립니다.

> "서버 종료하려고 합니다. 사용 중인 사람 있나요?"

아래가 실행 중이면 **끄지 않습니다**:

- `/answer` 질문 테스트
- Gold60 평가
- Docker build
- Git 작업
- DB 작업

앱 서버에서 한 번에 확인:

```bash
cd /srv/festival/app

echo "=== GIT ==="
git branch --show-current
git status --short
git rev-parse --short HEAD

echo "=== HOST API ==="
ps -ef | grep "app.api" | grep -v grep

echo "=== EVALUATION ==="
ps -ef | grep "evaluate_postgres_agent" | grep -v grep

echo "=== DOCKER ==="
docker ps
```

Gold60은 오래 걸립니다. **돌고 있으면 끝날 때까지 기다립니다.** 중간에 끊으면 결과가 못 쓰게 됩니다.

### 2. Git working tree를 깨끗하게 — untracked artifact는 먼저 보존

기대 상태는 `branch = dev`, `git status --short` 출력 없음입니다.

**파일이 나온다고 바로 지우면 안 됩니다.** 특히:

```
?? *.json          ← 진단·평가 결과일 수 있습니다
?? 기타 실험 결과
```

repo 밖으로 **먼저 옮긴 뒤** 종료합니다:

```bash
mkdir -p /srv/festival/local-artifacts/<YYYY-MM-DD>
mv <보존할-파일> /srv/festival/local-artifacts/<YYYY-MM-DD>/

git status --short          # 다시 확인
```

`/srv/festival/local-artifacts/`는 git 밖이라 다음 `pull`에 영향을 주지 않고, 다음 날 그대로 남아 있습니다.

**절대 임의로 하지 않습니다:**

```bash
git reset --hard      # 금지
git clean -fd         # 금지
```

내 파일이 아닐 수 있고, 한 번 지우면 끝입니다.

### 3. 서버를 `dev` 상태로 돌려놓는다

**working tree가 깨끗할 때만** 실행합니다.

```bash
git fetch origin
git switch dev
git pull --ff-only origin dev
```

최종 확인 — `dev`이고 status가 비어 있어야 합니다:

```bash
git branch --show-current
git status --short
git rev-parse --short HEAD
```

이 해시를 단톡에 남깁니다. 다음 날 **"내가 끄기 전과 같은 코드인가"**를 1초에 확인할 수 있습니다.

### 4. 실행 중인 host API를 종료한다

```bash
ps -ef | grep "app.api" | grep -v grep
```

**자기가 확인한 PID만** 종료합니다:

```bash
kill <PID>
```

다시 확인해서 **출력이 없어야** 합니다:

```bash
ps -ef | grep "app.api" | grep -v grep
ss -tlnp | grep -E "8000|801"       # 포트도 반환됐는지
```

여러 개가 떠 있는 게 정상입니다 (사람마다 포트가 다르므로). 종료 담당자는 단톡에서
**전원이 반환을 확인한 뒤** 남은 것을 정리합니다.

### 5. 앱 서버 Docker API는 `stop`만 — 삭제하지 않습니다

```bash
docker ps
docker stop <API_CONTAINER_NAME>
```

**금지:**

```bash
docker rm ...                      # 금지
docker system prune --volumes      # 금지
docker compose down -v             # 금지
```

> **`docker ps`는 어느 서버에서 실행했는지가 중요합니다.**
> 앱 서버의 `docker ps`가 비어 있어도 **DB가 사라진 것이 아닙니다.**
> PostgreSQL은 **별도의 DB 서버**에서 돕니다. 앱 서버에서 안 보이는 게 정상입니다.
> 여기서 놀라서 DB를 "복구"하려 들면 그게 진짜 사고입니다.

### 6. 앱 서버를 끄기 전에 DB 상태를 확인한다

DB 서버는 앱 서버를 통해 접근하므로, **앱 서버를 정지하기 전에** 확인하는 것이 편합니다.

```bash
ssh root@<DB_SERVER_IP> 'docker ps'
```

정상:

```
festival-postgres
STATUS:  Up ...
IMAGE:   pgvector/pgvector:pg16
```

`festival-postgres`가 `Up`이면 됩니다. **서버를 끈다고 이 컨테이너를 직접 건드리지 않습니다:**

```bash
docker stop festival-postgres      # 하지 않습니다
docker rm festival-postgres        # 절대 금지
```

DB VM 자체를 NCP에서 정상 정지하면 컨테이너도 같이 멈춥니다.

### 7. 마지막 디스크 flush

```bash
sync
exit
```

### 8. NCP에서 정지 — 앱 서버 먼저

```
1. 앱 서버 정지
2. 상태가 완전히 "정지"가 된 것 확인
3. DB 서버 정지
```

버튼은 **`정지`만** 사용합니다.

| | |
|---|---|
| ✅ | **정지** |
| ❌ | 반납 / 삭제 |
| ❌ | Public IP 반납 |
| ❌ | Block Storage 삭제 |
| ❌ | NAT Gateway 삭제 |
| ❌ | VPC / Subnet / ACG 삭제 |

**버튼 이름을 확인하고 누르세요.** "정지"와 "반납"은 콘솔에서 가까이 붙어 있고,
반납은 되돌릴 방법이 없습니다 ([3번](#3-인프라-보호--지우면-못-되돌리는-것)).

### 9. 단톡에 상태 남기기

```
앱 서버 정지 완료
DB 서버 정지 완료 (또는: DB 서버는 켜 둠)
마지막 dev commit: <git rev-parse --short HEAD 결과>
작업 중인 프로세스 없음
보존한 artifact: /srv/festival/local-artifacts/<YYYY-MM-DD>/ (있다면)
```

---

### 다음 날 서버 다시 켜기 — 순서는 정확히 반대

### (1) DB 서버부터 시작

NCP에서 **DB 서버 → 시작**. `Running`이 될 때까지 기다립니다.

### (2) 앱 서버 시작

NCP에서 **앱 서버 → 시작**. 그 후 SSH 접속.

### (3) PostgreSQL 확인

앱 서버에서:

```bash
ssh root@<DB_SERVER_IP> 'docker ps'
```

`festival-postgres`가 `Up`이면 정상입니다.

**안 보이더라도 삭제하거나 새로 만들지 않습니다.** 먼저 멈춘 컨테이너를 확인하세요:

```bash
ssh root@<DB_SERVER_IP> 'docker ps -a'
```

`Exited` 상태라면 **기존 컨테이너를 시작**합니다 — 새로 만드는 게 아닙니다:

```bash
ssh root@<DB_SERVER_IP> 'docker start festival-postgres'
```

> `docker run`으로 새 컨테이너를 만들면 **volume 연결이 끊긴 빈 DB**가 뜰 수 있습니다.
> 데이터가 사라진 것처럼 보이는 사고의 대부분이 여기서 나옵니다. 반드시 `docker start`입니다.

### (4) 앱 서버 Git 상태 확인

```bash
cd /srv/festival/app
git status --short
git branch --show-current
git rev-parse --short HEAD
```

기대: `branch = dev`, `status = clean`. 어제 남긴 해시와 같은지 대조하세요.

```bash
git fetch origin
git pull --ff-only origin dev
```

### (5) 환경 로드

```bash
source .venv/bin/activate
set -a
source .env
set +a
```

### (6) 개발 API 시작

```bash
FESTIVAL_API_PORT=8010 \
FESTIVAL_HCX_ENABLED=false \
python -m app.api
```

백그라운드:

```bash
FESTIVAL_API_PORT=8010 FESTIVAL_HCX_ENABLED=false \
  nohup python -m app.api > /tmp/api-8010.log 2>&1 &
```

확인:

```bash
curl -s http://localhost:8010/healthz
```

> **`/healthz`만 보고 끝내지 마세요. 실제 질문 하나까지 확인합니다.**
> DB 연결은 요청 시점에 열리므로, **DB 서버가 안 켜져 있어도 API는 정상적으로 뜨고
> `/healthz`는 `ok`를 줍니다.** 질문만 `503 {"reason": "database_unavailable"}`로 떨어집니다.
> 이걸 코드 버그로 오해하고 디버깅하지 마세요 — DB 서버가 안 켜진 것입니다.

---

## 9. 브랜치 작업법 — `dev`에서 따고 PR로 `dev`에 합친다

기준 브랜치는 **`dev`** 입니다. `main`에는 직접 올리지 않습니다.

### (1) 작업 시작 — `dev`에서 최신 받기

```bash
git checkout dev
git fetch origin
git pull
```

오래된 코드에서 수정한 뒤 push하지 마세요. 충돌 해결에 드는 시간이 수정 시간보다 깁니다.

### (2) 개인 브랜치 생성

```bash
git checkout -b feature/<이름>-<주제>
```

예시:

```
feature/taeyoon-query-routing
feature/<이름>-periodic-fix
feature/<이름>-holding-fix
```

### (3) 커밋은 목적별로 작게

query routing 수정 + UI 수정 + retrieval 튜닝을 한 커밋에 섞지 마세요.
나중에 성능이 떨어졌을 때 **어느 변경이 범인인지** 못 찾습니다.

### (4) push 전 `git status` 확인

```bash
git status --short
```

의도치 않은 파일이 섞이지 않았는지 봅니다. 특히 **대용량 데이터, `docs/architecture/`, `.env`, DB export**.
`.gitignore`가 `data/processed/`, `data/exports/`, `data/db_export/`, `.env`, `.env.*`를 이미 막고 있지만,
새로 만든 파일은 직접 확인해야 합니다. `git add .` 대신 파일을 명시하는 습관이 안전합니다.

### (5) 검증 — [12번](#12-수정했으면-반드시-세-가지-확인) 3단계를 먼저 통과시킨다

**검증 안 된 브랜치는 `dev`에 올리지 않습니다.**

### (6) PR로 `dev`에 머지

```bash
git push -u origin feature/<이름>-<주제>
```

GitHub에서 **PR (`feature/… → dev`)** 를 만들고 팀원 확인 후 머지합니다.
무엇이 언제 왜 들어갔는지 기록이 남고, 문제가 생겼을 때 PR 단위로 되돌릴 수 있습니다.
지금까지의 변경도 전부 이 방식으로 들어갔습니다 (최신 multi-event 작업 = PR #21).

PR 본문에 이것만 적어 주면 충분합니다:

```
무엇을 고쳤나:
어떤 질문에서 발견했나:
수정 전 / 후 답변:
pytest: 통과 (failures 0)
Gold60: R@10 ____ / E2E ____ / __-60   (또는 "해당 없음 — 서버 검증 불필요")
```

`--no-ff` 머지가 기본이라 문제가 생기면 `git revert -m 1 <머지커밋>`으로 통째로 되돌릴 수 있습니다.
**머지 직후 `dev`에서 pytest를 한 번 더 돌리세요.** 각자 브랜치에선 통과했는데 합치면 깨지는 경우가 있습니다.

### (7) 서버 반영

서버는 `dev`를 봅니다. 서버에서는 **읽기만** 합니다.

```bash
ssh root@<APP_SERVER_IP>
cd /srv/festival/app
git status --short              # 깨끗한지 확인. 아니면 중단
git branch --show-current       # dev 확인
git fetch origin
git pull --ff-only origin dev
# → API 재시작 (7-C)
```

---

## 10. 질문 하나 테스트하기

```bash
curl -sG http://localhost:8010/answer \
  --data-urlencode 'question_id=T01' \
  --data-urlencode 'question=하이브가 에스엠 지분을 처분한 뒤 보유 주식 수는?' | jq
```

응답 최상위는 **정확히 5개 필드**입니다 ([2번](#2-api-계약-freeze)):
`question_id`, `question`, `retrieved_context`, `think_trace`, `answer`.

### 먼저 볼 것은 답변이 아니라 `think_trace`

```bash
curl -sG http://localhost:8010/answer \
  --data-urlencode 'question_id=T01' \
  --data-urlencode 'question=<질문>' \
  | jq '{answer, trace: (.think_trace | {task_type, route, answerable, warnings, hcx_status, retrieval_count, selected_evidence_count})}'
```

Top-10에 정답이 들어왔는지 확인:

```bash
curl -sG http://localhost:8010/answer \
  --data-urlencode 'question_id=T01' \
  --data-urlencode 'question=<질문>' \
  | jq '.retrieved_context[] | {rank, corp_name, report_nm, rcept_dt, chunk_type}'
```

주요 값:

- `task_type` — `holding_event` / `periodic_fact` / `general_evidence` / `unknown`
- `route` — 실제 사용된 resolver (없으면 `general_evidence`)
- `hcx_status` — `disabled`(개발 기본) / `success` / `skipped_*` / `fallback_*` / `not_configured`

`hcx_status`가 `fallback_*`이어도 **버그가 아닙니다.** 검증에 실패한 모델 출력을 버리고
결정적 답변을 서빙한 정상 동작입니다.

`answerable`이 `false`인 것도, 질문이 corpus 범위 밖이라면 **정답 동작입니다** ([1번](#1-대회-규칙--외부-데이터-금지)).

---

## 11. 답이 이상할 때 — retrieval부터 고치지 않는다

**먼저 어디서 실패했는지 분류합니다.**

| 증상 | 실제 원인 위치 | 확인 방법 |
|---|---|---|
| 회사를 잘못 인식 | **company resolver** | `retrieved_context[].corp_name`이 엉뚱한 회사 |
| 공시 유형 판단이 틀림 | **Query Understanding / Router** | `think_trace.task_type`, `route` |
| 정답이 Top-10에 아예 없음 | **Retrieval** | `retrieved_context` 전수 확인 |
| Top-10에는 있는데 오답 | **Resolver / Composer** | `answerable`, `warnings`, `selected_evidence_count` |
| 사실은 맞는데 보기가 안 좋음 | **Answer Generator / Presentation** | `answer` 문자열만 문제 |

마지막 두 줄이 지금 가장 흔합니다. **retrieval은 이미 R@10=0.90이고 남은 6건은 전부
retrieval miss로 확인된 상태**라, 새 질문이 틀렸을 때 첫 용의자가 retrieval인 경우는 드뭅니다.

**그리고 corpus에 애초에 없는 내용이면 어느 레이어도 고칠 대상이 아닙니다.**
외부에서 가져오지 말고 근거 부족으로 두세요 ([1번](#1-대회-규칙--외부-데이터-금지)).

수정 위치 지도:

| 레이어 | 파일 |
|---|---|
| Query Understanding / Router | `app/reasoning/query_understanding.py`, `app/reasoning/router.py`, `app/agent/task_router.py` |
| Retrieval | `app/retrieval/` |
| Resolver | `app/reasoning/holding_event_resolver.py`, `app/reasoning/periodic_fact_resolver.py` |
| Composer | `app/reasoning/answer_composer.py` |
| Presentation | `app/generation/answer_generator.py` |
| HCX 문장화 | `app/generation/hcx_verbalizer.py`, `app/generation/lossless_verbalization.py` |

---

## 12. 수정했으면 반드시 세 가지 확인

### (1) 해당 질문이 실제로 개선됐는가

수정 전/후 `answer`와 `think_trace`를 둘 다 남깁니다.

### (2) pytest 전체 통과

```bash
python -m pytest -q
```

**성공 조건은 `failures 0`입니다.** 테스트 개수는 지금 약 793건이지만 수정할 때마다 늘어납니다.
숫자가 달라졌다고 이상한 게 아니고, **실패가 하나라도 있으면** 이상한 것입니다.

일부만 돌리지 말고 전체를 돌립니다. presentation 수정이 resolver 테스트를 깨는 경우가 실제로 있었습니다.

**새 수정에는 새 테스트를 같이 넣습니다.** 그리고 그 테스트는 **일반 규칙**을 검증해야 합니다.
"이 회사 이 날짜면 이렇게"가 아니라 "이런 형태의 입력이면 이렇게"여야 합니다.

### (3) Gold60 회귀 (앱 서버에서, 한 번에 한 명만)

> ### Gold60은 시험문제가 아닙니다
>
> Gold60은 **기존 기능이 깨지지 않았는지 확인하는 내부 regression set**입니다.
> 공식 대회 평가 결과가 아니고, 개발 목표도 아닙니다.
>
> - **54/60을 올리는 것이 목표가 아닙니다.** 60/60을 만들어도 그게 실제 성능을 뜻하지 않습니다.
> - **특정 문항·회사·값에 맞춘 튜닝 금지.** Gold60 문항을 보고 코드를 맞추면
>   그 60문제만 잘 푸는 시스템이 됩니다. 실제 채점 질문은 다릅니다.
> - **새 수정의 정당화는 "실제로 발견한 질문 + 일반화된 단위 테스트"입니다.**
>   Gold60은 그다음에, **마지막 회귀 확인용으로만** 씁니다.
> - **점수가 올랐다는 이유만으로 머지하지 않습니다.** 왜 올랐는지 설명할 수 있어야 합니다.
>   설명이 안 되면 과적합이거나 우연입니다.

실행 전에 **단톡에 알립니다.** DB·임베딩 부하가 커서 동시에 두 명이 돌리면
서로 느려지고 결과 비교도 어려워집니다. **한 번에 한 명만.**

```bash
# 서버 .env 그대로. 환경변수를 앞에 붙이지 않습니다 (baseline과 동일 조건 유지)
python scripts/evaluate_postgres_agent_gold60.py \
  --output-dir data/processed/postgres_agent_gold60_<이름>_<주제>
```

**현재 기준선 — 이보다 떨어지면 `dev`에 머지하지 않는다:**

| 지표 | 기준선 |
|---|---|
| Recall@1 | 0.483333 |
| Recall@5 | 0.783333 |
| Recall@10 | **0.900000** |
| `answerable_rate` | 1.000000 |
| `gold_doc_citation_rate` | 0.916667 |
| `gold_chunk_citation_rate` | 0.883333 |
| `all_evidence_terms_rate` | 0.950000 |
| `end_to_end_success_rate` | **0.900000** |
| 성공 | **54 / 60** |

**E2E만 보지 마세요.** E2E는 그대로인데 아래 지표만 조용히 떨어지는 경우가 있습니다.

| 떨어진 지표 | 대체로 이런 뜻 |
|---|---|
| R@1 / R@5 / R@10 | retrieval을 안 건드렸다면 **완전히 동일해야** 합니다. 바뀌었다면 의도치 않게 검색 경로를 건드린 것 |
| `answerable_rate` | 답할 수 있는 질문을 근거 부족으로 판정하기 시작한 것 |
| `gold_doc_citation_rate` | 맞는 답인데 **다른 문서를 인용**하고 있는 것 |
| `gold_chunk_citation_rate` | 문서는 맞는데 **chunk 단위 인용이 어긋난** 것. E2E로는 안 보입니다 |
| `all_evidence_terms_rate` | 십중팔구 **검증된 값을 답변에서 지운** 것 (0번 표 참조) |

`--output-dir`은 항상 새 이름으로. 기존 보존 artifact(`reports/evaluation/gold60/2026-08-21-agent-90pct/`)는
덮어쓰지 않습니다.

보존된 평가 artifact 무결성 확인:

```bash
python scripts/preserve_evaluation_artifact.py --verify
```

---

## 13. 문제 질문은 기록해 둔다

Gold60에 맞추는 대신 **우리가 실제로 발견한 질문**을 모읍니다. 이게 나중에 DEV/HOLDOUT 평가셋이 됩니다.
이슈나 공용 문서에 아래 형식으로 남겨 주세요.

```
질문:
question_id:
task_type / route:
warnings:
실제 답변:
기대 답변:
분류(11번 표 기준):
corpus 범위 안인가:
수정 전 결과:
수정 후 결과:
관련 브랜치 / PR:
```

`corpus 범위 안인가`가 "아니오"면 **고칠 대상이 아닙니다.** 그것도 기록해 두면
같은 질문으로 두 번 헤매지 않습니다.

**공시 Q&A 카드 초안**은 에이전트와 분리된 [qa-tool/README.md](qa-tool/README.md)를 사용합니다. (`origin/taeyoon`에서 `feat/qa-tool-curation` 등 별도 브랜치)

---

## 14. 복구용 안정 태그 (이동·덮어쓰기 금지)

| 태그 | 커밋 | 내용 |
|---|---|---|
| `agent-gold60-90pct-2026-08-21` | `d8a5f74` | Gold60 54/60을 만든 코드 지점 |
| `api-hcx-lossless-safe-2026-08-21` | `5c1042e` | API + HCX 안전 계층 |
| `submission-ready-v1-2026-08-21` | `019045b` | 제출 가능 v1 |
| `holding-scope-tuning-v2-2026-08-21` | `5f46e94` | holding event 범위 좁히기 |
| `presentation-ux-v1-2026-08-21` | `4f6a0b6` | 단일 event 자연문 렌더링 |
| `multi-event-presentation-v1-2026-08-22` | `a8e2b38` | 다중 event 압축 렌더링 |

무언가 크게 잘못되면 여기로 돌아옵니다:

```bash
git checkout -b recover/<이름> <태그명>
```

---

## 15. 30초 체크리스트

**규칙**

- [ ] 외부 데이터(OpenDART·뉴스·검색)를 붙이지 않았다
- [ ] `GET /answer` 최상위 5개 필드를 건드리지 않았다
- [ ] NCP 리소스·Docker volume·공용 `.venv`를 건드리지 않았다

**로컬에서**

- [ ] `git checkout dev && git fetch origin && git pull` 했다
- [ ] `feature/<이름>-<주제>` 개인 브랜치를 만들었다
- [ ] 실패 원인을 [11번](#11-답이-이상할-때--retrieval부터-고치지-않는다) 표로 **먼저 분류**했다
- [ ] 회사명·날짜·question_id를 하드코딩하지 않았다
- [ ] 검증된 값을 지우지 않고 표현만 압축했다
- [ ] 새 수정에 **일반 규칙을 검증하는** 테스트를 같이 넣었다
- [ ] `python -m pytest -q` — **failures 0**
- [ ] `git status --short`에 `.env`·대용량 파일이 없다
- [ ] 커밋이 목적 하나만 담고 있다

**서버에서**

- [ ] `git status --short`가 깨끗하고 `git branch --show-current`가 **`dev`** 다
- [ ] `FESTIVAL_HCX_ENABLED=false`로 띄웠다 (HCX 테스트가 아니라면)
- [ ] 포트를 단톡 점유표에 올렸다 / 끝나고 반환했다
- [ ] 개발 포트를 ACG에 열지 않았다 (브라우저로 볼 땐 SSH 터널)
- [ ] Gold60을 돌린다면 **단톡에 알리고 혼자** 돌렸다
- [ ] Gold60 8개 지표 전부 기준선 이상 (E2E만 보지 않았다)
- [ ] 코드 갱신 후 **API를 재시작**했다 (Docker면 `build`까지)

**머지**

- [ ] PR로 올렸다 (`feature/… → dev`)
- [ ] 머지 후 `dev`에서 pytest를 한 번 더 돌렸다

**서버 끄기 전** ([8번](#8-서버-종료--재기동--하루-작업-마감-절차))

- [ ] 다른 사람이 API / Gold60을 사용 중이지 않다
- [ ] `dev` + clean working tree 상태다
- [ ] untracked 실험 파일을 확인·보존했다 (`/srv/festival/local-artifacts/<날짜>/`)
- [ ] host API를 종료했다 (`ps -ef | grep app.api` 출력 없음)
- [ ] 앱 Docker API를 `stop`했다 (`rm` 아님)
- [ ] DB 서버의 `festival-postgres`가 `Up`인지 확인했다
- [ ] `sync` 했다
- [ ] **앱 서버 → DB 서버** 순서로 "정지"했다 (반납·삭제 아님)
- [ ] 단톡에 마지막 `dev` commit 해시를 남겼다
