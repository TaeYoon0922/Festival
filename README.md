# Disclosure Agent

DART 공시 코퍼스를 파싱·검색·분석해 답변과 보고서를 생성하기 위한 Python 프로젝트입니다.

## 구조

```text
.
├── app/
│   ├── api/          # API 계층
│   ├── parsing/      # DART XML 파싱
│   ├── retrieval/    # 검색·인덱싱
│   ├── reasoning/    # 분석·추론
│   └── generation/   # 답변·보고서 생성
├── data/corpus/      # 메타데이터와 로컬 raw 데이터 연결
├── tests/
├── scripts/
├── requirements.txt
└── Dockerfile
```

## 데이터 연결

대용량 원문은 Git에 올리지 않습니다. 기본 경로는 `data/corpus`이며, 이 저장소에서는
`data/corpus/raw`가 로컬에 압축 해제된 공시 원문을 가리킵니다.

다른 위치의 데이터를 사용하려면 환경변수를 지정합니다.

```bash
export DISCLOSURE_DATA_DIR=/absolute/path/to/corpus
python -m app
```

연결 상태는 다음 명령으로 확인합니다.

```bash
python -m app
python -m unittest discover
```

## 20건 파싱 파일럿

Vector DB나 외부 LLM을 사용하지 않고 다음 단계만 수행합니다.

```text
raw XML → Section / Table → Chunk
```

정정공시를 제외한 XML 중 정기·거래소·주요사항·지분공시를 각각 5건씩
선정합니다. 정기공시 표본에는 삼성전자 사업보고서가 포함됩니다.

```bash
python scripts/parse_sample.py
python scripts/validate_sample.py
```

결과는 `data/processed/sample_20`에 생성됩니다.

- `selection.json`: 선정된 20개 문서
- `summary.json`: 유형별 문서·섹션·표·청크 통계
- `documents/*.json`: 문서별 Section, Table, Chunk 결과

생성 결과는 재생성 가능하므로 Git에서 제외합니다.

## 전체 코퍼스 파싱

파일럿 검증 후 전체 4,204개 문서의 XML 4,616개와 대체 HTML 3개를 처리합니다.
정정공시와 정기공시 첨부 XML도 포함합니다. 결과는 재시작 가능한 압축 JSON으로
`data/processed/full`에 저장됩니다.

```bash
python scripts/parse_all.py --workers 4
python scripts/validate_all.py
```

- `source_plan.jsonl`: 전체 입력 파일 계획
- `index.jsonl`: 파일별 파싱 통계와 출력 경로
- `summary.json`: 전체 문서·섹션·표·청크 통계
- `documents/<유형>/<문서 ID>/*.json.gz`: XML/HTML 파일별 파싱 결과

## Docker

공시 데이터는 이미지에 포함하지 않고 읽기 전용 볼륨으로 연결합니다.

```bash
docker build -t disclosure-agent .
docker run --rm \
  -v "/absolute/path/to/corpus/raw:/app/data/corpus/raw:ro" \
  disclosure-agent
```

`raw/`, ZIP 파일, 로컬 환경설정은 `.gitignore` 및 `.dockerignore`에서 제외됩니다.
