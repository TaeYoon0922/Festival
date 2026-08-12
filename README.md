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

## 구조 보존 청킹

파서가 복원한 문서 계층과 표 구조를 유지한 채 검색용 chunk를 생성합니다.

- 본문: `section → paragraph → sentence → chunk`
- 표: 병합 셀을 논리 그리드로 복원한 뒤 `header → row group`
- 공시 유형: `periodic`, `major/exchange`, `holding`별 독립 전략
- 검색 문맥: 원문 `content`와 별도로 기업·공시·section 경로를 포함한
  `retrieval_text` 생성
- 추적성: 문서/section/table metadata, 결정적 chunk ID, 앞뒤 chunk 링크 보존

표 행은 문자 길이 때문에 분할하지 않습니다. 작은 이벤트·key-value 표는 통째로
유지하고, 큰 표만 행 묶음으로 나누며 모든 묶음에 동일한 header와 표 context를
반복합니다.

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

기존 고정 길이 청커와 구조 청커를 동일한 20건에서 비교하려면 다음 명령을
사용합니다.

```bash
python scripts/compare_chunking_pilot.py
```

결과는 `data/processed/chunking_pilot_20`에 생성됩니다.

- `comparison.json`: 전체·공시 유형별 정량 지표와 검토 표본
- `comparison.md`: 사람이 읽을 수 있는 비교표와 대표 chunk
- `documents/*.json`: 구조 청커로 생성한 문서별 schema 2.0 결과

20건 비교 결과를 검토하기 전에는 전체 코퍼스를 다시 처리하지 않습니다.

전체 재처리 전 최종 freeze gate는 저장된 20건 pilot에 대한 길이·제외 표 audit와
Legacy/Structural BM25 비교, 유형별 정정공시 3건씩의 추가 검증만 수행합니다.

```bash
python scripts/run_final_chunking_validation.py
```

결과는 `data/processed/chunking_final_validation`에 생성됩니다. 이 명령은 전체
4,204개 문서를 재처리하지 않습니다.

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
