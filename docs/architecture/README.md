# Festival architecture diagrams (2026.08)

이 디렉터리의 다이어그램은 현재 저장소 구현을 기준으로 작성했습니다.

- `festival-agent-architecture-2026-08.svg`: 편집 가능한 최신 Agent 구조도
- `festival-agent-architecture-2026-08.png`: 공유·발표용 PNG
- `festival-postgres-erd-2026-08.svg`: 편집 가능한 PostgreSQL ERD
- `festival-postgres-erd-2026-08.png`: 공유·발표용 PNG

## Source of truth

- API와 실행 흐름: `app/api`, `app/query_understanding`, `app/retrieval`, `app/agent`, `app/reasoning`, `app/generation`
- PostgreSQL 스키마: `db/001_schema.sql`, `db/004_vector_search.sql`, `db/005_table_chunk_provenance_backfill.sql`

`QueryPlan`, `HybridExecution`, `EvidenceSet`, resolver output, `AnswerDraft`, `GeneratedAnswer`는 immutable runtime object이며 PostgreSQL 테이블이 아닙니다. ERD에는 실제 영속 테이블만 표시했습니다.

