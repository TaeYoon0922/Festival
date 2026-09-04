# Festival architecture

저장소의 현재 구현(`dev`)을 기준으로 작성했습니다.

| 문서 | 내용 |
|---|---|
| [SERVING_ARCHITECTURE.md](SERVING_ARCHITECTURE.md) | 설계 기준, 평면 · 관문 · 레인 · 방어선 4개 관점, 실행 단계 어휘, 저장 스키마, 검색 구성, 응답 계약 |

## Source of truth

- API와 실행 흐름: `app/api`, `app/reasoning`, `app/retrieval`, `app/agent`, `app/generation`
- PostgreSQL 스키마: `db/001_schema.sql`, `db/004_vector_search.sql`,
  `db/005_table_chunk_provenance_backfill.sql`, `db/006_correction_graph.sql`,
  `db/007_corporate_event_timeline.sql`
- 동결 이력과 재개봉 정책: [`docs/FREEZE_LOG.md`](../FREEZE_LOG.md)

`QueryPlan`, `HybridExecution`, `EvidenceSet`, resolver output, `AnswerDraft`,
`GeneratedAnswer`는 immutable runtime object이며 PostgreSQL 테이블이 아닙니다. 스키마
문서에는 실제 영속 테이블만 표시합니다.
