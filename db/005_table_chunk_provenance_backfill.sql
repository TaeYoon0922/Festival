\set ON_ERROR_STOP on

-- Metadata-only backfill for ordinary table chunks. This migration never updates
-- chunk identity, content, retrieval_text, ordering, or embeddings.
BEGIN;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM chunks c
        WHERE c.chunk_type = 'table'
          AND (
              NOT (c.metadata ? 'source_refs')
              OR c.metadata -> 'source_refs' IS NULL
              OR c.metadata -> 'source_refs' = '[]'::jsonb
          )
          AND NOT EXISTS (
              SELECT 1
              FROM chunk_source_refs r
              WHERE r.chunk_id = c.chunk_id
                AND r.source_part_id = c.source_part_id
                AND r.table_id = c.table_id
                AND r.row_start = (c.metadata ->> 'row_start')::integer
                AND r.row_end = (c.metadata ->> 'row_end')::integer
          )
    ) THEN
        RAISE EXCEPTION
            'table provenance backfill aborted: a target chunk has no matching normalized source reference';
    END IF;
END
$$;

WITH canonical_refs AS (
    SELECT DISTINCT ON (c.chunk_id)
        c.chunk_id,
        c.table_id,
        r.row_start,
        r.row_end
    FROM chunks c
    JOIN chunk_source_refs r
      ON r.chunk_id = c.chunk_id
     AND r.source_part_id = c.source_part_id
     AND r.table_id = c.table_id
     AND r.row_start = (c.metadata ->> 'row_start')::integer
     AND r.row_end = (c.metadata ->> 'row_end')::integer
    WHERE c.chunk_type = 'table'
      AND (
          NOT (c.metadata ? 'source_refs')
          OR c.metadata -> 'source_refs' IS NULL
          OR c.metadata -> 'source_refs' = '[]'::jsonb
      )
    ORDER BY
        c.chunk_id,
        CASE r.source_type WHEN 'table_chunk_range' THEN 0 ELSE 1 END,
        r.source_ref_id
),
updated AS (
    UPDATE chunks c
    SET metadata = jsonb_set(
        jsonb_set(
            jsonb_set(
                c.metadata,
                '{source_table_id}',
                to_jsonb(canonical_refs.table_id),
                true
            ),
            '{source_table_ids}',
            jsonb_build_array(canonical_refs.table_id),
            true
        ),
        '{source_refs}',
        jsonb_build_array(
            jsonb_build_object(
                'table_id', canonical_refs.table_id,
                'row_start', canonical_refs.row_start,
                'row_end', canonical_refs.row_end
            )
        ),
        true
    )
    FROM canonical_refs
    WHERE c.chunk_id = canonical_refs.chunk_id
    RETURNING c.chunk_id
)
SELECT count(*) AS updated_table_chunk_count FROM updated;

COMMIT;

SELECT count(*) AS table_chunks_still_missing_source_refs
FROM chunks
WHERE chunk_type = 'table'
  AND (
      NOT (metadata ? 'source_refs')
      OR metadata -> 'source_refs' IS NULL
      OR metadata -> 'source_refs' = '[]'::jsonb
  );

