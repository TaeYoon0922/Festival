\set ON_ERROR_STOP on

SELECT 'companies' AS entity, count(*) AS row_count FROM companies
UNION ALL SELECT 'disclosures', count(*) FROM disclosures
UNION ALL SELECT 'sections', count(*) FROM sections
UNION ALL SELECT 'disclosure_tables', count(*) FROM disclosure_tables
UNION ALL SELECT 'chunks', count(*) FROM chunks
UNION ALL SELECT 'chunk_source_refs', count(*) FROM chunk_source_refs
ORDER BY entity;

WITH validation AS (
    SELECT 'orphan_disclosure_company' AS check_name, count(*) AS error_count
    FROM disclosures d LEFT JOIN companies c USING (corp_code)
    WHERE c.corp_code IS NULL
    UNION ALL
    SELECT 'orphan_section_disclosure', count(*)
    FROM sections s LEFT JOIN disclosures d USING (doc_id)
    WHERE d.doc_id IS NULL
    UNION ALL
    SELECT 'orphan_section_parent', count(*)
    FROM sections s
    LEFT JOIN sections p
      ON p.source_part_id = s.source_part_id
     AND p.section_id = s.parent_section_id
    WHERE s.parent_section_id IS NOT NULL AND p.section_id IS NULL
    UNION ALL
    SELECT 'orphan_table_disclosure', count(*)
    FROM disclosure_tables t LEFT JOIN disclosures d USING (doc_id)
    WHERE d.doc_id IS NULL
    UNION ALL
    SELECT 'orphan_table_section', count(*)
    FROM disclosure_tables t
    LEFT JOIN sections s
      ON s.source_part_id = t.source_part_id AND s.section_id = t.section_id
    WHERE s.section_id IS NULL
    UNION ALL
    SELECT 'orphan_chunk_disclosure', count(*)
    FROM chunks c LEFT JOIN disclosures d USING (doc_id)
    WHERE d.doc_id IS NULL
    UNION ALL
    SELECT 'orphan_chunk_section', count(*)
    FROM chunks c
    LEFT JOIN sections s
      ON s.source_part_id = c.source_part_id AND s.section_id = c.section_id
    WHERE s.section_id IS NULL
    UNION ALL
    SELECT 'orphan_chunk_table', count(*)
    FROM chunks c
    LEFT JOIN disclosure_tables t
      ON t.source_part_id = c.source_part_id AND t.table_id = c.table_id
    WHERE c.table_id IS NOT NULL AND t.table_id IS NULL
    UNION ALL
    SELECT 'orphan_source_ref_chunk', count(*)
    FROM chunk_source_refs r LEFT JOIN chunks c USING (chunk_id)
    WHERE c.chunk_id IS NULL
    UNION ALL
    SELECT 'orphan_source_ref_table', count(*)
    FROM chunk_source_refs r
    LEFT JOIN disclosure_tables t
      ON t.source_part_id = r.source_part_id AND t.table_id = r.table_id
    WHERE t.table_id IS NULL
    UNION ALL
    SELECT 'invalid_source_ref_range', count(*)
    FROM chunk_source_refs r
    JOIN disclosure_tables t
      ON t.source_part_id = r.source_part_id AND t.table_id = r.table_id
    WHERE r.row_start < 0
       OR r.row_end < r.row_start
       OR r.row_end >= t.row_count
    UNION ALL
    SELECT 'table_or_projection_without_source_ref', count(*)
    FROM chunks c
    WHERE c.chunk_type IN ('table', 'table_projection')
      AND NOT EXISTS (
          SELECT 1 FROM chunk_source_refs r WHERE r.chunk_id = c.chunk_id
      )
    UNION ALL
    SELECT 'projection_field_without_source_ref', count(*)
    FROM chunks c
    CROSS JOIN LATERAL
        jsonb_object_keys(COALESCE(c.metadata->'projection_fields', '{}'::jsonb)) f(field_name)
    WHERE c.chunk_type = 'table_projection'
      AND NOT EXISTS (
          SELECT 1
          FROM chunk_source_refs r
          WHERE r.chunk_id = c.chunk_id AND r.field_name = f.field_name
      )
    UNION ALL
    SELECT 'required_chunk_metadata_missing', count(*)
    FROM chunks
    WHERE chunk_id IS NULL OR doc_id IS NULL OR source_part_id IS NULL
       OR section_id IS NULL OR chunk_type IS NULL OR chunk_order IS NULL
       OR content IS NULL OR retrieval_text IS NULL OR char_count IS NULL
       OR metadata IS NULL
)
SELECT * FROM validation ORDER BY check_name;

DO $$
DECLARE
    extension_version text;
BEGIN
    SELECT extversion INTO extension_version FROM pg_extension WHERE extname = 'vector';
    IF extension_version IS NULL THEN
        RAISE EXCEPTION 'pgvector extension is not installed';
    END IF;
    IF extension_version <> '0.8.6' THEN
        RAISE EXCEPTION 'expected pgvector 0.8.6, found %', extension_version;
    END IF;
    IF (SELECT count(*) FROM companies) <> 70 THEN
        RAISE EXCEPTION 'companies count mismatch';
    END IF;
    IF (SELECT count(*) FROM disclosures) <> 4204 THEN
        RAISE EXCEPTION 'disclosures count mismatch';
    END IF;
    IF (SELECT count(*) FROM sections) <> 147399 THEN
        RAISE EXCEPTION 'sections count mismatch';
    END IF;
    IF (SELECT count(*) FROM disclosure_tables) <> 1216982 THEN
        RAISE EXCEPTION 'disclosure_tables count mismatch';
    END IF;
    IF (SELECT count(*) FROM chunks) <> 1363336 THEN
        RAISE EXCEPTION 'chunks count mismatch';
    END IF;
    IF (SELECT count(*) FROM chunks WHERE chunk_type = 'table_projection') <> 62243 THEN
        RAISE EXCEPTION 'projection count mismatch';
    END IF;
    IF (SELECT count(*) FROM chunk_source_refs) <> 1385444 THEN
        RAISE EXCEPTION 'chunk_source_refs count mismatch';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM chunk_source_refs r
        JOIN disclosure_tables t
          ON t.source_part_id = r.source_part_id AND t.table_id = r.table_id
        WHERE r.row_start < 0 OR r.row_end < r.row_start OR r.row_end >= t.row_count
    ) THEN
        RAISE EXCEPTION 'invalid source reference row range';
    END IF;
    IF EXISTS (
        SELECT 1 FROM chunks c
        WHERE c.chunk_type IN ('table', 'table_projection')
          AND NOT EXISTS (SELECT 1 FROM chunk_source_refs r WHERE r.chunk_id = c.chunk_id)
    ) THEN
        RAISE EXCEPTION 'missing table/projection source reference';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM chunks c
        CROSS JOIN LATERAL
            jsonb_object_keys(COALESCE(c.metadata->'projection_fields', '{}'::jsonb)) f(field_name)
        WHERE c.chunk_type = 'table_projection'
          AND NOT EXISTS (
              SELECT 1 FROM chunk_source_refs r
              WHERE r.chunk_id = c.chunk_id AND r.field_name = f.field_name
          )
    ) THEN
        RAISE EXCEPTION 'missing field-level projection source reference';
    END IF;
END $$;
