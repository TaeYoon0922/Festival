\set ON_ERROR_STOP on

-- Run only after all six COPY files have been loaded.
BEGIN;

CREATE INDEX IF NOT EXISTS idx_companies_stock_code
    ON companies (stock_code);
CREATE INDEX IF NOT EXISTS idx_companies_name
    ON companies (corp_name);

CREATE INDEX IF NOT EXISTS idx_disclosures_company_date
    ON disclosures (corp_code, rcept_dt DESC);
CREATE INDEX IF NOT EXISTS idx_disclosures_company_period
    ON disclosures (corp_code, base_year, base_month);
CREATE INDEX IF NOT EXISTS idx_disclosures_group_subtype
    ON disclosures (doc_group, doc_subtype);
CREATE INDEX IF NOT EXISTS idx_disclosures_correction
    ON disclosures (is_correction);

CREATE INDEX IF NOT EXISTS idx_sections_doc_order
    ON sections (doc_id, source_part_id, section_order);
CREATE INDEX IF NOT EXISTS idx_sections_parent
    ON sections (source_part_id, parent_section_id)
    WHERE parent_section_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sections_path
    ON sections USING gin (section_path jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_tables_doc_order
    ON disclosure_tables (doc_id, source_part_id, table_order);
CREATE INDEX IF NOT EXISTS idx_tables_section
    ON disclosure_tables (source_part_id, section_id);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_order
    ON chunks (doc_id, chunk_order);
CREATE INDEX IF NOT EXISTS idx_chunks_section
    ON chunks (source_part_id, section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_table
    ON chunks (source_part_id, table_id)
    WHERE table_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_type_priority
    ON chunks (chunk_type, retrieval_priority);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_retrieval_fts
    ON chunks USING gin (to_tsvector('simple', retrieval_text));

CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_chunk
    ON chunk_source_refs (chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_table_rows
    ON chunk_source_refs (source_part_id, table_id, row_start, row_end);
CREATE INDEX IF NOT EXISTS idx_chunk_source_refs_field
    ON chunk_source_refs (field_name)
    WHERE field_name IS NOT NULL;

COMMIT;
