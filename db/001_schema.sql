\set ON_ERROR_STOP on

-- PostgreSQL 16 / pgvector 0.8.6. No embedding column or data is created here.
BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    corp_code text PRIMARY KEY,
    stock_code text NOT NULL,
    corp_name text NOT NULL,
    listed_name text,
    market text,
    industry text,
    sector text,
    metadata jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS disclosures (
    doc_id text PRIMARY KEY,
    corp_code text NOT NULL REFERENCES companies(corp_code),
    rcept_no text NOT NULL,
    report_nm text NOT NULL,
    rcept_dt date,
    doc_group text NOT NULL,
    doc_subtype text,
    is_correction boolean NOT NULL,
    base_year integer,
    base_month integer,
    file_path text NOT NULL,
    file_format text NOT NULL,
    metadata jsonb NOT NULL
);

-- section_id is local to a source part in the frozen parser output.
CREATE TABLE IF NOT EXISTS sections (
    source_part_id text NOT NULL,
    section_id text NOT NULL,
    doc_id text NOT NULL REFERENCES disclosures(doc_id),
    parent_section_id text,
    section_title text NOT NULL,
    section_path jsonb NOT NULL,
    section_order integer NOT NULL,
    section_depth integer NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL,
    PRIMARY KEY (source_part_id, section_id),
    FOREIGN KEY (source_part_id, parent_section_id)
        REFERENCES sections(source_part_id, section_id)
);

-- table_id is also local to a source part and is preserved without rewriting.
CREATE TABLE IF NOT EXISTS disclosure_tables (
    source_part_id text NOT NULL,
    table_id text NOT NULL,
    doc_id text NOT NULL REFERENCES disclosures(doc_id),
    section_id text NOT NULL,
    table_title text,
    table_order integer NOT NULL,
    source_path text NOT NULL,
    row_count integer NOT NULL,
    attributes jsonb NOT NULL,
    table_rows jsonb NOT NULL,
    metadata jsonb NOT NULL,
    PRIMARY KEY (source_part_id, table_id),
    FOREIGN KEY (source_part_id, section_id)
        REFERENCES sections(source_part_id, section_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY,
    doc_id text NOT NULL REFERENCES disclosures(doc_id),
    source_part_id text NOT NULL,
    section_id text NOT NULL,
    table_id text,
    chunk_type text NOT NULL,
    chunk_order integer NOT NULL,
    content text NOT NULL,
    retrieval_text text NOT NULL,
    char_count integer NOT NULL,
    retrieval_priority text,
    metadata jsonb NOT NULL,
    FOREIGN KEY (source_part_id, section_id)
        REFERENCES sections(source_part_id, section_id),
    FOREIGN KEY (source_part_id, table_id)
        REFERENCES disclosure_tables(source_part_id, table_id)
);

CREATE TABLE IF NOT EXISTS chunk_source_refs (
    source_ref_id text PRIMARY KEY,
    chunk_id text NOT NULL REFERENCES chunks(chunk_id),
    source_type text NOT NULL,
    source_part_id text NOT NULL,
    table_id text NOT NULL,
    row_start integer NOT NULL,
    row_end integer NOT NULL,
    field_name text,
    source_ref jsonb NOT NULL,
    metadata jsonb NOT NULL,
    FOREIGN KEY (source_part_id, table_id)
        REFERENCES disclosure_tables(source_part_id, table_id)
);

COMMIT;
