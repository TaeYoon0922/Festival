\set ON_ERROR_STOP on

-- Embeddings are intentionally separated from frozen chunks. Model/version rows
-- can be replaced without rewriting chunk content, IDs, or provenance.
BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id text NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding_model text NOT NULL,
    embedding_version text NOT NULL,
    embedding_dimensions smallint NOT NULL,
    embedding vector NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, embedding_model, embedding_version),
    CHECK (embedding_dimensions BETWEEN 1 AND 2000),
    CHECK (vector_dims(embedding) = embedding_dimensions)
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_version
    ON chunk_embeddings (embedding_model, embedding_version, chunk_id);

-- Default production dimension. For a replacement model with a different
-- dimension, create the equivalent expression index with vector(<dimension>).
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw_cosine_1024
    ON chunk_embeddings
    USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 128)
    WHERE embedding_dimensions = 1024;

COMMIT;
