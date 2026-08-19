"""Safe SQL generation for dimension-specific pgvector HNSW indexes."""

from __future__ import annotations


def generate_hnsw_index_sql(
    dimensions: int,
    *,
    concurrently: bool = True,
    m: int = 16,
    ef_construction: int = 128,
) -> str:
    """Generate SQL only; callers decide when to run the expensive index build."""

    if not 1 <= dimensions <= 2000:
        raise ValueError("vector dimensions must be between 1 and 2000")
    if not 2 <= m <= 100:
        raise ValueError("HNSW m must be between 2 and 100")
    if not 4 <= ef_construction <= 1000:
        raise ValueError("HNSW ef_construction must be between 4 and 1000")
    concurrent = " CONCURRENTLY" if concurrently else ""
    return (
        f"CREATE INDEX{concurrent} IF NOT EXISTS "
        f"idx_chunk_embeddings_hnsw_cosine_{dimensions}\n"
        "    ON chunk_embeddings\n"
        f"    USING hnsw ((embedding::vector({dimensions})) vector_cosine_ops)\n"
        f"    WITH (m = {m}, ef_construction = {ef_construction})\n"
        f"    WHERE embedding_dimensions = {dimensions};\n"
    )
