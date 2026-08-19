"""Conservative planning estimates for pgvector table and HNSW storage."""

from __future__ import annotations

from typing import Any


FULL_CORPUS_CHUNKS = 1_363_336


def estimate_embedding_storage(
    candidate_count: int,
    dimensions: int,
    *,
    full_corpus_count: int = FULL_CORPUS_CHUNKS,
    float_bytes: int = 4,
    table_overhead_bytes_per_row: int = 128,
    hnsw_vector_multiplier: float = 1.5,
    hnsw_graph_bytes_per_row: int = 384,
    storage_capacity_gb: float = 100.0,
) -> dict[str, Any]:
    """Return planning estimates, not byte-accurate PostgreSQL guarantees."""

    if candidate_count < 0 or full_corpus_count < 0:
        raise ValueError("chunk counts must be non-negative")
    if not 1 <= dimensions <= 2000:
        raise ValueError("embedding dimensions must be between 1 and 2000")
    if float_bytes <= 0 or table_overhead_bytes_per_row < 0:
        raise ValueError("byte assumptions must be non-negative")
    if hnsw_vector_multiplier < 0.0 or hnsw_graph_bytes_per_row < 0:
        raise ValueError("HNSW assumptions must be non-negative")
    if storage_capacity_gb <= 0.0:
        raise ValueError("storage capacity must be positive")

    def scope(rows: int) -> dict[str, Any]:
        raw = rows * dimensions * float_bytes
        table_overhead = rows * table_overhead_bytes_per_row
        table_total = raw + table_overhead
        hnsw = int(raw * hnsw_vector_multiplier) + rows * hnsw_graph_bytes_per_row
        total = table_total + hnsw
        return {
            "rows": rows,
            "raw_vector_bytes": raw,
            "table_overhead_bytes": table_overhead,
            "table_total_bytes": table_total,
            "hnsw_index_bytes": hnsw,
            "embedding_layer_total_bytes": total,
            "embedding_layer_total_gib": round(total / (1024**3), 3),
        }

    candidate = scope(candidate_count)
    full = scope(full_corpus_count)
    capacity_bytes = int(storage_capacity_gb * (1024**3))
    return {
        "disclaimer": (
            "Conservative planning estimate only. PostgreSQL page fill, TOAST, WAL, "
            "index build workspace, model metadata, and existing DB usage vary."
        ),
        "assumptions": {
            "dimensions": dimensions,
            "float_bytes": float_bytes,
            "table_overhead_bytes_per_row": table_overhead_bytes_per_row,
            "hnsw_vector_multiplier": hnsw_vector_multiplier,
            "hnsw_graph_bytes_per_row": hnsw_graph_bytes_per_row,
            "storage_capacity_gb": storage_capacity_gb,
        },
        "candidate_subset": candidate,
        "full_corpus": full,
        "capacity": {
            "bytes": capacity_bytes,
            "full_embedding_layer_fraction": round(
                full["embedding_layer_total_bytes"] / capacity_bytes, 6
            ),
            "fits_embedding_layer_only": (
                full["embedding_layer_total_bytes"] <= capacity_bytes
            ),
            "requires_existing_db_and_free_space_check": True,
        },
    }
