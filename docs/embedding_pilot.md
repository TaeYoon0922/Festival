# Gold60 candidate embedding pilot

This pilot deliberately embeds only the union of metadata-routed Gold60 candidates.
It does not add gold documents or chunks to the candidate set and does not alter the
frozen parser, chunker, export, provenance, IDs, or `retrieval_text`.

## Decision sequence

1. Collect the candidate union from question text through QueryUnderstanding and Router.
2. Estimate subset and full-corpus storage before any embedding call.
3. Apply the separate vector table migration after backup and model review.
4. Dry-run missing-row detection.
5. Embed/upsert the candidate subset in resumable batches.
6. Run Hybrid Gold60 and compare with the lexical baseline: 0.233 / 0.350 / 0.433.
7. Consider the full 1,363,336 chunks only after meaningful pilot improvement.

## Environment

Database credentials use `DATABASE_URL` or standard PostgreSQL `PG*` variables. Provider
credentials are runtime-only environment variables and are never written to reports.

```bash
export FESTIVAL_EMBEDDING_PROVIDER=bge_m3_local
export FESTIVAL_EMBEDDING_MODEL='BAAI/bge-m3'
export FESTIVAL_EMBEDDING_VERSION='6892b95fed65c899a30896eb40d619ae284d0455'
export FESTIVAL_EMBEDDING_DIMENSIONS=1024
export FESTIVAL_EMBEDDING_BATCH_SIZE=4
export FESTIVAL_EMBEDDING_MAX_LENGTH=8192
export FESTIVAL_EMBEDDING_DEVICE=cpu
```

`bge_m3_local` lazily loads `BGEM3FlagModel`; install its optional dependencies with
`pip install -r requirements-embedding.txt`. It requests dense vectors only and applies
final L2 normalization to query and document vectors. This matches pgvector cosine
distance. Query and documents use the same model/version, max length, dense flags, and
normalization path.

For a remote service wrapping the documented BGE-M3 HTTP payload, change only:

```bash
export FESTIVAL_EMBEDDING_PROVIDER=bge_m3_http
export FESTIVAL_EMBEDDING_API_URL='https://provider.example/bge-m3/embed'
export FESTIVAL_EMBEDDING_API_KEY='<secret>'
export FESTIVAL_EMBEDDING_API_KEY_HEADER='Authorization'
export FESTIVAL_EMBEDDING_API_KEY_PREFIX='Bearer'
export FESTIVAL_EMBEDDING_TIMEOUT_SECONDS=60
```

The existing OpenAI-compatible provider remains available. Other providers can
implement `EmbeddingProvider` without changing collection, batching, PostgreSQL, or
hybrid evaluation.

## Server commands

From `/srv/festival/app` on the test server:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
pip install -r requirements-embedding.txt

python scripts/collect_gold60_embedding_candidates.py

python scripts/estimate_embedding_storage.py \
  --candidate-ids data/processed/gold60_embedding_candidates/candidate_chunk_ids.txt \
  --dimensions "$FESTIVAL_EMBEDDING_DIMENSIONS" \
  --storage-capacity-gb 100 \
  --output data/processed/gold60_embedding_candidates/storage_estimate.json

python scripts/generate_vector_index_sql.py \
  --dimensions "$FESTIVAL_EMBEDDING_DIMENSIONS" \
  --output data/processed/gold60_embedding_candidates/hnsw_index.sql
```

On the CPU-only test server, benchmark before any migration or DB write:

```bash
python scripts/benchmark_embedding_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --limit 10 --batch-size 2 --device cpu

python scripts/benchmark_embedding_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --limit 100 --batch-size 4 --device cpu
```

The benchmark never constructs a PostgreSQL backend. It reports model load time,
embedding throughput, mean/p50/p95 batch latency, estimated durations for 76,438 and
1,363,336 chunks, and process peak RSS when supported by the operating system.

The index helper only writes SQL. Review it; do not build the full-corpus HNSW index as
part of candidate collection. When the migration is approved and backed up, apply it
from the test server over the private DB network:

```bash
psql "$DATABASE_URL" -X -v ON_ERROR_STOP=1 -f db/004_vector_search.sql
```

Check missing work without calling the provider or writing embeddings:

```bash
python scripts/embed_chunk_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --limit 100 --batch-size 4 --device cpu --dry-run
```

The future approved pilot write is the same command without `--dry-run`:

```bash
python scripts/embed_chunk_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --resume \
  --limit 100 \
  --batch-size 4 \
  --device cpu
```

Each committed batch is its own resume checkpoint. Existing rows with the same
model/version are skipped. Use `--force` only for an intentional overwrite. Errors are
appended to `embedding_errors.jsonl` by default.

After candidate coverage reaches the intended level, run the existing evaluator with
the exact same provider/model/version/dimensions environment:

```bash
python scripts/evaluate_postgres_hybrid_gold60.py \
  --lexical-top-n 50 \
  --vector-top-n 50 \
  --top-k 10 \
  --rrf-k 60
```

The report includes unique embedded candidate coverage and per-question zero-coverage
diagnostics. Storage estimates are conservative planning figures, not byte guarantees;
verify actual free space, existing PostgreSQL size, WAL headroom, and index workspace.

`db/004_vector_search.sql` already defines a partial HNSW cosine expression index over
`vector(1024)`, matching BGE-M3 dense output. No migration rewrite is required. Building
or rebuilding the full index remains a separate operational step and is not part of the
smoke benchmark.
