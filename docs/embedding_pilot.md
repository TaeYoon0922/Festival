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
export FESTIVAL_EMBEDDING_PROVIDER=openai_compatible
export FESTIVAL_EMBEDDING_MODEL='<provider-model-name>'
export FESTIVAL_EMBEDDING_VERSION='<immutable-pilot-version>'
export FESTIVAL_EMBEDDING_DIMENSIONS=768
export FESTIVAL_EMBEDDING_BATCH_SIZE=32
export FESTIVAL_EMBEDDING_API_URL='https://provider.example/v1/embeddings'
export FESTIVAL_EMBEDDING_API_KEY='<secret>'
export FESTIVAL_EMBEDDING_API_KEY_HEADER='Authorization'
export FESTIVAL_EMBEDDING_API_KEY_PREFIX='Bearer'
export FESTIVAL_EMBEDDING_TIMEOUT_SECONDS=60
```

The production adapter uses an OpenAI-compatible embeddings JSON contract behind an
injectable HTTP transport. Other providers can implement the same `EmbeddingProvider`
interface without changing collection, batching, PostgreSQL, or hybrid evaluation.

## Server commands

From `/srv/festival/app` on the test server:

```bash
source .venv/bin/activate
python -m unittest discover -s tests

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
  --dry-run
```

The future approved pilot write is the same command without `--dry-run`:

```bash
python scripts/embed_chunk_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --resume \
  --batch-size "$FESTIVAL_EMBEDDING_BATCH_SIZE"
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
