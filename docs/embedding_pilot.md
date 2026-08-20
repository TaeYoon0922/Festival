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
export FESTIVAL_EMBEDDING_CUDA_OOM_RETRY=true
export FESTIVAL_EMBEDDING_MIN_BATCH_SIZE=1
```

`bge_m3_local` lazily loads `BGEM3FlagModel`; install its optional dependencies with
`pip install -r requirements-embedding.txt`. It requests dense vectors only and applies
final L2 normalization to query and document vectors. This matches pgvector cosine
distance. Query and documents use the same model/version, max length, dense flags, and
normalization path.

For GPU inference, install the PyTorch build matching the server CUDA driver before
installing `requirements-embedding.txt`, set `FESTIVAL_EMBEDDING_DEVICE=cuda` (or
`cuda:0`), and tune `FESTIVAL_EMBEDDING_BATCH_SIZE` to 8, 16, or 32. On a recognized
CUDA out-of-memory error, the local provider clears the CUDA cache, halves the batch
size down to `FESTIVAL_EMBEDDING_MIN_BATCH_SIZE`, and retains the successful size for
later batches. Non-CUDA and non-OOM failures are not treated as an OOM.

For a remote service wrapping the documented BGE-M3 HTTP payload, change only:

```bash
export FESTIVAL_EMBEDDING_PROVIDER=bge_m3_http
export FESTIVAL_EMBEDDING_API_URL='https://provider.example/bge-m3/embed'
export FESTIVAL_EMBEDDING_API_KEY='<secret>'
export FESTIVAL_EMBEDDING_API_KEY_HEADER='Authorization'
export FESTIVAL_EMBEDDING_API_KEY_PREFIX='Bearer'
export FESTIVAL_EMBEDDING_TIMEOUT_SECONDS=60
```

The HTTP adapter sends the same immutable model revision, max length, and dense-only
normalization request as the local adapter. Workflow retries apply exponential backoff
to timeouts, connection errors, HTTP 429, and HTTP 5xx; `Retry-After` is honored when
present, capped by `--max-retry-delay-seconds`. Other HTTP 4xx responses fail
immediately. Transport errors are sanitized so
API keys, authorization headers, request bodies, and response bodies are not logged.
Both the native BGE-M3 response (`dense_vecs`) and an OpenAI-style ordered `data` array
are accepted, then dimension-checked and L2-normalized locally.

The existing OpenAI-compatible provider remains available. Other providers can
implement `EmbeddingProvider` without changing collection, batching, PostgreSQL, or
hybrid evaluation.

### CLOVA Studio OpenAI-compatible endpoint

CLOVA Studio is selected explicitly and is never inferred from the endpoint hostname:

```bash
export FESTIVAL_EMBEDDING_PROVIDER=clova_studio
export FESTIVAL_EMBEDDING_MODEL=bge-m3
export FESTIVAL_EMBEDDING_VERSION='<deployed-model-version>'
export FESTIVAL_EMBEDDING_DIMENSIONS=1024
export FESTIVAL_EMBEDDING_BATCH_SIZE=1
export FESTIVAL_EMBEDDING_API_URL='https://clovastudio.stream.ntruss.com/v1/openai/embeddings'
export FESTIVAL_EMBEDDING_API_KEY='<secret>'
export FESTIVAL_EMBEDDING_API_KEY_HEADER='Authorization'
export FESTIVAL_EMBEDDING_API_KEY_PREFIX='Bearer'
export FESTIVAL_EMBEDDING_LONG_TEXT_SEGMENT_CHARS=1800
```

The CLOVA request mode sends `input` as a single string together with
`encoding_format=float` and the configured `dimensions`. When the pipeline batch size
is greater than one, the adapter uses a deliberate sequential fallback: one
single-string HTTP request per document, returned in the original order. A failure is
reported to the existing pipeline retry boundary, so resume, retry, validation, and DB
upsert contracts remain unchanged. The generic `openai_compatible` provider continues
to send one list-valued batch request and does not infer behavior from the hostname.

If and only if CLOVA returns HTTP 400 with `error.code=40003`, the adapter retries the
input through the long-text path. It splits the in-memory embedding input at paragraph,
line, sentence, or whitespace boundaries, with a conservative default ceiling of 1,800
characters; an oversized single paragraph is hard-split without dropping any range.
Every segment is embedded through the same endpoint, element-wise mean pooled, then L2
normalized into one 1024-dimensional vector. The frozen `retrieval_text`, chunk row,
chunk ID, and DB schema are never changed. Other HTTP 4xx errors still fail, while 429,
5xx, and transport timeouts retain the existing pipeline retry behavior.

Benchmark and subset-pipeline summaries expose only aggregate
`provider_statistics.long_text_fallbacks` and `long_text_segments`; input text,
response bodies, and credentials are not logged. The transport retains a sanitized
response error code/message as exception metadata, but exception text contains only the
HTTP status and sanitized error code.

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
1,363,336 chunks, CPU-baseline speedup, process peak RSS, and CUDA allocator peak when
supported. Local CPU, local CUDA, and HTTP all emit the same JSON field schema. The
default comparison baseline is the measured 1.489 documents/second and can be replaced
with `--cpu-baseline-documents-per-second`.

GPU benchmark (10 candidates, no database writes):

```bash
python scripts/benchmark_embedding_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --provider bge_m3_local \
  --model BAAI/bge-m3 \
  --version 6892b95fed65c899a30896eb40d619ae284d0455 \
  --dimensions 1024 --max-length 8192 \
  --limit 10 --batch-size 8 --min-batch-size 1 --device cuda
```

GPU benchmark (100 candidates, no database writes):

```bash
python scripts/benchmark_embedding_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --provider bge_m3_local \
  --model BAAI/bge-m3 \
  --version 6892b95fed65c899a30896eb40d619ae284d0455 \
  --dimensions 1024 --max-length 8192 \
  --limit 100 --batch-size 16 --min-batch-size 1 --device cuda
```

HTTP benchmark (10 candidates, no database writes; credentials stay in environment):

```bash
export FESTIVAL_EMBEDDING_API_URL='https://provider.example/bge-m3/embed'
export FESTIVAL_EMBEDDING_API_KEY='<secret>'
python scripts/benchmark_embedding_subset.py \
  --input data/processed/gold60_embedding_candidates/candidate_chunks.jsonl \
  --provider bge_m3_http \
  --model BAAI/bge-m3 \
  --version 6892b95fed65c899a30896eb40d619ae284d0455 \
  --dimensions 1024 --max-length 8192 \
  --limit 10 --batch-size 8 --max-attempts 3
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
