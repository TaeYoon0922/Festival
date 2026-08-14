# Structural v2.1 PostgreSQL load

Target: PostgreSQL 16 with pgvector 0.8.6. The schema enables `vector` but deliberately
creates no embedding column and loads no embedding data.

## Actual-output field decisions

- `doc_id`, `chunk_id`, local `section_id`, and local `table_id` are preserved exactly.
- Because section/table IDs restart for every parsed source, `source_part_id` is part of
  their primary and foreign keys.
- `retrieval_priority` exists on saved table/projection chunks and is exported.
- `market`, `parser_mode`, and `structure_type` do not exist in the manifest or frozen
  payloads. `market` is exported as NULL; parser/structure fields are not invented.
- `table_title` is copied only when an actual saved chunk for that table contains it.
- Projection provenance comes only from saved `projection_field_refs` and `source_refs`.
  Table chunk ranges come directly from the saved `table_id`, `row_start`, and `row_end`.

## Files

The COPY files are generated under `data/db_export`:

```powershell
.\.venv\Scripts\python.exe scripts\export_db_release.py
.\.venv\Scripts\python.exe scripts\validate_db_export.py
```

CSV format is UTF-8, RFC-style quoted CSV with a header and `\N` as SQL NULL. Embedded
quotes, commas, CR/LF, and newlines in `content`, `retrieval_text`, and JSON are handled
by the CSV quote rules and validated by reading every logical record back.

## Load order

Run the schema first, load with client-side `\copy`, then create retrieval indexes:

```text
psql -X -v ON_ERROR_STOP=1 -f db/001_schema.sql
```

From a `psql` session at the repository root:

```sql
\copy companies FROM 'data/db_export/companies.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\N')
\copy disclosures FROM 'data/db_export/disclosures.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\N')
\copy sections FROM 'data/db_export/sections.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\N')
\copy disclosure_tables FROM 'data/db_export/disclosure_tables.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\N')
\copy chunks FROM 'data/db_export/chunks.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\N')
\copy chunk_source_refs FROM 'data/db_export/chunk_source_refs.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\N')
```

Then run:

```text
psql -X -v ON_ERROR_STOP=1 -f db/002_indexes.sql
psql -X -v ON_ERROR_STOP=1 -f db/003_validation.sql
```

Alternatively, `scripts/load_postgres.py` uses staging-table COPY plus
`ON CONFLICT DO NOTHING`, records completed tables, and supports `--dry-run` and restart.
Connection credentials are read from `DATABASE_URL` or standard `PG*` environment
variables and are never stored in code.
