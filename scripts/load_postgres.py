"""Restartable PostgreSQL bulk loader using psql COPY via staging tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOAD_ORDER = (
    "companies", "disclosures", "sections", "disclosure_tables", "chunks",
    "chunk_source_refs",
)


def _connection_environment() -> dict[str, str]:
    env = os.environ.copy()
    url = env.get("DATABASE_URL")
    if not url:
        return env
    parsed = urlsplit(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("DATABASE_URL must use postgres:// or postgresql://")
    mapping = {
        "PGHOST": parsed.hostname,
        "PGPORT": str(parsed.port) if parsed.port else None,
        "PGUSER": unquote(parsed.username) if parsed.username else None,
        "PGPASSWORD": unquote(parsed.password) if parsed.password else None,
        "PGDATABASE": unquote(parsed.path.lstrip("/")) if parsed.path else None,
    }
    query = parse_qs(parsed.query)
    if query.get("sslmode"):
        mapping["PGSSLMODE"] = query["sslmode"][0]
    for key, value in mapping.items():
        if value:
            env[key] = value
    return env


def _run_psql(psql: str, sql: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        [psql, "-X", "--set", "ON_ERROR_STOP=1", "--file", "-"],
        input=sql,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def _psql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _copy_sql(table: str, columns: list[str], path: Path) -> str:
    column_sql = ", ".join(f'"{column}"' for column in columns)
    return f"""\
\\set ON_ERROR_STOP on
BEGIN;
CREATE TEMP TABLE load_{table} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP;
\\copy load_{table} ({column_sql}) FROM '{_psql_path(path)}' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '\\N')
INSERT INTO {table} ({column_sql})
SELECT {column_sql} FROM load_{table}
ON CONFLICT DO NOTHING;
COMMIT;
"""


def _validate_export(export_dir: Path) -> tuple[dict, dict[str, list[str]]]:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing export manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    headers: dict[str, list[str]] = {}
    for table in LOAD_ORDER:
        path = export_dir / manifest["files"][table]
        if not path.is_file():
            raise FileNotFoundError(f"missing COPY file: {path}")
        with path.open("r", encoding="utf-8", newline="") as source:
            headers[table] = next(csv.reader(source))
        if not headers[table]:
            raise ValueError(f"empty CSV header: {path}")
    report_path = next(
        (
            path
            for path in (
                export_dir / "validation_report.json",
                export_dir / "export_report.json",
                export_dir / "reports" / "export_validation.json",
            )
            if path.is_file()
        ),
        None,
    )
    if report_path is None:
        raise FileNotFoundError("missing export validation report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise ValueError(f"export validation is not valid: {report_path}")
    return manifest, headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "db_export",
    )
    parser.add_argument("--psql", default=os.getenv("PSQL", "psql"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--apply-indexes", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--state-file", type=Path)
    args = parser.parse_args()

    manifest, headers = _validate_export(args.export_dir)
    state_path = args.state_file or args.export_dir / ".import_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file() and not args.no_resume
        else {"completed_tables": []}
    )
    completed = set(state.get("completed_tables") or [])
    plan = [table for table in LOAD_ORDER if table not in completed]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid_export": True,
                    "database_connection_attempted": False,
                    "load_order": list(LOAD_ORDER),
                    "already_completed": sorted(completed),
                    "pending": plan,
                    "counts": manifest["counts"],
                    "connection_source": "DATABASE_URL or standard PG* environment variables",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    env = _connection_environment()
    if args.apply_schema:
        _run_psql(args.psql, (PROJECT_ROOT / "db" / "001_schema.sql").read_text(encoding="utf-8"), env)
    for table in plan:
        path = args.export_dir / manifest["files"][table]
        print(f"loading {table} from {path}", flush=True)
        _run_psql(args.psql, _copy_sql(table, headers[table], path), env)
        completed.add(table)
        state = {
            "completed_tables": [name for name in LOAD_ORDER if name in completed],
            "export_manifest": str((args.export_dir / "manifest.json").resolve()),
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if args.apply_indexes:
        _run_psql(args.psql, (PROJECT_ROOT / "db" / "002_indexes.sql").read_text(encoding="utf-8"), env)
    if args.validate:
        output = _run_psql(
            args.psql,
            (PROJECT_ROOT / "db" / "003_validation.sql").read_text(encoding="utf-8"),
            env,
        )
        print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"load failed: {error}", file=sys.stderr)
        raise SystemExit(1)
