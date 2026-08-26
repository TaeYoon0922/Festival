"""Build the correction graph offline and report its invariants.

The script is read-only with respect to the corpus and the database.  It reads
the frozen manifest plus the frozen parser output, resolves every correction
edge, and prints the diagnostics required to review the graph before anything
is written to PostgreSQL.  ``--write`` hands the same deterministic result to
``PostgresCorrectionRepository`` instead of inventing a second code path.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.sampling import load_manifest  # noqa: E402
from app.reasoning.correction_graph import (  # noqa: E402
    CorrectionNotice,
    DisclosureRecord,
    build_correction_graph,
    extract_correction_notice,
)


def _index_records(processed_dir: Path) -> list[dict[str, Any]]:
    path = processed_dir / "index.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tables(processed_dir: Path, output_path: str) -> Iterator[dict[str, Any]]:
    path = processed_dir / str(output_path).replace("\\", "/")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    for table in payload.get("tables") or []:
        yield table


def collect_notices(
    processed_dir: Path, correction_doc_ids: set[str]
) -> dict[str, CorrectionNotice]:
    """Read the correction notice out of each correcting document's tables."""

    notices: dict[str, CorrectionNotice] = {}
    for record in _index_records(processed_dir):
        doc_id = str(record.get("doc_id") or "")
        if doc_id not in correction_doc_ids or doc_id in notices:
            continue
        notice = extract_correction_notice(
            doc_id, _tables(processed_dir, record["output_path"])
        )
        if notice is not None:
            notices[doc_id] = notice
    return notices


def edge_report(graph: Any, records: list[Any]) -> list[dict[str, Any]]:
    """Every edge with both documents' metadata, for reviewing one rule's output.

    Weak rules are the ones worth auditing: filter the result on
    ``resolution_source`` to review exactly what a rule produced and how far
    apart the two filings are.
    """

    by_doc = {record.doc_id: record for record in records}

    def described(doc_id: str | None) -> dict[str, Any]:
        record = by_doc.get(str(doc_id or ""))
        if record is None:
            return {}
        return {
            "doc_id": record.doc_id,
            "rcept_no": record.rcept_no,
            "rcept_dt": record.rcept_dt,
            "report_nm": record.report_nm,
            "doc_group": record.doc_group,
            "doc_subtype": record.doc_subtype,
            "is_correction": record.is_correction,
        }

    rows: list[dict[str, Any]] = []
    for relation in graph.relations:
        source = by_doc.get(relation.source_doc_id)
        target = by_doc.get(relation.target_doc_id or "")
        gap = None
        if source and target and source.rcept_dt and target.rcept_dt:
            gap = (
                date.fromisoformat(source.rcept_dt) - date.fromisoformat(target.rcept_dt)
            ).days
        rows.append(
            {
                "relation_id": relation.relation_id,
                "resolution_status": relation.resolution_status,
                "resolution_source": relation.resolution_source,
                "confidence": relation.confidence,
                "corp_code": source.corp_code if source else None,
                "gap_days": gap,
                "source": described(relation.source_doc_id),
                "target": described(relation.target_doc_id),
                "evidence": dict(relation.evidence),
            }
        )
    rows.sort(
        key=lambda row: (
            row["resolution_source"],
            -(row["gap_days"] or 0),
            row["source"].get("doc_id", ""),
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "corpus" / "manifest.jsonl"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204",
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the diagnostics JSON here"
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=None,
        help=(
            "write every resolved/ambiguous/unresolved edge with both documents' "
            "metadata as JSON, for auditing one rule's output"
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="persist the graph through PostgresCorrectionRepository",
    )
    parser.add_argument(
        "--from-database",
        action="store_true",
        help="read disclosures and notices from PostgreSQL instead of the corpus",
    )
    parser.add_argument(
        "--allow-partial-write",
        action="store_true",
        help=(
            "permit --write when the build does not cover every disclosure; "
            "only the built disclosures' rows are touched"
        ),
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="print this many resolved chains"
    )
    args = parser.parse_args()

    if args.from_database:
        from app.retrieval.correction_repository import PostgresCorrectionRepository
        from app.retrieval.postgres_backend import PostgresBackend

        repository = PostgresCorrectionRepository(PostgresBackend())
        records = repository.load_disclosure_records()
        notices = repository.load_correction_notices()
    else:
        records = [
            DisclosureRecord.from_mapping(row) for row in load_manifest(args.manifest)
        ]
        correction_ids = {record.doc_id for record in records if record.is_correction}
        notices = collect_notices(args.processed_dir, correction_ids)

    graph = build_correction_graph(records, notices)
    diagnostics = {
        "disclosure_count": len(records),
        "correction_document_count": sum(
            1 for record in records if record.is_correction
        ),
        "notice_extracted_count": len(notices),
        **graph.diagnostics(),
    }
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    if args.sample:
        printed = 0
        seen: set[str] = set()
        for member in graph.members:
            group_id = member.correction_group_id
            if group_id in seen:
                continue
            seen.add(group_id)
            chain = graph.get_correction_chain(member.doc_id)
            if len(chain) < 2:
                continue
            print(f"\n{group_id}")
            for item in chain:
                print(
                    f"  #{item.correction_order} {item.doc_id} "
                    f"parent={item.parent_doc_id} latest={item.is_latest} "
                    f"source={item.resolution_source}"
                )
            printed += 1
            if printed >= args.sample:
                break

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.edges:
        args.edges.parent.mkdir(parents=True, exist_ok=True)
        args.edges.write_text(
            json.dumps(edge_report(graph, records), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {len(graph.relations)} edges to {args.edges}")

    if args.write:
        from app.retrieval.correction_repository import PostgresCorrectionRepository
        from app.retrieval.postgres_backend import PostgresBackend

        repository = PostgresCorrectionRepository(PostgresBackend())
        built = {record.doc_id for record in records}
        stored = repository.disclosure_doc_ids()
        missing = stored - built
        if missing and not args.allow_partial_write:
            # A build over part of the corpus resolves the rest as if it did not
            # exist. Rewriting the whole graph from it would delete correct
            # relations, so it is refused rather than scoped silently.
            raise SystemExit(
                f"refusing to write: this build covers {len(built)} of "
                f"{len(stored)} disclosures in the database "
                f"({len(missing)} missing, e.g. {sorted(missing)[:3]}). "
                "Point --manifest/--processed-dir at the full corpus, or use "
                "--from-database, or pass --allow-partial-write to update only "
                "the disclosures this build covers."
            )
        written = repository.persist_graph(
            graph, scope_doc_ids=None if not missing else built
        )
        print(json.dumps(written, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
