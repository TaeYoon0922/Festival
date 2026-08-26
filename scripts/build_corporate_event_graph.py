"""Build the corporate event timeline offline and report its invariants.

Read-only with respect to the corpus and the database.  It reads the frozen
manifest plus the frozen parser output, resolves every contract termination, and
prints the diagnostics required to review the graph before anything is written to
PostgreSQL.  ``--write`` hands the same deterministic result to
``PostgresCorporateEventRepository`` instead of inventing a second code path.

The correction graph is rebuilt alongside it and passed in as the
canonicalization layer, so P0-B always sees the same P0-A answer the serving path
would see.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.sampling import load_manifest  # noqa: E402
from app.reasoning.correction_graph import DisclosureRecord, build_correction_graph  # noqa: E402
from app.reasoning.corporate_event_graph import (  # noqa: E402
    AMBIGUOUS,
    FAMILY_SUPPLY_CONTRACT,
    FAMILY_TREASURY_TRUST,
    RESOLVED,
    UNRESOLVED,
    ContractDocument,
    classify_contract_document,
    extract_contract_document,
    build_corporate_event_graph,
)
from scripts.build_correction_graph import collect_notices  # noqa: E402


def _index_records(processed_dir: Path) -> list[dict[str, Any]]:
    path = processed_dir / "index.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tables(processed_dir: Path, output_path: str) -> Iterator[dict[str, Any]]:
    path = processed_dir / str(output_path).replace(chr(92), "/")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    for table in payload.get("tables") or []:
        yield table


def collect_contract_documents(
    processed_dir: Path, records: list[DisclosureRecord]
) -> dict[str, ContractDocument]:
    """Read the structured contract fields out of each v1 contract filing."""

    wanted = {
        record.doc_id: record
        for record in records
        if classify_contract_document(record) is not None
    }
    documents: dict[str, ContractDocument] = {}
    for entry in _index_records(processed_dir):
        doc_id = str(entry.get("doc_id") or "")
        record = wanted.get(doc_id)
        if record is None or doc_id in documents:
            continue
        document = extract_contract_document(
            record, _tables(processed_dir, entry["output_path"])
        )
        if document is not None:
            documents[doc_id] = document
    return documents


def termination_report(graph: Any, family: str) -> list[dict[str, Any]]:
    """Every termination of one family with the rule that resolved it.

    This is the manual-audit surface: it is small enough that both families can
    be reviewed edge by edge.
    """

    rows: list[dict[str, Any]] = []
    for event in graph.events:
        if event.event_family != family:
            continue
        for member in event.termination_members:
            document = member.evidence.get("document") or {}
            match = member.evidence.get("match") or {}
            contracts = [
                {
                    "doc_id": item.doc_id,
                    "canonical_doc_id": item.canonical_doc_id,
                    "role": item.member_role,
                    "event_date": item.event_date,
                    "correction_group_id": item.correction_group_id,
                    "correction_resolution_status": item.correction_resolution_status,
                }
                for item in event.contract_members
            ]
            rows.append(
                {
                    "termination_doc_id": member.doc_id,
                    "corp_code": event.corp_code,
                    "event_id": event.event_id,
                    "resolution_status": event.resolution_status,
                    "resolution_source": event.resolution_source,
                    "confidence": event.confidence,
                    "lifecycle_status": event.lifecycle_status,
                    "termination_date": document.get("termination_date"),
                    "termination_reason": document.get("termination_reason"),
                    "counterparty": document.get("counterparty"),
                    "subject": document.get("subject"),
                    "amount": document.get("amount"),
                    "period": [document.get("period_start"), document.get("period_end")],
                    "contract_members": contracts,
                    "contract_reference_dates": match.get("contract_reference_dates"),
                    "rejected_reference_titles": match.get("rejected_reference_titles"),
                    "candidate_doc_ids": match.get("candidate_doc_ids"),
                    "accepted_doc_ids": match.get("accepted_doc_ids"),
                    "rejected_doc_ids": match.get("rejected_doc_ids"),
                    "uncorroborated_doc_ids": match.get("uncorroborated_doc_ids"),
                    "comparisons": match.get("comparisons"),
                }
            )
    rows.sort(key=lambda row: row["termination_doc_id"])
    return rows


def family_diagnostics(graph: Any, family: str) -> dict[str, Any]:
    events = [event for event in graph.events if event.event_family == family]
    terminations = [
        (event, member)
        for event in events
        for member in event.termination_members
    ]
    by_status: dict[str, int] = {RESOLVED: 0, AMBIGUOUS: 0, UNRESOLVED: 0}
    for event, _ in terminations:
        by_status[event.resolution_status] = by_status.get(event.resolution_status, 0) + 1
    terminated = [event for event in events if event.is_terminated]
    raw_documents = sum(
        len((member.provenance or {}).get("collapsed_doc_ids") or (member.doc_id,))
        for event in events
        for member in event.members
    )
    return {
        "event_count": len(events),
        "raw_contract_document_count": raw_documents,
        "logical_event_member_count": sum(event.member_count for event in events),
        "resolved_correction_documents_collapsed": raw_documents
        - sum(event.member_count for event in events),
        "logical_contract_member_count": sum(
            len(event.contract_members) for event in events
        ),
        "termination_count": len(terminations),
        "resolved_termination": by_status.get(RESOLVED, 0),
        "ambiguous_termination": by_status.get(AMBIGUOUS, 0),
        "unresolved_termination": by_status.get(UNRESOLVED, 0),
        "terminated_event_count": len(terminated),
        "resolved_terminated_event_count": sum(
            1 for event in terminated if event.resolution_status == RESOLVED
        ),
        "multi_member_event_count": sum(1 for event in events if event.member_count > 1),
        # Two distinct logical contracts in one lifecycle, which is not the same
        # as a contract plus its termination.
        "multi_contract_event_count": sum(
            1 for event in events if len(event.contract_members) > 1
        ),
        "max_event_members": max((event.member_count for event in events), default=0),
        "resolution_sources": _counted(
            event.resolution_source for event, _ in terminations
        ),
        "confidence_distribution": _counted(
            f"{event.confidence:.2f}" for event, _ in terminations
        ),
    }


def _counted(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "manifest.jsonl",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--terminations",
        type=Path,
        default=None,
        help="write every termination with its rule and evidence, for manual audit",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="persist the graph through PostgresCorporateEventRepository",
    )
    parser.add_argument(
        "--from-database",
        action="store_true",
        help="read disclosures and contract tables from PostgreSQL instead of the corpus",
    )
    args = parser.parse_args()

    if args.from_database:
        from app.retrieval.corporate_event_repository import (
            PostgresCorporateEventRepository,
        )
        from app.retrieval.correction_repository import PostgresCorrectionRepository
        from app.retrieval.postgres_backend import PostgresBackend

        backend = PostgresBackend()
        repository = PostgresCorporateEventRepository(backend)
        records = repository.load_disclosure_records()
        documents = repository.load_contract_documents(records)
        correction_graph = PostgresCorrectionRepository(backend).load_graph()
    else:
        records = [
            DisclosureRecord.from_mapping(row) for row in load_manifest(args.manifest)
        ]
        documents = collect_contract_documents(args.processed_dir, records)
        correction_graph = build_correction_graph(
            records,
            collect_notices(
                args.processed_dir,
                {record.doc_id for record in records if record.is_correction},
            ),
        )

    graph = build_corporate_event_graph(
        records, documents, correction_graph=correction_graph
    )
    diagnostics = {
        "disclosure_count": len(records),
        "contract_document_count": len(documents),
        "supply_contract": family_diagnostics(graph, FAMILY_SUPPLY_CONTRACT),
        "treasury_trust_contract": family_diagnostics(graph, FAMILY_TREASURY_TRUST),
        "overall": graph.diagnostics(),
    }
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.terminations:
        args.terminations.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            FAMILY_SUPPLY_CONTRACT: termination_report(graph, FAMILY_SUPPLY_CONTRACT),
            FAMILY_TREASURY_TRUST: termination_report(graph, FAMILY_TREASURY_TRUST),
        }
        args.terminations.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"wrote {sum(len(rows) for rows in payload.values())} terminations "
            f"to {args.terminations}"
        )

    if args.write:
        from app.retrieval.corporate_event_repository import (
            PostgresCorporateEventRepository,
        )
        from app.retrieval.postgres_backend import PostgresBackend

        repository = PostgresCorporateEventRepository(PostgresBackend())
        built = {record.doc_id for record in records}
        stored = repository.disclosure_doc_ids()
        missing = stored - built
        if missing:
            # A partial build resolves the rest as if it did not exist, so
            # rewriting the whole graph from it would delete correct events.
            # v1 supports full-corpus rebuild only, which makes this a hard stop.
            raise SystemExit(
                f"refusing to write: this build covers {len(built)} of "
                f"{len(stored)} disclosures in the database "
                f"({len(missing)} missing, e.g. {sorted(missing)[:3]}). "
                "Point --manifest/--processed-dir at the full corpus, or use "
                "--from-database."
            )
        print(json.dumps(repository.persist_graph(graph), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
