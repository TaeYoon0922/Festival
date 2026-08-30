"""Materialize P0-A's holding correction groups as a corpus-bound artifact.

The correction rules are not restated here.  This script runs the frozen P0-A
builder over the whole corpus and writes down what it concluded about the
groups that touch holding filings -- root, ordered members, final member,
status, the rule that produced the relation and its confidence.  Where P0-A
said ambiguous, the artifact says ambiguous; where it proved a final member,
the artifact records that member and nothing about how it might have been
chosen differently.

Why an artifact at all, when the graph can be built on demand: reading the
correction notice out of every correcting document costs about thirty seconds
of gunzip and table scanning, which is fine once and impossible per request.
The artifact is bound to the corpus that produced it with the same identity
fields the holding report index uses, so the two can only be used together when
they describe the same corpus.

Read-only with respect to the corpus and the database; the only thing written
is the artifact.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.parsing.sampling import load_manifest
from app.reasoning.correction_graph import (
    RESOLVED,
    DisclosureRecord,
    build_correction_graph,
)
from app.reasoning.holding_correction_finality import (
    ARTIFACT_SCHEMA_VERSION,
    STATUS_AMBIGUOUS,
    STATUS_RESOLVED,
)
# The identity of a corpus is whatever the report index already says it is.
# Importing both rather than restating either is what keeps the two artifacts
# from agreeing on a corpus while disagreeing on how it is identified.
from scripts.build_correction_graph import collect_notices
from scripts.build_holding_report_index import (
    PROCESSED,
    git_commit,
    manifest_hash,
)

MANIFEST = ROOT / "data/corpus/manifest.jsonl"
DEFAULT_OUT = ROOT / "data/corpus/holding_correction_finality.json"

HOLDING = "holding"


def materialize_group(
    group_id: str,
    members,
    by_doc: Mapping[str, DisclosureRecord],
    holding_docs: set[str],
) -> dict:
    """Write down what P0-A concluded about one group.  Decide nothing.

    Every judgement here is a reading of P0-A's own member rows -- their order,
    their status, which one it marked latest.  The only thing this function
    adds is refusing to *call* a group proven when P0-A did not: a lone
    ambiguous correction also carries ``is_latest``, because nothing is known
    to supersede it, and reporting that as a final version would turn "we could
    not tell" into "this is the one".
    """

    ordered = sorted(members, key=lambda m: m.correction_order)
    terminals = [m for m in ordered if m.is_latest]
    resolved = (
        len(ordered) > 1
        and all(m.resolution_status == RESOLVED for m in ordered)
        and len(terminals) == 1
    )
    final = terminals[0] if resolved else None
    # The rules that produced this chain's own relations, and the weakest
    # confidence among them: provenance for review, never a threshold.
    edges = [m for m in ordered if m.parent_doc_id]
    root = by_doc.get(ordered[0].doc_id)
    status = (
        STATUS_RESOLVED if resolved
        else ordered[0].resolution_status if len(ordered) == 1
        else STATUS_AMBIGUOUS
    )
    return {
        "group_id": group_id,
        "root_doc_id": ordered[0].root_doc_id,
        "members": [m.doc_id for m in ordered],
        "final_doc_id": final.doc_id if final else None,
        "status": status,
        "resolution_rule": ",".join(sorted({m.resolution_source for m in edges})),
        "confidence": min((m.confidence for m in edges), default=0.0),
        # False when the chain's own root is a correction whose target lies
        # outside this corpus: complete at the tail, not at the head.
        "head_complete": not (root.is_correction if root else False),
        "provenance": {
            "member_rules": {m.doc_id: m.resolution_source for m in ordered},
            "member_orders": {m.doc_id: m.correction_order for m in ordered},
            "parents": {m.doc_id: m.parent_doc_id for m in ordered
                        if m.parent_doc_id},
            "holding_members": sorted(
                m.doc_id for m in ordered if m.doc_id in holding_docs),
            "non_holding_members": sorted(
                m.doc_id for m in ordered if m.doc_id not in holding_docs),
        },
    }


def build() -> tuple[list[dict], dict]:
    """Run frozen P0-A over the corpus and keep the holding groups it produced."""

    records = [DisclosureRecord.from_mapping(row) for row in load_manifest(MANIFEST)]
    holding_docs = {r.doc_id for r in records if r.doc_group == HOLDING}
    correction_docs = {r.doc_id for r in records if r.is_correction}
    holding_corrections = holding_docs & correction_docs

    notices = collect_notices(PROCESSED, correction_docs)
    graph = build_correction_graph(records, notices)

    by_group: dict[str, list] = {}
    for member in graph.members:
        by_group.setdefault(member.correction_group_id, []).append(member)

    by_doc = {r.doc_id: r for r in records}
    groups: list[dict] = []
    counts = collections.Counter()
    classified: set[str] = set()

    for group_id, members in by_group.items():
        if not any(member.doc_id in holding_docs for member in members):
            continue
        group = materialize_group(group_id, members, by_doc, holding_docs)
        counts[group["status"]] += 1
        classified.update(
            doc for doc in group["members"] if doc in holding_docs)
        groups.append(group)

    groups.sort(key=lambda g: g["group_id"])

    stats = {
        "disclosures_loaded": len(records),
        "holding_documents": len(holding_docs),
        "holding_correction_documents": len(holding_corrections),
        "holding_correction_documents_classified": len(
            holding_corrections & classified),
        "unclassified_holding_correction_documents": sorted(
            holding_corrections - classified),
        "notices_extracted": len(notices),
        "groups_touching_holding": len(groups),
        "status_counts": dict(counts),
        "graph": graph.diagnostics(),
    }
    stats["complete"] = (
        bool(holding_docs)
        and not stats["unclassified_holding_correction_documents"]
        and stats["graph"]["cycle_count"] == 0
        and stats["graph"]["invalid_latest_group_count"] == 0
        and stats["graph"]["duplicate_relation_count"] == 0
        and stats["graph"]["self_reference_count"] == 0
    )
    return groups, stats


def _serialize(payload: dict) -> str:
    """One group per line, so a corpus change reads as a line diff."""

    head = {k: v for k, v in payload.items() if k != "groups"}
    lines = [json.dumps(head, ensure_ascii=False, indent=1)[:-2].rstrip() + ",",
             ' "groups": [']
    rows = [json.dumps(group, ensure_ascii=False, separators=(",", ":"),
                       sort_keys=True)
            for group in payload["groups"]]
    lines.append(",\n".join("  " + row for row in rows))
    lines.append(" ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="report coverage without writing the artifact")
    args = parser.parse_args()

    groups, stats = build()

    print("=== COVERAGE ===")
    for key in ("disclosures_loaded", "holding_documents",
                "holding_correction_documents",
                "holding_correction_documents_classified",
                "notices_extracted", "groups_touching_holding", "complete"):
        print(f"  {key:44s} {stats[key]}")
    for key, value in sorted(stats["status_counts"].items()):
        print(f"  status {key:37s} {value}")
    missing = stats["unclassified_holding_correction_documents"]
    if missing:
        print(f"  UNCLASSIFIED correction documents: {missing[:5]}")

    if not stats["complete"]:
        print("\nREFUSED: the source does not cover the active holding corpus.")
        return 2

    header = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "corpus_snapshot_id": PROCESSED.name,
        "corpus_manifest_sha256": manifest_hash(),
        "source_disclosure_count": stats["disclosures_loaded"],
        # Spelled as the report index spells it: the two artifacts compare this
        # field directly, so the name is part of the contract.
        "source_holding_disclosure_count": stats["holding_documents"],
        "source_holding_correction_document_count":
            stats["holding_correction_documents"],
        "generation_commit": git_commit(),
        "generation_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
    }
    payload = {"header": header, "complete": True, "groups": groups}
    blob = _serialize(payload)

    if args.dry_run:
        print(f"\nDRY RUN: {len(groups)} groups, "
              f"{len(blob.encode('utf-8'))/1024:.1f} KiB")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written through a temporary file so a failed run cannot leave a partial
    # artifact where a complete one is expected.
    staging = out.with_suffix(out.suffix + ".tmp")
    staging.write_text(blob, encoding="utf-8")
    staging.replace(out)
    print(f"\nwrote {out} -- {len(groups)} groups, "
          f"{out.stat().st_size/1024:.1f} KiB, "
          f"resolved {stats['status_counts'].get(STATUS_RESOLVED, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
