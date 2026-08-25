"""PostgreSQL repository for the deterministic correction graph.

Every correction-aware caller goes through this repository instead of writing
its own SQL, so ``QueryPlan``, the router, and any diagnostic tool see one
consistent answer.  The repository owns persistence and lookup only; the rules
that decide which document corrects which live in
:mod:`app.reasoning.correction_graph` and never touch a database.

The backfill is idempotent.  Relations are keyed by a deterministic
``relation_id``, group members by ``doc_id``, and a rebuild upserts the current
result and deletes rows the current result no longer contains.  Running it twice
produces the same rows with the same ``created_at``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

from app.reasoning.correction_graph import (
    CORRECTION_RELATION_TYPE,
    RESOLVED,
    CorrectionGraphUnavailable,
    CorrectionDocumentState,
    CorrectionGroup,
    CorrectionGroupMember,
    CorrectionGraph,
    CorrectionNotice,
    CorrectionRelation,
    DisclosureRecord,
    build_correction_graph,
    correction_diagnostics,
    extract_correction_notice,
)


#: Every correction notice observed in the frozen corpus is complete within the
#: first four tables of a part; the default keeps a wide margin over that.
DEFAULT_NOTICE_TABLE_LIMIT = 16

_MEMBER_COLUMNS = (
    "doc_id",
    "correction_group_id",
    "root_doc_id",
    "parent_doc_id",
    "correction_order",
    "is_latest",
    "is_correction",
    "resolution_status",
    "resolution_source",
    "confidence",
    "evidence",
)
_RELATION_COLUMNS = (
    "relation_id",
    "source_doc_id",
    "target_doc_id",
    "relation_type",
    "resolution_status",
    "resolution_source",
    "confidence",
    "evidence",
)


def _json(value: Any) -> str:
    return json.dumps(dict(value or {}), ensure_ascii=False, sort_keys=True)


def _member(row: Mapping[str, Any]) -> CorrectionGroupMember:
    return CorrectionGroupMember(
        doc_id=str(row["doc_id"]),
        correction_group_id=str(row["correction_group_id"]),
        root_doc_id=str(row["root_doc_id"]),
        parent_doc_id=(
            str(row["parent_doc_id"]) if row.get("parent_doc_id") else None
        ),
        correction_order=int(row["correction_order"]),
        is_latest=bool(row["is_latest"]),
        resolution_status=str(row["resolution_status"]),
        resolution_source=str(row["resolution_source"]),
        confidence=float(row["confidence"]),
        is_correction=bool(row.get("is_correction")),
        evidence=dict(row.get("evidence") or {}),
    )


def _relation(row: Mapping[str, Any]) -> CorrectionRelation:
    return CorrectionRelation(
        relation_id=str(row["relation_id"]),
        source_doc_id=str(row["source_doc_id"]),
        relation_type=str(row["relation_type"]),
        target_doc_id=str(row["target_doc_id"]) if row.get("target_doc_id") else None,
        resolution_status=str(row["resolution_status"]),
        resolution_source=str(row["resolution_source"]),
        confidence=float(row["confidence"]),
        evidence=dict(row.get("evidence") or {}),
    )


class PostgresCorrectionRepository:
    """Build, persist, and read the correction graph stored in PostgreSQL."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    # ------------------------------------------------------------------ read

    @contextmanager
    def _expected_failures(self, operation: str) -> Iterator[None]:
        """Translate only "the graph is not there" into a degradable error.

        ``UndefinedTable`` means db/006 has not been applied; an
        ``OperationalError`` means the database is unreachable.  Both are
        conditions a caller may fall back from.  Every other psycopg error --
        an undefined column, a type mismatch, a broken statement -- keeps its
        own type and propagates, so a real defect is never disguised as a
        missing migration.
        """

        try:
            yield
        except psycopg.errors.UndefinedTable as error:
            raise CorrectionGraphUnavailable(
                f"correction graph tables are missing during {operation}; "
                "apply db/006_correction_graph.sql"
            ) from error
        except psycopg.OperationalError as error:
            raise CorrectionGraphUnavailable(
                f"database unreachable during {operation}"
            ) from error

    def _fetch(
        self, query: str, params: Sequence[Any] = (), *, operation: str = "read"
    ) -> list[dict[str, Any]]:
        with self._expected_failures(operation):
            with self._backend.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(query, tuple(params))
                    return [dict(row) for row in cursor.fetchall()]

    def get_correction_group(self, doc_id: str) -> CorrectionGroup | None:
        """Return the whole group one document belongs to, or None."""

        rows = self._fetch(
            """
            SELECT m.*
            FROM correction_group_members m
            WHERE m.correction_group_id = (
                SELECT correction_group_id
                FROM correction_group_members
                WHERE doc_id = %s
            )
            ORDER BY m.correction_order, m.doc_id
            """,
            [str(doc_id)],
            operation="get_correction_group",
        )
        if not rows:
            return None
        members = tuple(_member(row) for row in rows)
        return CorrectionGroup(
            correction_group_id=members[0].correction_group_id,
            root_doc_id=members[0].root_doc_id,
            members=members,
            resolution_status=(
                members[0].resolution_status if len(members) == 1 else RESOLVED
            ),
        )

    def get_correction_chain(self, doc_id: str) -> tuple[CorrectionGroupMember, ...]:
        """Original first, then every correction in receipt order.

        This is the ordering a before/after comparison needs: element ``0`` is
        the document as first filed and the last element is the final valid
        version.
        """

        group = self.get_correction_group(doc_id)
        return group.members if group is not None else ()

    def get_latest_report(
        self, doc_id: str | None = None, *, correction_group_id: str | None = None
    ) -> str | None:
        """Resolve the final valid document for a report or a whole group."""

        if correction_group_id is not None:
            rows = self._fetch(
                """
                SELECT doc_id
                FROM correction_group_members
                WHERE correction_group_id = %s AND is_latest
                """,
                [str(correction_group_id)],
            )
            return str(rows[0]["doc_id"]) if rows else None
        if doc_id is None:
            raise ValueError("doc_id or correction_group_id is required")
        rows = self._fetch(
            """
            SELECT latest.doc_id
            FROM correction_group_members AS self
            JOIN correction_group_members AS latest
              ON latest.correction_group_id = self.correction_group_id
             AND latest.is_latest
            WHERE self.doc_id = %s
            """,
            [str(doc_id)],
        )
        # A document outside any group is already its own latest version.
        return str(rows[0]["doc_id"]) if rows else str(doc_id)

    def document_states(
        self, doc_ids: Iterable[str]
    ) -> dict[str, CorrectionDocumentState]:
        """Return correction state for the subset of ``doc_ids`` in a group."""

        unique = sorted({str(doc_id) for doc_id in doc_ids if str(doc_id)})
        if not unique:
            return {}
        rows = self._fetch(
            """
            SELECT
                doc_id,
                correction_group_id,
                root_doc_id,
                parent_doc_id,
                correction_order,
                is_latest,
                is_correction,
                resolution_status
            FROM correction_group_members
            WHERE doc_id = ANY(%s)
            """,
            [unique],
            operation="document_states",
        )
        return {
            str(row["doc_id"]): CorrectionDocumentState(
                doc_id=str(row["doc_id"]),
                correction_group_id=str(row["correction_group_id"]),
                root_doc_id=str(row["root_doc_id"]),
                parent_doc_id=(
                    str(row["parent_doc_id"]) if row.get("parent_doc_id") else None
                ),
                correction_order=int(row["correction_order"]),
                is_latest=bool(row["is_latest"]),
                resolution_status=str(row["resolution_status"]),
                is_correction=bool(row["is_correction"]),
            )
            for row in rows
        }

    def load_graph(self) -> CorrectionGraph:
        """Materialize the persisted graph for in-process reuse."""

        members = [
            _member(row)
            for row in self._fetch(
                "SELECT * FROM correction_group_members "
                "ORDER BY correction_group_id, correction_order, doc_id"
            )
        ]
        relations = [
            _relation(row)
            for row in self._fetch(
                "SELECT * FROM correction_relations ORDER BY relation_id"
            )
        ]
        return CorrectionGraph(members, relations)

    # ----------------------------------------------------------------- build

    def load_disclosure_records(self) -> list[DisclosureRecord]:
        rows = self._fetch(
            """
            SELECT
                doc_id, corp_code, doc_group, doc_subtype, report_nm,
                rcept_no, rcept_dt, base_year, base_month, is_correction
            FROM disclosures
            ORDER BY doc_id
            """
        )
        return [DisclosureRecord.from_mapping(row) for row in rows]

    def disclosure_doc_ids(self) -> set[str]:
        """Every disclosure id, so a caller can tell a full build from a partial one."""

        return {
            str(row["doc_id"])
            for row in self._fetch(
                "SELECT doc_id FROM disclosures", operation="disclosure_doc_ids"
            )
        }

    def load_correction_notices(
        self, *, table_limit: int = DEFAULT_NOTICE_TABLE_LIMIT
    ) -> dict[str, CorrectionNotice]:
        """Read each correcting document's own correction notice tables.

        Only the leading tables of a correcting disclosure are read: the notice
        block is always the first thing a correcting filing states.
        """

        if table_limit <= 0:
            raise ValueError("table_limit must be positive")
        rows = self._fetch(
            """
            SELECT t.doc_id, t.table_id, t.table_rows
            FROM disclosure_tables t
            JOIN disclosures d ON d.doc_id = t.doc_id
            WHERE d.is_correction
              AND t.table_order < %s
            ORDER BY t.doc_id, t.source_part_id, t.table_order
            """,
            [int(table_limit)],
        )
        tables_by_doc: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tables_by_doc.setdefault(str(row["doc_id"]), []).append(
                {"table_id": row.get("table_id"), "table_rows": row.get("table_rows")}
            )
        notices: dict[str, CorrectionNotice] = {}
        for doc_id, tables in tables_by_doc.items():
            notice = extract_correction_notice(doc_id, tables)
            if notice is not None:
                notices[doc_id] = notice
        return notices

    def build_or_backfill_graph(
        self, *, table_limit: int = DEFAULT_NOTICE_TABLE_LIMIT
    ) -> dict[str, Any]:
        """Rebuild the graph from the frozen tables and persist the result."""

        records = self.load_disclosure_records()
        notices = self.load_correction_notices(table_limit=table_limit)
        graph = build_correction_graph(records, notices)
        # Every disclosure was loaded, so stale rows anywhere are safe to drop.
        written = self.persist_graph(graph, scope_doc_ids=None)
        return {
            "disclosure_count": len(records),
            "notice_count": len(notices),
            **written,
            **graph.diagnostics(),
        }

    def persist_graph(
        self,
        graph: CorrectionGraph,
        *,
        scope_doc_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Upsert the computed graph inside one transaction.

        ``scope_doc_ids`` states which disclosures this graph was computed over
        and is the safety boundary for deletion.

        * ``None`` means the graph covers every disclosure, so any stored row
          the graph no longer contains is stale and is removed.
        * A set of ids means the graph covers only those disclosures.  Rows
          outside the set are never touched, so a build over part of the corpus
          cannot delete correction relations it never looked at.

        Deletions run first; then ``is_latest`` is cleared across exactly the
        groups being written, so the unique one-latest-per-group index cannot be
        violated part-way through the upsert.  ``created_at`` survives a
        rebuild, ``updated_at`` does not.
        """

        relation_ids = [relation.relation_id for relation in graph.relations]
        member_ids = [member.doc_id for member in graph.members]
        group_ids = sorted({member.correction_group_id for member in graph.members})
        scope = sorted({str(doc_id) for doc_id in scope_doc_ids}) if scope_doc_ids is not None else None
        full_rebuild = scope is None
        with self._expected_failures("persist_graph"):
            with self._backend.connection() as connection:
                with connection.cursor() as cursor:
                    if full_rebuild:
                        cursor.execute(
                            "DELETE FROM correction_relations "
                            "WHERE NOT (relation_id = ANY(%s))",
                            (relation_ids,),
                        )
                    else:
                        cursor.execute(
                            "DELETE FROM correction_relations "
                            "WHERE source_doc_id = ANY(%s) "
                            "AND NOT (relation_id = ANY(%s))",
                            (scope, relation_ids),
                        )
                    removed_relations = int(cursor.rowcount)
                    if full_rebuild:
                        cursor.execute(
                            "DELETE FROM correction_group_members "
                            "WHERE NOT (doc_id = ANY(%s))",
                            (member_ids,),
                        )
                    else:
                        cursor.execute(
                            "DELETE FROM correction_group_members "
                            "WHERE doc_id = ANY(%s) AND NOT (doc_id = ANY(%s))",
                            (scope, member_ids),
                        )
                    removed_members = int(cursor.rowcount)
                    # Clear the flag only where it is about to be rewritten, so a
                    # scoped write leaves untouched groups with their latest intact.
                    cursor.execute(
                        "UPDATE correction_group_members SET is_latest = false "
                        "WHERE is_latest "
                        "AND (doc_id = ANY(%s) OR correction_group_id = ANY(%s))",
                        (member_ids, group_ids),
                    )
                    if graph.relations:
                        cursor.executemany(
                            f"""
                            INSERT INTO correction_relations ({", ".join(_RELATION_COLUMNS)})
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (relation_id) DO UPDATE SET
                                source_doc_id = EXCLUDED.source_doc_id,
                                target_doc_id = EXCLUDED.target_doc_id,
                                relation_type = EXCLUDED.relation_type,
                                resolution_status = EXCLUDED.resolution_status,
                                resolution_source = EXCLUDED.resolution_source,
                                confidence = EXCLUDED.confidence,
                                evidence = EXCLUDED.evidence,
                                updated_at = now()
                            """,
                            [
                                (
                                    relation.relation_id,
                                    relation.source_doc_id,
                                    relation.target_doc_id,
                                    relation.relation_type or CORRECTION_RELATION_TYPE,
                                    relation.resolution_status,
                                    relation.resolution_source,
                                    float(relation.confidence),
                                    _json(relation.evidence),
                                )
                                for relation in graph.relations
                            ],
                        )
                    if graph.members:
                        cursor.executemany(
                            f"""
                            INSERT INTO correction_group_members ({", ".join(_MEMBER_COLUMNS)})
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (doc_id) DO UPDATE SET
                                correction_group_id = EXCLUDED.correction_group_id,
                                root_doc_id = EXCLUDED.root_doc_id,
                                parent_doc_id = EXCLUDED.parent_doc_id,
                                correction_order = EXCLUDED.correction_order,
                                is_latest = EXCLUDED.is_latest,
                                is_correction = EXCLUDED.is_correction,
                                resolution_status = EXCLUDED.resolution_status,
                                resolution_source = EXCLUDED.resolution_source,
                                confidence = EXCLUDED.confidence,
                                evidence = EXCLUDED.evidence,
                                updated_at = now()
                            """,
                            [
                                (
                                    member.doc_id,
                                    member.correction_group_id,
                                    member.root_doc_id,
                                    member.parent_doc_id,
                                    int(member.correction_order),
                                    bool(member.is_latest),
                                    bool(member.is_correction),
                                    member.resolution_status,
                                    member.resolution_source,
                                    float(member.confidence),
                                    _json(member.evidence),
                                )
                                for member in graph.members
                            ],
                        )
                connection.commit()
        return {
            "relations_written": len(graph.relations),
            "members_written": len(graph.members),
            "relations_removed": removed_relations,
            "members_removed": removed_members,
            "full_rebuild": full_rebuild,
            "scope_doc_count": len(scope) if scope is not None else None,
        }

    # ----------------------------------------------------------- diagnostics

    def diagnostics(self) -> dict[str, Any]:
        """Read-only invariants computed from the persisted graph."""

        graph = self.load_graph()
        report = correction_diagnostics(graph.members, graph.relations)
        report["duplicate_relation_count"] = int(
            self._fetch(
                """
                SELECT count(*) AS value FROM (
                    SELECT source_doc_id, relation_type, coalesce(target_doc_id, '')
                    FROM correction_relations
                    GROUP BY 1, 2, 3
                    HAVING count(*) > 1
                ) AS duplicated
                """
            )[0]["value"]
        )
        report["invalid_latest_group_count"] = int(
            self._fetch(
                """
                SELECT count(*) AS value FROM (
                    SELECT correction_group_id
                    FROM correction_group_members
                    GROUP BY correction_group_id
                    HAVING count(*) FILTER (WHERE is_latest) <> 1
                ) AS invalid
                """
            )[0]["value"]
        )
        return report
