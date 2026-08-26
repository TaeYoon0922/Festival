"""PostgreSQL repository for the deterministic corporate event timeline.

Every event-aware caller goes through this repository instead of writing its own
SQL, so the resolver, retrieval expansion, and any diagnostic tool see one
consistent answer.  The repository owns persistence and lookup only; the rules
that decide which filings belong to one contract live in
:mod:`app.reasoning.corporate_event_graph` and never touch a database.

The backfill is idempotent. Events are keyed by a deterministic ``event_id``
derived from their company/family/opening anchor, members by
``(event_id, doc_id)``, and relations by a deterministic ``relation_id``. A
rebuild upserts the current result and
deletes rows the current result no longer contains, so running it twice produces
the same rows with the same ``created_at``.

v1 persists a full-corpus rebuild only.  A partial build resolves the rest of the
corpus as if it did not exist, so scoping the write would silently delete correct
events; the builder refuses instead.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

from app.reasoning.corporate_event import (
    CorporateEvent,
    CorporateEventMember,
    CorporateEventRelation,
    CorporateEventState,
)
from app.reasoning.corporate_event_graph import (
    ContractDocument,
    CorporateEventGraph,
    CorporateEventGraphUnavailable,
    DisclosureRecord,
    build_corporate_event_graph,
    classify_contract_document,
    corporate_event_diagnostics,
    extract_contract_document,
)


#: Contract filings state their structured fields in the leading tables of the
#: filing; the default keeps a wide margin over the deepest one observed.
DEFAULT_TABLE_LIMIT = 16

#: Date fields enumeration may anchor on, mapped to the column each one means.
#: A whitelist rather than a format string: the column name reaches SQL by
#: lookup, so a caller can never steer it.
_ENUMERATION_DATE_FIELDS = {
    "opened_at": "e.opened_at",
    "event_date": "m.event_date",
}

_EVENT_COLUMNS = (
    "event_id",
    "corp_code",
    "event_family",
    "root_logical_key",
    "lifecycle_status",
    "resolution_status",
    "resolution_source",
    "opened_at",
    "closed_at",
    "member_count",
    "confidence",
    "evidence",
)
_MEMBER_COLUMNS = (
    "event_id",
    "doc_id",
    "canonical_doc_id",
    "member_role",
    "member_order",
    "event_date",
    "root_doc_id",
    "correction_group_id",
    "correction_resolution_status",
    "correction_chain",
    "is_correction",
    "confidence",
    "provenance",
    "evidence",
)
_RELATION_COLUMNS = (
    "relation_id",
    "event_id",
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


def _json_array(value: Any) -> str:
    return json.dumps(list(value or ()), ensure_ascii=False)


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)[:10] if not hasattr(value, "isoformat") else value.isoformat()


def _member(row: Mapping[str, Any]) -> CorporateEventMember:
    evidence = dict(row.get("evidence") or {})
    document = evidence.get("document") or {}
    return CorporateEventMember(
        event_id=str(row["event_id"]),
        corp_code=str(row.get("event_corp_code") or document.get("corp_code") or ""),
        doc_id=str(row["doc_id"]),
        canonical_doc_id=str(row["canonical_doc_id"]),
        member_role=str(row["member_role"]),
        member_order=int(row["member_order"]),
        event_date=_date(row.get("event_date")),
        root_doc_id=str(row["root_doc_id"]) if row.get("root_doc_id") else None,
        correction_group_id=(
            str(row["correction_group_id"]) if row.get("correction_group_id") else None
        ),
        correction_resolution_status=(
            str(row["correction_resolution_status"])
            if row.get("correction_resolution_status")
            else None
        ),
        correction_chain=tuple(str(item) for item in (row.get("correction_chain") or ())),
        is_correction=bool(row.get("is_correction")),
        confidence=float(row.get("confidence") or 0.0),
        provenance=dict(row.get("provenance") or {}),
        evidence=evidence,
    )


def _relation(row: Mapping[str, Any]) -> CorporateEventRelation:
    return CorporateEventRelation(
        relation_id=str(row["relation_id"]),
        source_doc_id=str(row["source_doc_id"]),
        relation_type=str(row["relation_type"]),
        target_doc_id=str(row["target_doc_id"]) if row.get("target_doc_id") else None,
        event_id=str(row["event_id"]) if row.get("event_id") else None,
        resolution_status=str(row["resolution_status"]),
        resolution_source=str(row["resolution_source"]),
        confidence=float(row.get("confidence") or 0.0),
        evidence=dict(row.get("evidence") or {}),
    )


def _event(row: Mapping[str, Any], members: Sequence[CorporateEventMember]) -> CorporateEvent:
    return CorporateEvent(
        event_id=str(row["event_id"]),
        corp_code=str(row["corp_code"]),
        event_family=str(row["event_family"]),
        root_logical_key=str(row["root_logical_key"]),
        lifecycle_status=str(row["lifecycle_status"]),
        resolution_status=str(row["resolution_status"]),
        resolution_source=str(row["resolution_source"]),
        members=tuple(sorted(members, key=lambda item: item.member_order)),
        opened_at=_date(row.get("opened_at")),
        closed_at=_date(row.get("closed_at")),
        confidence=float(row.get("confidence") or 0.0),
        evidence=dict(row.get("evidence") or {}),
    )


class PostgresCorporateEventRepository:
    """Build, persist, and read the corporate event timeline stored in PostgreSQL."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    # ------------------------------------------------------------------ read

    @contextmanager
    def _expected_failures(self, operation: str) -> Iterator[None]:
        """Translate only "the graph is not there" into a degradable error.

        ``UndefinedTable`` means db/007 has not been applied; an
        ``OperationalError`` means the database is unreachable.  Every other
        psycopg error keeps its own type and propagates, so a real defect is
        never disguised as a missing migration.
        """

        try:
            yield
        except psycopg.errors.UndefinedTable as error:
            raise CorporateEventGraphUnavailable(
                f"corporate event tables are missing during {operation}; "
                "apply db/007_corporate_event_timeline.sql"
            ) from error
        except psycopg.OperationalError as error:
            raise CorporateEventGraphUnavailable(
                f"database unreachable during {operation}"
            ) from error

    def _fetch(
        self, query: str, params: Sequence[Any] = (), *, operation: str = "read"
    ) -> list[dict[str, Any]]:
        with self._expected_failures(operation):
            with self._backend.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(query, params)
                    return [dict(row) for row in cursor.fetchall()]

    #: Matches a member row by its own id *or* by any raw filing it stands for.
    #:
    #: Membership is stored per logical contract, so the original of a
    #: correction chain has no row of its own.  A question can still be anchored
    #: on that original, so lookup resolves through the chain and the collapsed
    #: id list the row already carries.  This is a read-side alias only: no raw
    #: filing is ever written back as a second member row.
    _ALIAS_PREDICATE = """(
        m.doc_id = %s
        OR m.correction_chain ? %s
        OR m.provenance -> 'collapsed_doc_ids' ? %s
    )"""

    @staticmethod
    def _alias_params(doc_id: str, *extra: Any) -> list[Any]:
        value = str(doc_id)
        return [value, value, value, *extra]

    def get_event(self, doc_id: str) -> CorporateEvent | None:
        """The lifecycle this document belongs to, with every member.

        Accepts a superseded filing as readily as the representative one.
        """

        rows = self._fetch(
            f"""
            SELECT DISTINCT e.*
            FROM corporate_events e
            JOIN corporate_event_members m ON m.event_id = e.event_id
            WHERE {self._ALIAS_PREDICATE}
            """,
            self._alias_params(doc_id),
            operation="get_event",
        )
        if not rows:
            return None
        return _event(rows[0], self._members_of(str(rows[0]["event_id"])))

    def get_member(self, doc_id: str) -> CorporateEventMember | None:
        """The logical member this filing belongs to, collapsed versions included."""

        rows = self._fetch(
            f"""
            SELECT m.*, e.corp_code AS event_corp_code
            FROM corporate_event_members m
            JOIN corporate_events e ON e.event_id = m.event_id
            WHERE {self._ALIAS_PREDICATE}
            ORDER BY (m.doc_id = %s) DESC, m.member_order
            LIMIT 1
            """,
            self._alias_params(doc_id, str(doc_id)),
            operation="get_member",
        )
        return _member(rows[0]) if rows else None

    def get_event_by_id(self, event_id: str) -> CorporateEvent | None:
        rows = self._fetch(
            "SELECT * FROM corporate_events WHERE event_id = %s",
            [str(event_id)],
            operation="get_event_by_id",
        )
        if not rows:
            return None
        return _event(rows[0], self._members_of(str(event_id)))

    def _members_of(self, event_id: str) -> list[CorporateEventMember]:
        return [
            _member(row)
            for row in self._fetch(
                "SELECT m.*, e.corp_code AS event_corp_code "
                "FROM corporate_event_members m "
                "JOIN corporate_events e ON e.event_id = m.event_id "
                "WHERE m.event_id = %s "
                "ORDER BY member_order, doc_id",
                [str(event_id)],
                operation="event_members",
            )
        ]

    def get_event_timeline(self, doc_id: str) -> tuple[CorporateEventMember, ...]:
        """Every filing of this document's lifecycle, oldest first."""

        rows = self._fetch(
            f"""
            SELECT m.*, e.corp_code AS event_corp_code
            FROM corporate_event_members m
            JOIN corporate_events e ON e.event_id = m.event_id
            WHERE m.event_id = (
                SELECT m2.event_id FROM corporate_event_members m2
                WHERE {self._ALIAS_PREDICATE.replace("m.", "m2.")}
                LIMIT 1
            )
            ORDER BY m.member_order, m.doc_id
            """,
            self._alias_params(doc_id),
            operation="get_event_timeline",
        )
        return tuple(_member(row) for row in rows)

    def get_related_documents(self, doc_id: str) -> tuple[str, ...]:
        """The other logical members of this lifecycle, in timeline order.

        A superseded filing is answered as its representative, so asking about
        the original of a chain never lists that chain's own latest version as a
        "related" document.
        """

        own = self.get_member(doc_id)
        own_doc_id = own.doc_id if own is not None else str(doc_id)
        return tuple(
            member.doc_id
            for member in self.get_event_timeline(doc_id)
            if member.doc_id != own_doc_id
        )

    def get_event_state(self, doc_id: str) -> CorporateEventState | None:
        states = self.event_states([doc_id])
        return states.get(str(doc_id))

    def event_states(
        self, doc_ids: Iterable[str]
    ) -> dict[str, CorporateEventState]:
        """One compact row per document, for gating retrieval expansion."""

        unique = sorted({str(doc_id) for doc_id in doc_ids if str(doc_id)})
        if not unique:
            return {}
        # ``asked_doc_id`` keeps the caller's own identifier as the key while the
        # state itself describes the logical member, so a seed that happens to be
        # a superseded filing is still answered.
        rows = self._fetch(
            """
            SELECT
                asked.doc_id AS asked_doc_id,
                m.doc_id, m.event_id, m.member_role, m.canonical_doc_id,
                m.correction_group_id, m.correction_resolution_status,
                e.corp_code, e.event_family, e.lifecycle_status,
                e.resolution_status, e.member_count
            FROM unnest(%s::text[]) AS asked(doc_id)
            JOIN corporate_event_members m
              ON m.doc_id = asked.doc_id
              OR m.correction_chain ? asked.doc_id
              OR m.provenance -> 'collapsed_doc_ids' ? asked.doc_id
            JOIN corporate_events e ON e.event_id = m.event_id
            """,
            [unique],
            operation="event_states",
        )
        return {
            str(row["asked_doc_id"]): CorporateEventState(
                doc_id=str(row["doc_id"]),
                event_id=str(row["event_id"]),
                corp_code=str(row["corp_code"]),
                event_family=str(row["event_family"]),
                member_role=str(row["member_role"]),
                lifecycle_status=str(row["lifecycle_status"]),
                resolution_status=str(row["resolution_status"]),
                canonical_doc_id=str(row["canonical_doc_id"]),
                member_count=int(row["member_count"]),
                correction_group_id=(
                    str(row["correction_group_id"])
                    if row.get("correction_group_id")
                    else None
                ),
                correction_resolution_status=(
                    str(row["correction_resolution_status"])
                    if row.get("correction_resolution_status")
                    else None
                ),
            )
            for row in rows
        }

    def enumerate_events(
        self,
        *,
        corp_code: str,
        event_family: str,
        member_role: str = "contract",
        date_field: str = "opened_at",
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[CorporateEventState, ...]:
        """List one company's logical lifecycles inside a half-open date range.

        This is the Tier 1 enumeration primitive: it answers "which contracts
        exist", not "which chunks look relevant".  The event timeline is already
        one row per lifecycle with P0-A corrections collapsed into it, so the
        count returned here is the *logical* count.  Counting raw filings
        instead would roughly double it -- half of the supply-contract filings in
        this corpus are ``[기재정정]`` corrections of an earlier one.

        ``member_role`` selects the opening filing by default, which is what
        keeps a termination from being counted as a contract: a terminated
        lifecycle still contributes exactly one ``contract`` member, and its
        termination is reported through ``lifecycle_status`` rather than as a
        second list entry.

        ``date_from``/``date_to`` are half-open (``>= from``, ``< to``) so that
        adjacent years partition the corpus without overlapping on 12-31.  The
        default ``date_field`` is the event's own ``opened_at`` -- the date the
        filing puts on the contract -- because "2025년에 체결한 계약" asks when the
        contract was signed, not when the disclosure was received.  The two
        disagree for a large minority of this corpus.

        One statement, no per-seed round trip.
        """

        if not str(corp_code).strip():
            raise ValueError("corp_code is required for enumeration")
        if not str(event_family).strip():
            raise ValueError("event_family is required for enumeration")
        if date_field not in _ENUMERATION_DATE_FIELDS:
            raise ValueError(
                "date_field must be one of "
                + ", ".join(sorted(_ENUMERATION_DATE_FIELDS))
            )
        if date_from and date_to and str(date_from) > str(date_to):
            raise ValueError("date_from must not be after date_to")

        # Whitelisted above, never interpolated from caller input directly.
        date_column = _ENUMERATION_DATE_FIELDS[date_field]
        conditions = ["e.corp_code = %s", "e.event_family = %s", "m.member_role = %s"]
        params: list[Any] = [str(corp_code), str(event_family), str(member_role)]
        if date_from:
            conditions.append(f"{date_column} >= %s::date")
            params.append(str(date_from))
        if date_to:
            conditions.append(f"{date_column} < %s::date")
            params.append(str(date_to))

        rows = self._fetch(
            f"""
            SELECT
                m.doc_id, m.event_id, m.member_role, m.canonical_doc_id,
                m.correction_group_id, m.correction_resolution_status,
                e.corp_code, e.event_family, e.lifecycle_status,
                e.resolution_status, e.resolution_source, e.opened_at,
                e.member_count
            FROM corporate_events e
            JOIN corporate_event_members m ON m.event_id = e.event_id
            WHERE {" AND ".join(conditions)}
            ORDER BY e.opened_at NULLS LAST, e.event_id, m.member_order, m.doc_id
            """,
            params,
            operation="enumerate_events",
        )
        return tuple(
            CorporateEventState(
                doc_id=str(row["doc_id"]),
                event_id=str(row["event_id"]),
                corp_code=str(row["corp_code"]),
                event_family=str(row["event_family"]),
                member_role=str(row["member_role"]),
                lifecycle_status=str(row["lifecycle_status"]),
                resolution_status=str(row["resolution_status"]),
                canonical_doc_id=str(row["canonical_doc_id"]),
                member_count=int(row["member_count"]),
                correction_group_id=(
                    str(row["correction_group_id"])
                    if row.get("correction_group_id")
                    else None
                ),
                correction_resolution_status=(
                    str(row["correction_resolution_status"])
                    if row.get("correction_resolution_status")
                    else None
                ),
                opened_at=_date(row.get("opened_at")),
                resolution_source=(
                    str(row["resolution_source"])
                    if row.get("resolution_source")
                    else None
                ),
            )
            for row in rows
        )

    def load_graph(self) -> CorporateEventGraph:
        """Materialize the persisted graph for in-process reuse."""

        members_by_event: dict[str, list[CorporateEventMember]] = {}
        for row in self._fetch(
            "SELECT m.*, e.corp_code AS event_corp_code "
            "FROM corporate_event_members m "
            "JOIN corporate_events e ON e.event_id = m.event_id "
            "ORDER BY m.event_id, m.member_order, m.doc_id"
        ):
            member = _member(row)
            members_by_event.setdefault(member.event_id, []).append(member)
        events = [
            _event(row, members_by_event.get(str(row["event_id"]), []))
            for row in self._fetch("SELECT * FROM corporate_events ORDER BY event_id")
        ]
        relations = [
            _relation(row)
            for row in self._fetch(
                "SELECT * FROM corporate_event_relations ORDER BY relation_id"
            )
        ]
        return CorporateEventGraph(events, relations)

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

    def load_contract_documents(
        self,
        records: Sequence[DisclosureRecord] | None = None,
        *,
        table_limit: int = DEFAULT_TABLE_LIMIT,
    ) -> dict[str, ContractDocument]:
        """Read the structured contract fields out of each v1 contract filing."""

        if table_limit <= 0:
            raise ValueError("table_limit must be positive")
        by_doc = {
            record.doc_id: record
            for record in (records if records is not None else self.load_disclosure_records())
            if classify_contract_document(record) is not None
        }
        if not by_doc:
            return {}
        rows = self._fetch(
            """
            SELECT t.doc_id, t.table_id, t.table_rows
            FROM disclosure_tables t
            WHERE t.doc_id = ANY(%s)
              AND t.table_order < %s
            ORDER BY t.doc_id, t.source_part_id, t.table_order
            """,
            [sorted(by_doc), int(table_limit)],
            operation="load_contract_documents",
        )
        tables_by_doc: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            tables_by_doc.setdefault(str(row["doc_id"]), []).append(
                {"table_id": row.get("table_id"), "table_rows": row.get("table_rows")}
            )
        documents: dict[str, ContractDocument] = {}
        for doc_id, tables in tables_by_doc.items():
            document = extract_contract_document(by_doc[doc_id], tables)
            if document is not None:
                documents[doc_id] = document
        return documents

    def build_or_backfill_graph(
        self,
        *,
        table_limit: int = DEFAULT_TABLE_LIMIT,
        correction_graph: Any | None = None,
    ) -> dict[str, Any]:
        """Rebuild the timeline from the frozen tables and persist the result."""

        if correction_graph is None:
            from app.retrieval.correction_repository import PostgresCorrectionRepository

            correction_graph = PostgresCorrectionRepository(self._backend).load_graph()
        records = self.load_disclosure_records()
        documents = self.load_contract_documents(records, table_limit=table_limit)
        graph = build_corporate_event_graph(
            records, documents, correction_graph=correction_graph
        )
        written = self.persist_graph(graph)
        return {
            "disclosure_count": len(records),
            "contract_document_count": len(documents),
            **written,
            **graph.diagnostics(),
        }

    def persist_graph(self, graph: CorporateEventGraph) -> dict[str, Any]:
        """Upsert the computed timeline inside one transaction.

        A full-corpus rebuild is the only supported write, so any stored row the
        graph no longer contains is stale and is removed.  Children are deleted
        before parents and inserted after them, and a document that moved to a
        different event has its old membership row removed in the same statement
        pass, so the one-event-per-document index cannot be violated part-way
        through.  ``created_at`` survives a rebuild, ``updated_at`` does not.
        """

        event_ids = [event.event_id for event in graph.events]
        members = list(graph.members)
        member_keys = [f"{member.event_id}\x1f{member.doc_id}" for member in members]
        relation_ids = [relation.relation_id for relation in graph.relations]

        with self._expected_failures("persist_graph"):
            with self._backend.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM corporate_event_relations "
                        "WHERE NOT (relation_id = ANY(%s))",
                        (relation_ids,),
                    )
                    removed_relations = int(cursor.rowcount)
                    cursor.execute(
                        "DELETE FROM corporate_event_members "
                        "WHERE NOT (event_id || chr(31) || doc_id = ANY(%s))",
                        (member_keys,),
                    )
                    removed_members = int(cursor.rowcount)
                    cursor.execute(
                        "DELETE FROM corporate_events WHERE NOT (event_id = ANY(%s))",
                        (event_ids,),
                    )
                    removed_events = int(cursor.rowcount)
                    if graph.events:
                        cursor.executemany(
                            f"""
                            INSERT INTO corporate_events ({", ".join(_EVENT_COLUMNS)})
                            VALUES (
                                %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s::jsonb
                            )
                            ON CONFLICT (event_id) DO UPDATE SET
                                corp_code = EXCLUDED.corp_code,
                                event_family = EXCLUDED.event_family,
                                root_logical_key = EXCLUDED.root_logical_key,
                                lifecycle_status = EXCLUDED.lifecycle_status,
                                resolution_status = EXCLUDED.resolution_status,
                                resolution_source = EXCLUDED.resolution_source,
                                opened_at = EXCLUDED.opened_at,
                                closed_at = EXCLUDED.closed_at,
                                member_count = EXCLUDED.member_count,
                                confidence = EXCLUDED.confidence,
                                evidence = EXCLUDED.evidence,
                                updated_at = now()
                            """,
                            [
                                (
                                    event.event_id,
                                    event.corp_code,
                                    _value(event.event_family),
                                    event.root_logical_key,
                                    _value(event.lifecycle_status),
                                    _value(event.resolution_status),
                                    event.resolution_source,
                                    event.opened_at,
                                    event.closed_at,
                                    event.member_count,
                                    float(event.confidence),
                                    _json(event.evidence),
                                )
                                for event in graph.events
                            ],
                        )
                    if members:
                        cursor.executemany(
                            f"""
                            INSERT INTO corporate_event_members ({", ".join(_MEMBER_COLUMNS)})
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb
                            )
                            ON CONFLICT (event_id, doc_id) DO UPDATE SET
                                canonical_doc_id = EXCLUDED.canonical_doc_id,
                                member_role = EXCLUDED.member_role,
                                member_order = EXCLUDED.member_order,
                                event_date = EXCLUDED.event_date,
                                root_doc_id = EXCLUDED.root_doc_id,
                                correction_group_id = EXCLUDED.correction_group_id,
                                correction_resolution_status =
                                    EXCLUDED.correction_resolution_status,
                                correction_chain = EXCLUDED.correction_chain,
                                is_correction = EXCLUDED.is_correction,
                                confidence = EXCLUDED.confidence,
                                provenance = EXCLUDED.provenance,
                                evidence = EXCLUDED.evidence,
                                updated_at = now()
                            """,
                            [
                                (
                                    member.event_id,
                                    member.doc_id,
                                    member.canonical_doc_id,
                                    _value(member.member_role),
                                    int(member.member_order),
                                    member.event_date,
                                    member.root_doc_id,
                                    member.correction_group_id,
                                    _value(member.correction_resolution_status),
                                    _json_array(member.correction_chain),
                                    bool(member.is_correction),
                                    float(member.confidence),
                                    _json(member.provenance),
                                    _json(member.evidence),
                                )
                                for member in members
                            ],
                        )
                    if graph.relations:
                        cursor.executemany(
                            f"""
                            INSERT INTO corporate_event_relations ({", ".join(_RELATION_COLUMNS)})
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                            ON CONFLICT (relation_id) DO UPDATE SET
                                event_id = EXCLUDED.event_id,
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
                                    relation.event_id,
                                    relation.source_doc_id,
                                    relation.target_doc_id,
                                    _value(relation.relation_type),
                                    _value(relation.resolution_status),
                                    relation.resolution_source,
                                    float(relation.confidence),
                                    _json(relation.evidence),
                                )
                                for relation in graph.relations
                            ],
                        )
                connection.commit()
        return {
            "events_written": len(graph.events),
            "members_written": len(members),
            "relations_written": len(graph.relations),
            "events_removed": removed_events,
            "members_removed": removed_members,
            "relations_removed": removed_relations,
        }

    # ----------------------------------------------------------- diagnostics

    def diagnostics(self) -> dict[str, Any]:
        """Read-only invariants computed from the persisted graph."""

        graph = self.load_graph()
        report = corporate_event_diagnostics(graph.events, graph.relations)
        # How many raw filings the stored rows answer for beyond their own id.
        # This is the DB-side counterpart of the in-memory alias index, and it is
        # what makes "990 rows cover 1,216 documents" checkable from SQL alone.
        report["raw_alias_count"] = int(
            self._fetch(
                """
                SELECT count(DISTINCT alias) AS value
                FROM corporate_event_members m,
                     LATERAL (
                         SELECT jsonb_array_elements_text(
                             m.correction_chain
                             || coalesce(
                                 m.provenance -> 'collapsed_doc_ids', '[]'::jsonb
                             )
                         ) AS alias
                     ) AS aliases
                WHERE alias <> m.doc_id
                """,
                operation="diagnostics",
            )[0]["value"]
        )
        report["stored_event_count"] = int(
            self._fetch("SELECT count(*) AS value FROM corporate_events")[0]["value"]
        )
        report["stored_member_count"] = int(
            self._fetch("SELECT count(*) AS value FROM corporate_event_members")[0][
                "value"
            ]
        )
        report["stored_relation_count"] = int(
            self._fetch("SELECT count(*) AS value FROM corporate_event_relations")[0][
                "value"
            ]
        )
        report["duplicate_membership_count"] = int(
            self._fetch(
                """
                SELECT count(*) AS value FROM (
                    SELECT doc_id FROM corporate_event_members
                    GROUP BY doc_id HAVING count(*) > 1
                ) AS duplicated
                """
            )[0]["value"]
        )
        report["duplicate_relation_count"] = int(
            self._fetch(
                """
                SELECT count(*) AS value FROM (
                    SELECT source_doc_id, relation_type, coalesce(target_doc_id, '')
                    FROM corporate_event_relations
                    GROUP BY 1, 2, 3
                    HAVING count(*) > 1
                ) AS duplicated
                """
            )[0]["value"]
        )
        # Computed against ``disclosures`` rather than the in-memory graph, so a
        # row written by anything other than this builder is still caught.
        report["cross_company_member_count"] = int(
            self._fetch(
                """
                SELECT count(*) AS value
                FROM corporate_event_members m
                JOIN corporate_events e ON e.event_id = m.event_id
                JOIN disclosures d ON d.doc_id = m.doc_id
                WHERE d.corp_code <> e.corp_code
                """
            )[0]["value"]
        )
        report["cross_company_relation_count"] = int(
            self._fetch(
                """
                SELECT count(*) AS value
                FROM corporate_event_relations r
                JOIN disclosures s ON s.doc_id = r.source_doc_id
                JOIN disclosures t ON t.doc_id = r.target_doc_id
                WHERE s.corp_code <> t.corp_code
                """
            )[0]["value"]
        )
        return report
