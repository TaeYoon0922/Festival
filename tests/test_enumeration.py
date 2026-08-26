"""P0-C Step 2: deterministic enumeration.

The local Gold60 set contains no enumeration question at all -- it is 60 single
document lookups -- so it cannot validate this capability.  The official task
does require it ("2025년에 체결한 주요 계약 중 이후 해지된 계약", "유형별로 정리",
"연도별 변화"), so the positive cases here are synthetic fixtures built to the
shape of those official questions, exercised through the *frozen* P0-A and P0-B
builders rather than through re-implemented matching.

S1  enumeration               3 independent contracts   -> 3
S2  correction collapse       A + 2 corrections         -> 1
S3  termination exclusion     contract + termination    -> 1
S4  two contracts             independent A, B          -> 2
S5  date boundary             signed 2024-12-28,
                              received 2025-01-02       -> not in the 2025 set
S6  ambiguous correction      unverified chain          -> not folded
S7  treasury trust            trust enumeration
S8  enumeration + lifecycle   [open, terminated, open]
"""

from __future__ import annotations

import unittest

from app.reasoning.corporate_event import CorporateEventState
from app.reasoning.corporate_event_graph import (
    FAMILY_SUPPLY_CONTRACT,
    FAMILY_TREASURY_TRUST,
    ContractDocument,
    DisclosureRecord,
    build_corporate_event_graph,
    parse_related_disclosures,
)
from app.reasoning.correction_graph import (
    AMBIGUOUS,
    RESOLVED,
    CorrectionGraph,
    CorrectionGroupMember,
)
from app.retrieval.corporate_event_repository import PostgresCorporateEventRepository
from app.retrieval.enumeration import (
    ORIGIN_RESOLVED_GROUP,
    ORIGIN_STANDALONE,
    ORIGIN_UNVERIFIED_CORRECTION,
    collapse_logical_documents,
)
from app.retrieval.postgres_backend import PostgresBackend


CORP = "00123456"
SUPPLY_CONCLUSION = "단일판매공급계약체결"
SUPPLY_TERMINATION = "단일판매공급계약해지"
TRUST_CONCLUSION_NM = "주요사항보고서(자기주식취득신탁계약체결결정)"
TRUST_TERMINATION_NM = "주요사항보고서(자기주식취득신탁계약해지결정)"
REF_TITLE = "단일판매ㆍ공급계약체결"


# --------------------------------------------------------------- fixture data


def _record(
    doc_id,
    rcept_dt,
    *,
    corp_code=CORP,
    doc_group="exchange",
    doc_subtype=SUPPLY_CONCLUSION,
    report_nm="단일판매ㆍ공급계약체결",
    is_correction=False,
):
    return DisclosureRecord(
        doc_id=doc_id,
        corp_code=corp_code,
        doc_group=doc_group,
        report_nm=report_nm,
        rcept_no=doc_id,
        rcept_dt=rcept_dt,
        doc_subtype=doc_subtype,
        is_correction=is_correction,
    )


def _conclusion(
    doc_id,
    rcept_dt,
    *,
    corp_code=CORP,
    counterparty="Acme Corp",
    subject="Widget supply",
    amount="1,000,000",
    contract_date=None,
    period_start="2024-01-01",
    period_end="2029-12-31",
):
    return ContractDocument(
        doc_id=doc_id,
        corp_code=corp_code,
        event_family=FAMILY_SUPPLY_CONTRACT,
        kind="conclusion",
        rcept_dt=rcept_dt,
        counterparty=counterparty,
        subject=subject,
        amount=amount,
        contract_date=contract_date,
        period_start=period_start,
        period_end=period_end,
    )


def _termination(doc_id, rcept_dt, references, **kwargs):
    raw = " ".join(f"{date} {title}" for date, title in references)
    return ContractDocument(
        doc_id=doc_id,
        corp_code=kwargs.get("corp_code", CORP),
        event_family=FAMILY_SUPPLY_CONTRACT,
        kind="termination",
        rcept_dt=rcept_dt,
        counterparty=kwargs.get("counterparty", "Acme Corp"),
        subject=kwargs.get("subject", "Widget supply"),
        amount=kwargs.get("amount", "1,000,000"),
        period_start=kwargs.get("period_start", "2024-01-01"),
        period_end=kwargs.get("period_end", "2029-12-31"),
        termination_date=rcept_dt,
        termination_reason="계약상대방의 계약이행 불가",
        related=parse_related_disclosures(raw),
    )


def _trust_record(doc_id, rcept_dt, kind, *, corp_code=CORP):
    return _record(
        doc_id,
        rcept_dt,
        corp_code=corp_code,
        doc_group="major",
        doc_subtype=None,
        report_nm=TRUST_CONCLUSION_NM if kind == "conclusion" else TRUST_TERMINATION_NM,
    )


def _trust(doc_id, rcept_dt, kind, *, period_start="2024-01-01", period_end="2024-07-01"):
    return ContractDocument(
        doc_id=doc_id,
        corp_code=CORP,
        event_family=FAMILY_TREASURY_TRUST,
        kind=kind,
        rcept_dt=rcept_dt,
        period_start=period_start,
        period_end=period_end,
        termination_date=rcept_dt if kind == "termination" else None,
    )


def _build(records, documents, correction_graph=None):
    return build_corporate_event_graph(
        list(records),
        {document.doc_id: document for document in documents},
        correction_graph=correction_graph,
    )


def _resolved_chain(group_id, doc_ids):
    """A P0-A resolved correction group: original plus verified corrections."""

    last = len(doc_ids) - 1
    return [
        CorrectionGroupMember(
            doc_id=doc_id,
            correction_group_id=group_id,
            root_doc_id=doc_ids[0],
            parent_doc_id=doc_ids[index - 1] if index else None,
            correction_order=index,
            is_latest=index == last,
            resolution_status=RESOLVED,
            resolution_source="correction_notice",
            confidence=1.0,
            is_correction=index > 0,
        )
        for index, doc_id in enumerate(doc_ids)
    ]


def _openings(graph, family=FAMILY_SUPPLY_CONTRACT):
    """Logical opening events -- what Tier 1 enumeration returns."""

    return [
        event
        for event in graph.events
        if event.event_family == family
        and any(member.member_role == "contract" for member in event.members)
    ]


# ------------------------------------------------------- official-style cases


class OfficialStyleEnumerationTests(unittest.TestCase):
    """S1-S8: the counting semantics the official questions depend on."""

    def test_s1_three_independent_contracts_enumerate_as_three(self) -> None:
        records = [_record(f"c{i}", f"2025-0{i}-10") for i in (1, 2, 3)]
        documents = [
            _conclusion(f"c{i}", f"2025-0{i}-10", subject=f"Job {i}", contract_date=f"2025-0{i}-10")
            for i in (1, 2, 3)
        ]
        graph = _build(records, documents)
        self.assertEqual(len(_openings(graph)), 3)

    def test_s2_correction_chain_counts_as_one_contract(self) -> None:
        """Raw 3 filings, one logical contract.

        Half of this corpus's supply-contract filings are corrections, so
        counting raw disclosures would roughly double every answer.
        """

        doc_ids = ["a0", "a1", "a2"]
        records = [
            _record("a0", "2025-03-10"),
            _record("a1", "2025-04-10", is_correction=True),
            _record("a2", "2025-05-10", is_correction=True),
        ]
        documents = [
            _conclusion(doc_id, dt, contract_date="2025-03-10")
            for doc_id, dt in zip(doc_ids, ["2025-03-10", "2025-04-10", "2025-05-10"])
        ]
        graph = _build(
            records,
            documents,
            correction_graph=CorrectionGraph(_resolved_chain("g1", doc_ids)),
        )
        openings = _openings(graph)
        self.assertEqual(len(openings), 1)
        # The raw filings are folded into the one logical member, not dropped.
        self.assertEqual(openings[0].member_count, 1)

    def test_s3_termination_is_not_a_second_contract(self) -> None:
        records = [
            _record("a0", "2025-03-10"),
            _record("t0", "2025-09-10", doc_subtype=SUPPLY_TERMINATION,
                    report_nm="단일판매ㆍ공급계약해지"),
        ]
        documents = [
            _conclusion("a0", "2025-03-10", contract_date="2025-03-10"),
            _termination("t0", "2025-09-10", (("2025-03-10", REF_TITLE),)),
        ]
        graph = _build(records, documents)
        openings = _openings(graph)
        self.assertEqual(len(openings), 1)
        contracts = [
            member
            for event in openings
            for member in event.members
            if member.member_role == "contract"
        ]
        self.assertEqual(len(contracts), 1)
        self.assertEqual(openings[0].lifecycle_status, "terminated")

    def test_s4_two_independent_contracts_are_not_merged(self) -> None:
        records = [_record("a0", "2025-03-10"), _record("b0", "2025-06-10")]
        documents = [
            _conclusion("a0", "2025-03-10", counterparty="Acme Corp",
                        subject="Widget supply", contract_date="2025-03-10"),
            _conclusion("b0", "2025-06-10", counterparty="Beta Ltd",
                        subject="Gadget supply", contract_date="2025-06-10"),
        ]
        graph = _build(records, documents)
        self.assertEqual(len(_openings(graph)), 2)

    def test_s5_contract_signed_in_december_is_not_in_the_next_year_set(self) -> None:
        """The boundary case a receipt-date filter gets wrong.

        Signed 2024-12-28, filed 2025-01-02.  ``opened_at`` is the signing date,
        so a half-open 2025 window must exclude it.
        """

        records = [_record("a0", "2025-01-02")]
        documents = [_conclusion("a0", "2025-01-02", contract_date="2024-12-28")]
        graph = _build(records, documents)
        opening = _openings(graph)[0]
        self.assertEqual(str(opening.opened_at), "2024-12-28")
        self.assertNotEqual(str(opening.opened_at)[:4], "2025")
        # Half-open [2025-01-01, 2026-01-01) excludes it; the receipt year does not.
        self.assertFalse("2025-01-01" <= str(opening.opened_at) < "2026-01-01")

    def test_s6_ambiguous_correction_is_not_folded(self) -> None:
        """P0-A could not verify the chain, so the filings stay separate."""

        members = [
            CorrectionGroupMember(
                doc_id="a1",
                correction_group_id="g_amb",
                root_doc_id="a1",
                parent_doc_id=None,
                correction_order=0,
                is_latest=True,          # set even though nothing was verified
                resolution_status=AMBIGUOUS,
                resolution_source="multiple_candidates",
                confidence=0.0,
                is_correction=True,
            )
        ]
        records = [
            _record("a0", "2025-03-10"),
            _record("a1", "2025-04-10", is_correction=True),
        ]
        documents = [
            _conclusion("a0", "2025-03-10", subject="Widget supply",
                        contract_date="2025-03-10"),
            _conclusion("a1", "2025-04-10", subject="Different job",
                        counterparty="Beta Ltd", contract_date="2025-04-10"),
        ]
        graph = _build(records, documents, correction_graph=CorrectionGraph(members))
        self.assertEqual(len(_openings(graph)), 2)

    def test_s7_treasury_trust_enumerates_independently(self) -> None:
        records = [
            _trust_record("t1", "2024-01-05", "conclusion"),
            _trust_record("t2", "2024-08-05", "conclusion"),
        ]
        documents = [
            _trust("t1", "2024-01-05", "conclusion",
                   period_start="2024-01-05", period_end="2024-07-05"),
            _trust("t2", "2024-08-05", "conclusion",
                   period_start="2024-08-05", period_end="2025-02-05"),
        ]
        graph = _build(records, documents)
        trust = _openings(graph, family=FAMILY_TREASURY_TRUST)
        self.assertEqual(len(trust), 2)
        self.assertEqual(len(_openings(graph)), 0)

    def test_s8_enumeration_plus_lifecycle_is_answerable_from_one_set(self) -> None:
        """[A open, B terminated, C open] -- exactly the official question shape."""

        records = [
            _record("a0", "2025-02-10"),
            _record("b0", "2025-04-10"),
            _record("c0", "2025-06-10"),
            _record("tb", "2025-11-10", doc_subtype=SUPPLY_TERMINATION,
                    report_nm="단일판매ㆍ공급계약해지"),
        ]
        documents = [
            _conclusion("a0", "2025-02-10", counterparty="Acme Corp",
                        subject="Widget supply", contract_date="2025-02-10"),
            _conclusion("b0", "2025-04-10", counterparty="Beta Ltd",
                        subject="Gadget supply", contract_date="2025-04-10"),
            _conclusion("c0", "2025-06-10", counterparty="Gamma Inc",
                        subject="Sprocket supply", contract_date="2025-06-10"),
            _termination("tb", "2025-11-10", (("2025-04-10", REF_TITLE),),
                         counterparty="Beta Ltd", subject="Gadget supply"),
        ]
        graph = _build(records, documents)
        openings = _openings(graph)
        self.assertEqual(len(openings), 3)
        lifecycles = sorted(event.lifecycle_status for event in openings)
        self.assertEqual(lifecycles, ["open", "open", "terminated"])
        # "있는가" is answerable, and "which one" is too, without any search.
        terminated = [e for e in openings if e.lifecycle_status == "terminated"]
        self.assertEqual(len(terminated), 1)
        self.assertIn(
            "b0",
            {m.doc_id for m in terminated[0].members if m.member_role == "contract"},
        )


# ------------------------------------------------------------- Tier 1 SQL


class _RecordingConnection:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, row_factory=None):
        return _RecordingCursor(self._store)


class _RecordingCursor:
    def __init__(self, store):
        self._store = store
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=()):
        self._store.queries.append((query, list(params)))
        self._rows = self._store.rows_for(query, params)
        return self

    def fetchall(self):
        return self._rows


class _RecordingBackend:
    def __init__(self, rows=None):
        self.queries = []
        self._rows = rows or []

    def rows_for(self, query, params):
        return self._rows

    def connection(self):
        return _RecordingConnection(self)


def _event_row(**overrides):
    row = {
        "doc_id": "a0",
        "event_id": "evt_1",
        "member_role": "contract",
        "canonical_doc_id": "a0",
        "correction_group_id": None,
        "correction_resolution_status": None,
        "corp_code": CORP,
        "event_family": FAMILY_SUPPLY_CONTRACT,
        "lifecycle_status": "open",
        "resolution_status": "unresolved",
        "resolution_source": "single_document",
        "opened_at": "2025-02-10",
        "member_count": 1,
    }
    row.update(overrides)
    return row


class Tier1EnumerationSqlTests(unittest.TestCase):
    def _repo(self, rows=None):
        backend = _RecordingBackend(rows=rows)
        return PostgresCorporateEventRepository(backend), backend

    def test_uses_a_half_open_interval_on_opened_at(self) -> None:
        repo, backend = self._repo()
        repo.enumerate_events(
            corp_code=CORP,
            event_family=FAMILY_SUPPLY_CONTRACT,
            date_from="2025-01-01",
            date_to="2026-01-01",
        )
        sql, params = backend.queries[0]
        self.assertIn("e.opened_at >= %s::date", sql)
        self.assertIn("e.opened_at < %s::date", sql)
        self.assertNotIn("<=", sql)
        self.assertIn("2025-01-01", params)
        self.assertIn("2026-01-01", params)

    def test_selects_only_the_opening_role(self) -> None:
        repo, backend = self._repo()
        repo.enumerate_events(corp_code=CORP, event_family=FAMILY_SUPPLY_CONTRACT)
        sql, params = backend.queries[0]
        self.assertIn("m.member_role = %s", sql)
        self.assertIn("contract", params)

    def test_is_one_statement_with_deterministic_ordering(self) -> None:
        repo, backend = self._repo(rows=[_event_row()])
        repo.enumerate_events(corp_code=CORP, event_family=FAMILY_SUPPLY_CONTRACT)
        self.assertEqual(len(backend.queries), 1)
        sql, _ = backend.queries[0]
        self.assertIn("ORDER BY e.opened_at NULLS LAST, e.event_id", sql)

    def test_returns_the_shared_state_type_with_enumeration_fields(self) -> None:
        repo, _ = self._repo(rows=[_event_row()])
        states = repo.enumerate_events(
            corp_code=CORP, event_family=FAMILY_SUPPLY_CONTRACT
        )
        self.assertEqual(len(states), 1)
        state = states[0]
        self.assertIsInstance(state, CorporateEventState)
        self.assertEqual(state.opened_at, "2025-02-10")
        self.assertEqual(state.resolution_source, "single_document")
        self.assertEqual(state.event_id, "evt_1")
        self.assertEqual(state.canonical_doc_id, "a0")

    def test_single_document_event_is_not_a_dangling_reference(self) -> None:
        """746 of 941 real events are ``unresolved`` only because nothing
        followed them.  Treating that as an evidence gap would flag 79% of the
        corpus, so the distinction is drawn on resolution_source."""

        repo, _ = self._repo(rows=[_event_row()])
        state = repo.enumerate_events(
            corp_code=CORP, event_family=FAMILY_SUPPLY_CONTRACT
        )[0]
        self.assertEqual(state.resolution_status, "unresolved")
        self.assertFalse(state.has_dangling_reference)

    def test_out_of_corpus_reference_is_a_dangling_reference(self) -> None:
        repo, _ = self._repo(
            rows=[_event_row(resolution_source="related_reference_not_in_corpus")]
        )
        state = repo.enumerate_events(
            corp_code=CORP, event_family=FAMILY_SUPPLY_CONTRACT
        )[0]
        self.assertTrue(state.has_dangling_reference)

    def test_rejects_an_unknown_date_field(self) -> None:
        repo, _ = self._repo()
        with self.assertRaises(ValueError):
            repo.enumerate_events(
                corp_code=CORP,
                event_family=FAMILY_SUPPLY_CONTRACT,
                date_field="rcept_dt; DROP TABLE disclosures",
            )

    def test_requires_a_company_and_a_family(self) -> None:
        repo, _ = self._repo()
        with self.assertRaises(ValueError):
            repo.enumerate_events(corp_code="", event_family=FAMILY_SUPPLY_CONTRACT)
        with self.assertRaises(ValueError):
            repo.enumerate_events(corp_code=CORP, event_family="")

    def test_rejects_an_inverted_interval(self) -> None:
        repo, _ = self._repo()
        with self.assertRaises(ValueError):
            repo.enumerate_events(
                corp_code=CORP,
                event_family=FAMILY_SUPPLY_CONTRACT,
                date_from="2026-01-01",
                date_to="2025-01-01",
            )


# ------------------------------------------------------------- Tier 2 SQL


class _SqlCapturingBackend(PostgresBackend):
    def __init__(self):
        super().__init__(dsn="postgresql://unused/unused")
        self.queries = []

    def _fetch_all(self, query, params):
        self.queries.append((query, list(params)))
        return []


class Tier2EnumerationSqlTests(unittest.TestCase):
    def test_applies_group_subtype_and_date_as_hard_sql_predicates(self) -> None:
        backend = _SqlCapturingBackend()
        backend.enumerate_disclosures(
            corp_code=CORP,
            doc_group="exchange",
            doc_subtype=SUPPLY_CONCLUSION,
            date_from="2025-01-01",
            date_to="2026-01-01",
        )
        sql, params = backend.queries[0]
        self.assertIn("d.doc_group = %s", sql)
        self.assertIn("d.doc_subtype = %s", sql)
        self.assertIn("d.rcept_dt >= %s::date", sql)
        self.assertIn("d.rcept_dt < %s::date", sql)
        self.assertIn("exchange", params)
        self.assertIn(SUPPLY_CONCLUSION, params)

    def test_orders_deterministically_and_runs_one_statement(self) -> None:
        backend = _SqlCapturingBackend()
        backend.enumerate_disclosures(corp_code=CORP)
        self.assertEqual(len(backend.queries), 1)
        sql, _ = backend.queries[0]
        self.assertIn("ORDER BY d.doc_id", sql)

    def test_requires_a_company(self) -> None:
        backend = _SqlCapturingBackend()
        with self.assertRaises(ValueError):
            backend.enumerate_disclosures(corp_code="")

    def test_does_not_disturb_the_frozen_soft_boost_path(self) -> None:
        """filter_disclosures must keep treating these three as soft boosts."""

        import inspect

        source = inspect.getsource(PostgresBackend.filter_disclosures)
        self.assertNotIn("d.doc_group = %s", source)
        self.assertNotIn("d.doc_subtype = %s", source)
        self.assertNotIn("d.rcept_dt", source)


# ------------------------------------------- Tier 2 correction-aware collapse


class _State:
    """Stands in for P0-A's CorrectionDocumentState."""

    def __init__(self, group, order, is_latest, status):
        self.correction_group_id = group
        self.correction_order = order
        self.is_latest = is_latest
        self.resolution_status = status

    @property
    def is_resolved_latest(self):
        return self.resolution_status == RESOLVED and self.is_latest


class LogicalCollapseTests(unittest.TestCase):
    def test_resolved_group_collapses_to_one_logical_document(self) -> None:
        states = {
            "a0": _State("g1", 0, False, RESOLVED),
            "a1": _State("g1", 1, False, RESOLVED),
            "a2": _State("g1", 2, True, RESOLVED),
        }
        result = collapse_logical_documents(["a0", "a1", "a2"], states)
        self.assertEqual(result.raw_count, 3)
        self.assertEqual(result.logical_count, 1)
        document = result.documents[0]
        self.assertEqual(document.representative_doc_id, "a2")
        self.assertEqual(document.origin, ORIGIN_RESOLVED_GROUP)
        self.assertEqual(result.unresolved_doc_ids, ())

    def test_ambiguous_correction_is_never_folded(self) -> None:
        states = {
            "a0": _State("g1", 0, False, RESOLVED),
            "a1": _State("g2", 0, True, AMBIGUOUS),
        }
        result = collapse_logical_documents(["a0", "a1"], states)
        self.assertEqual(result.logical_count, 2)
        self.assertIn("a1", result.unresolved_doc_ids)
        origins = {d.representative_doc_id: d.origin for d in result.documents}
        self.assertEqual(origins["a1"], ORIGIN_UNVERIFIED_CORRECTION)

    def test_unresolved_correction_is_never_folded(self) -> None:
        states = {"a1": _State("g2", 0, True, "unresolved")}
        result = collapse_logical_documents(["a0", "a1"], states)
        self.assertEqual(result.logical_count, 2)
        self.assertIn("a1", result.unresolved_doc_ids)

    def test_is_latest_alone_never_makes_a_document_canonical(self) -> None:
        """An ambiguous one-member group also carries is_latest."""

        states = {"a1": _State("g2", 0, True, AMBIGUOUS)}
        result = collapse_logical_documents(["a1"], states)
        document = result.documents[0]
        self.assertEqual(document.origin, ORIGIN_UNVERIFIED_CORRECTION)
        self.assertNotEqual(document.origin, ORIGIN_RESOLVED_GROUP)

    def test_document_with_no_correction_is_standalone(self) -> None:
        result = collapse_logical_documents(["x0", "x1"], {})
        self.assertEqual(result.logical_count, 2)
        self.assertTrue(
            all(d.origin == ORIGIN_STANDALONE for d in result.documents)
        )

    def test_partial_chain_stays_inside_the_enumerated_window(self) -> None:
        """Only a0 and a1 were enumerated; a2 (the group latest) was not."""

        states = {
            "a0": _State("g1", 0, False, RESOLVED),
            "a1": _State("g1", 1, False, RESOLVED),
            "a2": _State("g1", 2, True, RESOLVED),
        }
        result = collapse_logical_documents(["a0", "a1"], states)
        self.assertEqual(result.logical_count, 1)
        self.assertEqual(result.documents[0].representative_doc_id, "a1")

    def test_representative_ids_are_deterministic(self) -> None:
        states = {
            "b0": _State("g2", 0, True, RESOLVED),
            "a0": _State("g1", 0, True, RESOLVED),
        }
        first = collapse_logical_documents(["a0", "b0"], states).representative_ids
        second = collapse_logical_documents(["b0", "a0"], states).representative_ids
        self.assertEqual(first, second)
        self.assertEqual(first, ("a0", "b0"))


if __name__ == "__main__":
    unittest.main()
