import re
import unittest
from pathlib import Path

from app.reasoning.corporate_event import (
    CorporateEvent,
    CorporateEventBuildResult,
    CorporateEventMember,
    CorporateEventRelation,
    EventFamily,
    EventLifecycleStatus,
    EventMemberRole,
    EventRelationType,
    EventResolutionStatus,
    ambiguous_event_anchor,
    deterministic_event_id,
    unresolved_event_anchor,
)
from app.reasoning.correction_graph import (
    AMBIGUOUS,
    RESOLVED,
    CorrectionGraph,
    CorrectionGroupMember,
    DisclosureRecord,
)
from app.reasoning.corporate_event_graph import (
    FAMILY_SUPPLY_CONTRACT,
    FAMILY_TREASURY_TRUST,
    ContractDocument,
    build_corporate_event_graph,
    parse_related_disclosures,
)


CORP_CODE = "00123456"


def member(
    doc_id: str,
    role: EventMemberRole = EventMemberRole.CONTRACT,
    *,
    corp_code: str = CORP_CODE,
    order: int = 0,
) -> CorporateEventMember:
    return CorporateEventMember(
        corp_code=corp_code,
        doc_id=doc_id,
        canonical_doc_id=doc_id,
        member_role=role,
        member_order=order,
        event_date=f"2026-01-{order + 1:02d}",
    )


def event(
    members: tuple[CorporateEventMember, ...],
    *,
    corp_code: str = CORP_CODE,
    event_family: EventFamily | str = EventFamily.SUPPLY_CONTRACT,
    root_logical_key: str | None = None,
    lifecycle_status: EventLifecycleStatus | str = EventLifecycleStatus.OPEN,
    resolution_status: EventResolutionStatus | str = EventResolutionStatus.RESOLVED,
) -> CorporateEvent:
    return CorporateEvent.create(
        corp_code=corp_code,
        event_family=event_family,
        root_logical_key=root_logical_key or members[0].logical_key,
        lifecycle_status=lifecycle_status,
        resolution_status=resolution_status,
        resolution_source="test_fixture",
        members=members,
        opened_at="2026-01-01",
        closed_at=(
            "2026-01-02"
            if lifecycle_status == EventLifecycleStatus.TERMINATED
            else None
        ),
        confidence=1.0,
    )


class CorporateEventDomainTests(unittest.TestCase):
    def test_graph_reuses_the_single_domain_model(self) -> None:
        from app.reasoning import corporate_event_graph

        self.assertIs(corporate_event_graph.CorporateEvent, CorporateEvent)
        self.assertIs(corporate_event_graph.CorporateEventMember, CorporateEventMember)
        self.assertIs(corporate_event_graph.CorporateEventRelation, CorporateEventRelation)

    def test_graph_assembly_keeps_opening_anchor_when_lifecycle_grows(self) -> None:
        from app.reasoning.corporate_event_graph import (
            FAMILY_SUPPLY_CONTRACT,
            RESOLVED,
            ContractDocument,
            CorrectionCanonicalizer,
            TerminationMatch,
            assemble_events,
        )
        from app.reasoning.correction_graph import DisclosureRecord

        records = {
            doc_id: DisclosureRecord(
                doc_id=doc_id,
                corp_code=CORP_CODE,
                doc_group="exchange",
                report_nm=doc_id,
                rcept_no=str(index),
                rcept_dt=f"2026-01-0{index}",
            )
            for index, doc_id in enumerate(("contract-a", "contract-b", "termination-c"), 1)
        }
        documents = {
            "contract-a": ContractDocument(
                "contract-a", CORP_CODE, FAMILY_SUPPLY_CONTRACT, "conclusion"
            ),
            "contract-b": ContractDocument(
                "contract-b", CORP_CODE, FAMILY_SUPPLY_CONTRACT, "conclusion"
            ),
            "termination-c": ContractDocument(
                "termination-c", CORP_CODE, FAMILY_SUPPLY_CONTRACT, "termination"
            ),
        }
        canonicalizer = CorrectionCanonicalizer()
        initial, _ = assemble_events(
            {"contract-a": documents["contract-a"]},
            (),
            records={"contract-a": records["contract-a"]},
            canonicalizer=canonicalizer,
        )
        expanded, _ = assemble_events(
            documents,
            (
                TerminationMatch(
                    termination_doc_id="termination-c",
                    event_family=FAMILY_SUPPLY_CONTRACT,
                    resolution_status=RESOLVED,
                    resolution_source="fixture",
                    confidence=1.0,
                    contract_doc_ids=("contract-a", "contract-b"),
                ),
            ),
            records=records,
            canonicalizer=canonicalizer,
        )

        self.assertEqual(initial[0].event_id, expanded[0].event_id)
        self.assertEqual(expanded[0].root_logical_key, "contract-a")

    def test_adding_contract_member_does_not_change_event_id(self) -> None:
        contract = member("contract-1", order=0)
        update = member("contract-2", EventMemberRole.CONTRACT_UPDATE, order=1)

        initial = event((contract,))
        expanded = event((contract, update))

        self.assertEqual(initial.event_id, expanded.event_id)

    def test_adding_termination_does_not_change_event_id(self) -> None:
        contract = member("contract-1", order=0)
        update = member("contract-2", EventMemberRole.CONTRACT_UPDATE, order=1)
        termination = member(
            "termination-1", EventMemberRole.TERMINATION, order=2
        )

        before = event((contract, update))
        after = event(
            (contract, update, termination),
            lifecycle_status=EventLifecycleStatus.TERMINATED,
        )

        self.assertEqual(before.event_id, after.event_id)

    def test_resolved_correction_latest_change_does_not_change_event_id(self) -> None:
        before = CorporateEventMember(
            corp_code=CORP_CODE,
            doc_id="contract-a",
            canonical_doc_id="contract-a1",
            member_role=EventMemberRole.CONTRACT,
            member_order=0,
            root_doc_id="contract-a",
            correction_group_id="correction-group-a",
            correction_resolution_status=EventResolutionStatus.RESOLVED,
            correction_chain=("contract-a", "contract-a1"),
        )
        after = CorporateEventMember(
            corp_code=CORP_CODE,
            doc_id="contract-a",
            canonical_doc_id="contract-a2",
            member_role=EventMemberRole.CONTRACT,
            member_order=0,
            root_doc_id="contract-a",
            correction_group_id="correction-group-a",
            correction_resolution_status=EventResolutionStatus.RESOLVED,
            correction_chain=("contract-a", "contract-a1", "contract-a2"),
        )

        self.assertEqual(event((before,)).event_id, event((after,)).event_id)

    def test_event_id_is_member_input_order_independent(self) -> None:
        contract = member("contract-1", order=0)
        update = member("contract-2", EventMemberRole.CONTRACT_UPDATE, order=1)
        forward = event((contract, update), root_logical_key=contract.logical_key)
        reversed_input = event(
            (update, contract), root_logical_key=contract.logical_key
        )

        self.assertEqual(forward.event_id, reversed_input.event_id)
        self.assertRegex(forward.event_id, r"^evt_[0-9a-f]{24}$")

        with self.assertRaisesRegex(ValueError, "event_id must equal deterministic id"):
            CorporateEvent(
                event_id="evt_not_the_deterministic_value",
                corp_code=CORP_CODE,
                event_family=EventFamily.SUPPLY_CONTRACT,
                root_logical_key=contract.logical_key,
                lifecycle_status=EventLifecycleStatus.OPEN,
                resolution_status=EventResolutionStatus.RESOLVED,
                resolution_source="test_fixture",
                members=(contract, update),
            )

    def test_different_roots_company_or_family_have_different_event_ids(self) -> None:
        identifiers = {
            deterministic_event_id(CORP_CODE, EventFamily.SUPPLY_CONTRACT, "root-a"),
            deterministic_event_id(CORP_CODE, EventFamily.SUPPLY_CONTRACT, "root-b"),
            deterministic_event_id("00999999", EventFamily.SUPPLY_CONTRACT, "root-a"),
            deterministic_event_id(
                CORP_CODE, EventFamily.TREASURY_TRUST_CONTRACT, "root-a"
            ),
        }
        self.assertEqual(len(identifiers), 4)

    def test_ambiguous_anchor_uses_source_reference_not_a_candidate(self) -> None:
        first = ambiguous_event_anchor(
            "termination-1", {"dates": ["2024-01-01"], "title": "contract"}
        )
        reordered = ambiguous_event_anchor(
            "termination-1", {"title": "contract", "dates": ["2024-01-01"]}
        )

        self.assertEqual(first, reordered)
        self.assertTrue(first.startswith("ambiguous:"))
        self.assertNotIn("candidate", first)

    def test_unresolved_termination_event_id_is_deterministic(self) -> None:
        anchor = unresolved_event_anchor("termination-1")
        repeated = unresolved_event_anchor("termination-1")
        referenced = unresolved_event_anchor(
            "termination-1", {"submitted_on": "2020-05-01"}
        )

        self.assertEqual(anchor, repeated)
        self.assertEqual(
            deterministic_event_id(CORP_CODE, EventFamily.SUPPLY_CONTRACT, anchor),
            deterministic_event_id(CORP_CODE, EventFamily.SUPPLY_CONTRACT, repeated),
        )
        self.assertNotEqual(anchor, referenced)

    def test_duplicate_document_member_is_rejected(self) -> None:
        duplicate = member("contract-1")
        with self.assertRaisesRegex(ValueError, "duplicate document membership"):
            event((duplicate, duplicate))

    def test_invalid_family_lifecycle_resolution_and_role_are_rejected(self) -> None:
        contract = member("contract-1")
        for field_name, overrides in (
            ("event_family", {"event_family": "free_text_event"}),
            ("lifecycle_status", {"lifecycle_status": "closed"}),
            ("resolution_status", {"resolution_status": "probable"}),
        ):
            arguments = {
                "corp_code": CORP_CODE,
                "event_family": EventFamily.SUPPLY_CONTRACT,
                "root_logical_key": contract.logical_key,
                "lifecycle_status": EventLifecycleStatus.OPEN,
                "resolution_status": EventResolutionStatus.RESOLVED,
                "resolution_source": "test_fixture",
                "members": (contract,),
            }
            arguments.update(overrides)
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, f"invalid {field_name}"):
                    CorporateEvent.create(**arguments)

        with self.assertRaisesRegex(ValueError, "invalid member_role"):
            CorporateEventMember(
                corp_code=CORP_CODE,
                doc_id="contract-2",
                canonical_doc_id="contract-2",
                member_role="modification",
                member_order=0,
            )

    def test_multi_member_lifecycle_and_internal_contract_update_are_allowed(self) -> None:
        lifecycle = event(
            (
                member("contract-1", order=0),
                member("contract-2", EventMemberRole.CONTRACT_UPDATE, order=1),
                member("contract-3", EventMemberRole.CONTRACT_UPDATE, order=2),
            )
        )

        self.assertEqual(lifecycle.member_count, 3)
        self.assertEqual(
            [item.member_role for item in lifecycle.members],
            [
                EventMemberRole.CONTRACT,
                EventMemberRole.CONTRACT_UPDATE,
                EventMemberRole.CONTRACT_UPDATE,
            ],
        )

    def test_terminated_lifecycle_accepts_a_termination_member(self) -> None:
        lifecycle = event(
            (
                member("contract-1", order=0),
                member("termination-1", EventMemberRole.TERMINATION, order=1),
            ),
            lifecycle_status=EventLifecycleStatus.TERMINATED,
        )

        self.assertEqual(lifecycle.lifecycle_status, EventLifecycleStatus.TERMINATED)
        self.assertEqual(lifecycle.closed_at, "2026-01-02")

    def test_cross_company_member_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross-company"):
            event(
                (
                    member("contract-1", order=0),
                    member("contract-2", corp_code="00999999", order=1),
                )
            )

    def test_resolved_p0a_group_preserves_provenance_and_uses_verified_latest(self) -> None:
        graph = CorrectionGraph(
            members=(
                CorrectionGroupMember(
                    doc_id="original",
                    correction_group_id="original",
                    root_doc_id="original",
                    parent_doc_id=None,
                    correction_order=0,
                    is_latest=False,
                    resolution_status=RESOLVED,
                    resolution_source="group_root",
                    confidence=1.0,
                ),
                CorrectionGroupMember(
                    doc_id="correction",
                    correction_group_id="original",
                    root_doc_id="original",
                    parent_doc_id="original",
                    correction_order=1,
                    is_latest=True,
                    resolution_status=RESOLVED,
                    resolution_source="correction_notice",
                    confidence=0.95,
                    is_correction=True,
                ),
            )
        )

        item = CorporateEventMember.from_correction_graph(
            corp_code=CORP_CODE,
            doc_id="original",
            member_role=EventMemberRole.CONTRACT,
            correction_graph=graph,
            provenance={"source": "p0-a"},
        )

        self.assertEqual(item.canonical_doc_id, "correction")
        self.assertEqual(item.root_doc_id, "original")
        self.assertEqual(item.correction_group_id, "original")
        self.assertEqual(item.correction_chain, ("original", "correction"))
        self.assertEqual(item.correction_resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(item.provenance, {"source": "p0-a"})

    def test_ambiguous_p0a_group_is_not_treated_as_verified_latest(self) -> None:
        graph = CorrectionGraph(
            members=(
                CorrectionGroupMember(
                    doc_id="ambiguous-correction",
                    correction_group_id="ambiguous-correction",
                    root_doc_id="ambiguous-correction",
                    parent_doc_id=None,
                    correction_order=0,
                    is_latest=True,
                    resolution_status=AMBIGUOUS,
                    resolution_source="multiple_candidates",
                    confidence=0.0,
                    is_correction=True,
                ),
            )
        )

        item = CorporateEventMember.from_correction_graph(
            corp_code=CORP_CODE,
            doc_id="ambiguous-correction",
            member_role=EventMemberRole.CONTRACT,
            correction_graph=graph,
        )

        self.assertEqual(item.canonical_doc_id, "ambiguous-correction")
        self.assertEqual(item.correction_resolution_status, EventResolutionStatus.AMBIGUOUS)

    def test_p0a_versions_stay_members_of_one_event_not_separate_events(self) -> None:
        first = CorporateEventMember(
            corp_code=CORP_CODE,
            doc_id="original",
            canonical_doc_id="correction",
            member_role=EventMemberRole.CONTRACT,
            member_order=0,
            root_doc_id="original",
            correction_group_id="original",
            correction_resolution_status=EventResolutionStatus.RESOLVED,
            correction_chain=("original", "correction"),
        )
        second = CorporateEventMember(
            corp_code=CORP_CODE,
            doc_id="correction",
            canonical_doc_id="correction",
            member_role=EventMemberRole.CONTRACT_UPDATE,
            member_order=1,
            root_doc_id="original",
            correction_group_id="original",
            correction_resolution_status=EventResolutionStatus.RESOLVED,
            correction_chain=("original", "correction"),
        )

        lifecycle = event((first, second), root_logical_key="original")
        self.assertEqual(lifecycle.member_count, 2)
        self.assertEqual(
            lifecycle.event_id,
            deterministic_event_id(
                CORP_CODE, EventFamily.SUPPLY_CONTRACT, "original"
            ),
        )

    def test_self_relation_and_duplicate_relation_are_rejected(self) -> None:
        lifecycle = event(
            (
                member("contract-1", order=0),
                member("termination-1", EventMemberRole.TERMINATION, order=1),
            ),
            lifecycle_status=EventLifecycleStatus.TERMINATED,
        )
        with self.assertRaisesRegex(ValueError, "cannot reference itself"):
            CorporateEventRelation.create(
                event_id=lifecycle.event_id,
                source_doc_id="contract-1",
                target_doc_id="contract-1",
                relation_type=EventRelationType.TERMINATES_EVENT,
                resolution_status=EventResolutionStatus.RESOLVED,
                resolution_source="related_disclosure",
            )

        relation = CorporateEventRelation.create(
            event_id=lifecycle.event_id,
            source_doc_id="termination-1",
            target_doc_id="contract-1",
            relation_type=EventRelationType.TERMINATES_EVENT,
            resolution_status=EventResolutionStatus.RESOLVED,
            resolution_source="related_disclosure",
        )
        with self.assertRaisesRegex(ValueError, "duplicate relation"):
            CorporateEventBuildResult(
                events=(lifecycle,), relations=(relation, relation)
            )


class CorporateEventMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parent.parent
        cls.sql = (project_root / "db" / "007_corporate_event_timeline.sql").read_text(
            encoding="utf-8"
        )
        cls.normalized = re.sub(r"\s+", " ", cls.sql.lower())

    def test_migration_is_additive_and_statically_idempotent(self) -> None:
        self.assertNotRegex(self.normalized, r"\bdrop\s+table\b")
        self.assertNotRegex(self.normalized, r"\btruncate\b")
        self.assertNotRegex(self.normalized, r"\binsert\s+into\b")
        self.assertNotRegex(self.normalized, r"\bdelete\s+from\b")
        self.assertNotRegex(
            self.normalized,
            r"\bupdate\s+(?:companies|disclosures|sections|disclosure_tables|chunks)\b",
        )
        self.assertNotIn("alter table correction_", self.normalized)

        non_idempotent_create = re.findall(
            r"create\s+(?:unique\s+)?(?:table|index)\s+(?!if\s+not\s+exists)",
            self.normalized,
        )
        self.assertEqual(non_idempotent_create, [])
        self.assertEqual(self.normalized.count("begin;"), 1)
        self.assertEqual(self.normalized.count("commit;"), 1)

    def test_migration_declares_required_tables_and_domain_guards(self) -> None:
        for table in (
            "corporate_events",
            "corporate_event_members",
            "corporate_event_relations",
        ):
            self.assertIn(f"create table if not exists {table}", self.normalized)

        self.assertIn("event_id text primary key", self.normalized)
        self.assertIn("root_logical_key text not null", self.normalized)
        self.assertIn("uq_corporate_events_logical_root", self.normalized)
        self.assertIn("trg_corporate_event_identity_immutable", self.normalized)
        self.assertIn("primary key (event_id, doc_id)", self.normalized)
        self.assertIn("uq_corporate_event_members_doc", self.normalized)
        self.assertIn("corporate_events_family_check", self.normalized)
        self.assertIn("corporate_events_lifecycle_check", self.normalized)
        self.assertIn("corporate_events_status_check", self.normalized)
        self.assertIn("corporate_event_members_role_check", self.normalized)
        self.assertIn("corporate_event_relations_no_self_reference", self.normalized)
        self.assertIn("uq_corporate_event_relations_edge", self.normalized)

    def test_cross_company_membership_is_blocked_by_idempotent_triggers(self) -> None:
        self.assertIn("enforce_corporate_event_member_company()", self.normalized)
        self.assertIn("trg_corporate_event_member_company", self.normalized)
        self.assertIn("enforce_corporate_event_relation_company()", self.normalized)
        self.assertIn("trg_corporate_event_relation_company", self.normalized)
        self.assertIn("if not exists ( select 1 from pg_trigger", self.normalized)
        self.assertIn("raise exception 'cross-company corporate event membership", self.normalized)

    def test_p0a_provenance_columns_are_preserved(self) -> None:
        for column in (
            "root_doc_id text references disclosures(doc_id)",
            "correction_group_id text",
            "correction_resolution_status text",
            "correction_chain jsonb not null",
            "provenance jsonb not null",
            "canonical_doc_id text not null",
        ):
            self.assertIn(column, self.normalized)
        self.assertIn(
            "corporate_event_members_unverified_canonical_check", self.normalized
        )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Step 2: the deterministic matcher itself.
#
# Every fixture below is modelled on a case the corpus audit actually found, so
# a regression here means the builder stopped agreeing with the frozen corpus.
# ---------------------------------------------------------------------------

SUPPLY_CONCLUSION = "단일판매공급계약체결"
SUPPLY_TERMINATION = "단일판매공급계약해지"
TRUST_CONCLUSION_NM = "주요사항보고서(자기주식취득신탁계약체결결정)"
TRUST_TERMINATION_NM = "주요사항보고서(자기주식취득신탁계약해지결정)"
#: The reference title a real termination writes, gauche middle dot included.
REF_TITLE = "단일판매ㆍ공급계약체결"


def _record(
    doc_id: str,
    rcept_dt: str,
    *,
    corp_code: str = CORP_CODE,
    doc_group: str = "exchange",
    doc_subtype: str | None = SUPPLY_CONCLUSION,
    report_nm: str = "단일판매ㆍ공급계약체결",
    is_correction: bool = False,
) -> DisclosureRecord:
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
    doc_id: str,
    rcept_dt: str,
    *,
    corp_code: str = CORP_CODE,
    counterparty: str | None = "Acme Corp",
    subject: str | None = "Widget supply",
    amount: str | None = "1,000,000",
    period_start: str | None = "2024-01-01",
    period_end: str | None = "2029-12-31",
) -> ContractDocument:
    return ContractDocument(
        doc_id=doc_id,
        corp_code=corp_code,
        event_family=FAMILY_SUPPLY_CONTRACT,
        kind="conclusion",
        rcept_dt=rcept_dt,
        counterparty=counterparty,
        subject=subject,
        amount=amount,
        period_start=period_start,
        period_end=period_end,
    )


def _termination(
    doc_id: str,
    rcept_dt: str,
    references: tuple[tuple[str, str], ...],
    *,
    corp_code: str = CORP_CODE,
    counterparty: str | None = "Acme Corp",
    subject: str | None = "Widget supply",
    amount: str | None = "1,000,000",
    period_start: str | None = "2024-01-01",
    period_end: str | None = "2029-12-31",
) -> ContractDocument:
    # Built through the real parser so the fixtures exercise the same
    # unrelated-title filtering the corpus goes through.
    raw = " ".join(f"{date} {title}" for date, title in references)
    return ContractDocument(
        doc_id=doc_id,
        corp_code=corp_code,
        event_family=FAMILY_SUPPLY_CONTRACT,
        kind="termination",
        rcept_dt=rcept_dt,
        counterparty=counterparty,
        subject=subject,
        amount=amount,
        period_start=period_start,
        period_end=period_end,
        termination_date=rcept_dt,
        termination_reason="계약상대방의 계약이행 불가",
        related=parse_related_disclosures(raw),
    )


def _trust(
    doc_id: str,
    rcept_dt: str,
    kind: str,
    *,
    corp_code: str = CORP_CODE,
    period_start: str | None = "2024-01-01",
    period_end: str | None = "2024-07-01",
) -> ContractDocument:
    return ContractDocument(
        doc_id=doc_id,
        corp_code=corp_code,
        event_family=FAMILY_TREASURY_TRUST,
        kind=kind,
        rcept_dt=rcept_dt,
        period_start=period_start,
        period_end=period_end,
        termination_date=rcept_dt if kind == "termination" else None,
    )


def _trust_record(
    doc_id: str, rcept_dt: str, kind: str, *, corp_code: str = CORP_CODE
) -> DisclosureRecord:
    return _record(
        doc_id,
        rcept_dt,
        corp_code=corp_code,
        doc_group="major",
        doc_subtype=None,
        report_nm=TRUST_CONCLUSION_NM if kind == "conclusion" else TRUST_TERMINATION_NM,
    )


def _build(records, documents, correction_graph=None):
    return build_corporate_event_graph(
        list(records),
        {document.doc_id: document for document in documents},
        correction_graph=correction_graph,
    )


def _event_of(graph, doc_id):
    found = graph.get_event(doc_id)
    assert found is not None, doc_id
    return found


def _match_evidence(event, doc_id):
    found = next(item for item in event.members if item.doc_id == doc_id)
    return (found.evidence or {}).get("match") or {}


class SupplyMatchingTests(unittest.TestCase):
    def test_conclusion_then_termination_is_one_resolved_lifecycle(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _termination("t1", "2025-05-01", (("2024-01-10", REF_TITLE),)),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(found.lifecycle_status, EventLifecycleStatus.TERMINATED)
        self.assertEqual(sorted(found.doc_ids), ["c1", "t1"])
        self.assertEqual(_event_of(graph, "c1").event_id, found.event_id)

    def test_unrelated_related_disclosure_titles_are_ignored(self) -> None:
        noise = (
            ("2024-01-10", REF_TITLE),
            ("2023-06-01", "기타 경영사항(자율공시)"),
            ("2023-07-01", "채무인수결정"),
            ("2023-08-01", "연결재무제표 기준 영업실적 등에 대한 전망(공정공시)"),
            ("2023-09-01", "조회공시 요구(풍문 또는 보도)"),
        )
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _termination("t1", "2025-05-01", noise),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        match = _match_evidence(found, "t1")
        self.assertEqual(match["contract_reference_dates"], ["2024-01-10"])
        self.assertEqual(len(match["rejected_reference_titles"]), 4)

    def test_a_termination_reference_is_not_a_contract_reference(self) -> None:
        references = parse_related_disclosures("2025-12-17 단일판매ㆍ공급계약 해지")
        self.assertEqual(len(references), 1)
        self.assertFalse(references[0].is_contract_reference)

    def test_original_outside_corpus_stays_unresolved(self) -> None:
        graph = _build(
            [
                _record("other", "2024-02-02"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("other", "2024-02-02"),
                _termination("t1", "2025-05-01", (("2021-03-24", REF_TITLE),)),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.UNRESOLVED)
        self.assertEqual(found.doc_ids, ("t1",))
        match = _match_evidence(found, "t1")
        # The external reference is preserved rather than swapped for a lookalike.
        self.assertEqual(match["contract_reference_dates"], ["2021-03-24"])
        self.assertEqual(match["reference_outcomes"], {"outside_corpus": 1})
        self.assertEqual(
            match["reference_resolutions"][0]["reference_title"], REF_TITLE
        )
        self.assertEqual(_event_of(graph, "other").doc_ids, ("other",))

    def test_root_reference_resolves_through_p0a_canonical_latest(self) -> None:
        """The real exchange_20240401800927 -> ..._20251226800767 -> 해지 shape."""

        correction = CorrectionGraph(
            [
                CorrectionGroupMember(
                    doc_id="root",
                    correction_group_id="g1",
                    root_doc_id="root",
                    parent_doc_id=None,
                    correction_order=0,
                    is_latest=False,
                    resolution_status=RESOLVED,
                    resolution_source="correction_notice",
                    confidence=0.95,
                ),
                CorrectionGroupMember(
                    doc_id="latest",
                    correction_group_id="g1",
                    root_doc_id="root",
                    parent_doc_id="root",
                    correction_order=1,
                    is_latest=True,
                    resolution_status=RESOLVED,
                    resolution_source="correction_notice",
                    confidence=0.95,
                    is_correction=True,
                ),
            ],
            [],
        )
        graph = _build(
            [
                _record("root", "2024-04-01"),
                _record("latest", "2025-12-26", is_correction=True),
                _record("t1", "2025-12-26", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                # The root states almost nothing; the correction carries the fields.
                _conclusion(
                    "root",
                    "2024-04-01",
                    amount=None,
                    period_start=None,
                    period_end=None,
                ),
                _conclusion("latest", "2025-12-26", amount="2,000,000"),
                _termination("t1", "2025-12-26", (("2024-04-01", REF_TITLE),)),
            ],
            correction_graph=correction,
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        # root and its correction are one logical contract, so the timeline is
        # [contract, termination] -- not three rows.
        self.assertEqual(sorted(found.doc_ids), ["latest", "t1"])
        by_doc = {item.doc_id: item for item in found.members}
        self.assertEqual(by_doc["latest"].canonical_doc_id, "latest")
        self.assertEqual(by_doc["latest"].correction_group_id, "g1")
        self.assertEqual(by_doc["latest"].root_doc_id, "root")
        self.assertEqual(
            list(by_doc["latest"].correction_chain), ["root", "latest"]
        )
        self.assertEqual(
            by_doc["latest"].provenance["collapsed_doc_ids"], ["root", "latest"]
        )
        # The superseded filing still reaches its lifecycle.
        self.assertEqual(_event_of(graph, "root").event_id, found.event_id)
        self.assertEqual(graph.get_member("root").doc_id, "latest")
        # The period only matched through the canonical latest filing.
        comparison = _match_evidence(found, "t1")["reference_resolutions"][0]
        self.assertEqual(
            comparison["comparisons"]["root"]["compared_against_doc_id"], "latest"
        )

    def test_same_date_multiple_contracts_are_separated(self) -> None:
        """The real 삼성중공업 2024-06-12 pair, terminated on one day."""

        graph = _build(
            [
                _record("a", "2024-06-12"),
                _record("b", "2024-06-12"),
                _record("ta", "2025-06-18", doc_subtype=SUPPLY_TERMINATION),
                _record("tb", "2025-06-18", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion(
                    "a",
                    "2024-06-12",
                    counterparty="유라시아 지역 선주",
                    subject="블록, 기자재 및 설계",
                    amount="2,045,300,000,000",
                    period_start="2021-10-15",
                ),
                _conclusion(
                    "b",
                    "2024-06-12",
                    counterparty="유럽 지역 선주",
                    subject="블록 및 기자재",
                    amount="2,807,200,000,000",
                    period_start="2020-11-20",
                ),
                _termination(
                    "ta",
                    "2025-06-18",
                    (("2024-06-12", REF_TITLE),),
                    counterparty="유라시아 지역 선주",
                    subject="블록, 기자재 및 설계",
                    amount="2,045,300,000,000",
                    period_start="2021-10-15",
                ),
                _termination(
                    "tb",
                    "2025-06-18",
                    (("2024-06-12", REF_TITLE),),
                    counterparty="유럽 지역 선주",
                    subject="블록 및 기자재",
                    amount="2,807,200,000,000",
                    period_start="2020-11-20",
                ),
            ],
        )
        self.assertEqual(sorted(_event_of(graph, "ta").doc_ids), ["a", "ta"])
        self.assertEqual(sorted(_event_of(graph, "tb").doc_ids), ["b", "tb"])
        self.assertNotEqual(
            _event_of(graph, "ta").event_id, _event_of(graph, "tb").event_id
        )
        self.assertEqual(graph.diagnostics()["duplicate_membership_count"], 0)

    def test_identical_contracts_are_separated_by_period(self) -> None:
        """The real LG에너지솔루션 Ford pair: only the contract period differs."""

        shared = dict(
            counterparty="Ford Motor Company",
            subject="전기차 배터리 공급계약",
            amount=None,
        )
        graph = _build(
            [
                _record("f1", "2024-10-15"),
                _record("f2", "2024-10-15"),
                _record("t1", "2025-12-17", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion(
                    "f1",
                    "2024-10-15",
                    period_start="2027-01-01",
                    period_end="2032-12-31",
                    **shared,
                ),
                _conclusion(
                    "f2",
                    "2024-10-15",
                    period_start="2026-10-01",
                    period_end="2030-12-31",
                    **shared,
                ),
                _termination(
                    "t1",
                    "2025-12-17",
                    (("2024-10-15", REF_TITLE),),
                    period_start="2027-01-01",
                    period_end="2032-12-31",
                    **shared,
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(sorted(found.doc_ids), ["f1", "t1"])
        self.assertNotIn("f2", found.doc_ids)

    def test_end_date_alone_never_rejects_a_candidate(self) -> None:
        """종료일 is mutable, so a difference is a change and not a conflict."""

        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10", period_end="2029-12-31"),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE),),
                    period_end="2026-06-30",
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        comparison = _match_evidence(found, "t1")["reference_resolutions"][0]
        self.assertIn("period_end", comparison["comparisons"]["c1"]["changed_fields"])
        self.assertEqual(
            comparison["comparisons"]["c1"]["conflicting_identity_fields"], []
        )

    def test_amount_alone_never_rejects_a_candidate(self) -> None:
        """The real 대우건설 case: 362,307,038,000 terminated vs 351,882,914,000 signed."""

        graph = _build(
            [
                _record("c1", "2023-02-28"),
                _record("t1", "2024-11-15", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion(
                    "c1",
                    "2023-02-28",
                    counterparty="경안리버시티개발 주식회사",
                    subject="광주 경안2지구 도시개발사업",
                    amount="351,882,914,000",
                    period_start=None,
                    period_end=None,
                ),
                _termination(
                    "t1",
                    "2024-11-15",
                    (("2023-02-28", REF_TITLE),),
                    counterparty="경안리버시티개발 주식회사",
                    subject="광주 경안2지구 도시개발사업",
                    amount="362,307,038,000",
                    period_start=None,
                    period_end=None,
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        comparison = _match_evidence(found, "t1")["reference_resolutions"][0]
        self.assertIn("amount", comparison["comparisons"]["c1"]["changed_fields"])

    def test_same_counterparty_different_contract_is_not_merged(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-03-20"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10", subject="Widget supply"),
                _conclusion("c2", "2024-03-20", subject="Gadget supply"),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE),),
                    subject="Widget supply",
                ),
            ],
        )
        self.assertEqual(sorted(_event_of(graph, "t1").doc_ids), ["c1", "t1"])
        self.assertEqual(_event_of(graph, "c2").doc_ids, ("c2",))

    def test_same_amount_different_contract_is_not_merged(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion(
                    "c1",
                    "2024-01-10",
                    counterparty="Acme Corp",
                    subject="Widget supply",
                    amount="500",
                ),
                _conclusion(
                    "c2",
                    "2024-01-10",
                    counterparty="Globex Ltd",
                    subject="Sprocket supply",
                    amount="500",
                    period_start="2020-01-01",
                ),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE),),
                    counterparty="Acme Corp",
                    subject="Widget supply",
                    amount="500",
                ),
            ],
        )
        self.assertEqual(sorted(_event_of(graph, "t1").doc_ids), ["c1", "t1"])
        self.assertEqual(_event_of(graph, "c2").doc_ids, ("c2",))

    def test_multiple_references_form_one_multi_member_lifecycle(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-06-10"),
                _record("c3", "2024-09-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _conclusion("c2", "2024-06-10"),
                _conclusion("c3", "2024-09-10"),
                _termination(
                    "t1",
                    "2025-05-01",
                    (
                        ("2024-01-10", REF_TITLE),
                        ("2024-06-10", REF_TITLE),
                        ("2024-09-10", REF_TITLE),
                    ),
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(sorted(found.doc_ids), ["c1", "c2", "c3", "t1"])
        # Chronological, with roles rather than a rewritten disclosure subtype.
        self.assertEqual([item.doc_id for item in found.members], ["c1", "c2", "c3", "t1"])
        roles = {item.doc_id: str(item.member_role.value) for item in found.members}
        self.assertEqual(roles["c1"], "contract")
        self.assertEqual(roles["c2"], "contract_update")
        self.assertEqual(roles["c3"], "contract_update")
        self.assertEqual(roles["t1"], "termination")

    def test_multi_reference_alone_is_never_ambiguous(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-06-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _conclusion("c2", "2024-06-10"),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE), ("2024-06-10", REF_TITLE)),
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(
            _match_evidence(found, "t1")["reference_outcomes"], {"resolved": 2}
        )

    def test_one_reference_with_two_equal_candidates_is_ambiguous(self) -> None:
        """A tie behind a single reference must never become a lifecycle."""

        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10", period_start=None, period_end=None),
                _conclusion("c2", "2024-01-10", period_start=None, period_end=None),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE),),
                    period_start=None,
                    period_end=None,
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.AMBIGUOUS)
        self.assertEqual(found.doc_ids, ("t1",))
        # Neither candidate is quietly absorbed.
        self.assertEqual(_event_of(graph, "c1").doc_ids, ("c1",))
        self.assertEqual(_event_of(graph, "c2").doc_ids, ("c2",))
        self.assertEqual(
            _match_evidence(found, "t1")["reference_outcomes"], {"tie": 1}
        )

    def test_strong_identity_conflict_is_ambiguous_not_resolved(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion(
                    "c1",
                    "2024-01-10",
                    counterparty="Globex Ltd",
                    subject="Sprocket supply",
                    period_start="2020-01-01",
                ),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE),),
                    counterparty="Acme Corp",
                    subject="Widget supply",
                    period_start="2024-01-01",
                ),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.AMBIGUOUS)
        self.assertEqual(found.doc_ids, ("t1",))

    def test_cross_company_reference_never_produces_a_candidate(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10", corp_code="00999999"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10", corp_code="00999999"),
                _termination("t1", "2025-05-01", (("2024-01-10", REF_TITLE),)),
            ],
        )
        found = _event_of(graph, "t1")
        self.assertEqual(found.resolution_status, EventResolutionStatus.UNRESOLVED)
        self.assertEqual(found.doc_ids, ("t1",))
        self.assertEqual(graph.diagnostics()["cross_company_event_count"], 0)

    def test_a_contract_filed_after_the_termination_is_not_a_candidate(self) -> None:
        graph = _build(
            [
                _record("c1", "2026-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2026-01-10"),
                _termination("t1", "2025-05-01", (("2026-01-10", REF_TITLE),)),
            ],
        )
        self.assertEqual(_event_of(graph, "t1").doc_ids, ("t1",))

    def test_graph_invariants_hold_and_no_modifies_relation_exists(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-06-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _conclusion("c2", "2024-06-10"),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE), ("2024-06-10", REF_TITLE)),
                ),
            ],
        )
        report = graph.diagnostics()
        for guard in (
            "duplicate_membership_count",
            "duplicate_relation_count",
            "self_relation_count",
            "cross_company_event_count",
            "cycle_count",
            "invalid_resolution_relation_count",
        ):
            self.assertEqual(report[guard], 0, guard)
        kinds = {str(item.relation_type.value) for item in graph.relations}
        self.assertLessEqual(kinds, {"belongs_to_event", "terminates_event"})


class SupplyEventIdentityTests(unittest.TestCase):
    def _fixture(self, *, with_update=False, with_termination=False):
        records = [_record("c1", "2024-01-10")]
        documents = [_conclusion("c1", "2024-01-10")]
        references = [("2024-01-10", REF_TITLE)]
        if with_update:
            records.append(_record("c2", "2024-06-10"))
            documents.append(_conclusion("c2", "2024-06-10"))
            references.append(("2024-06-10", REF_TITLE))
        if with_termination:
            records.append(_record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION))
            documents.append(_termination("t1", "2025-05-01", tuple(references)))
        return records, documents

    def test_event_id_is_stable_when_a_termination_is_added(self) -> None:
        before = _build(*self._fixture())
        after = _build(*self._fixture(with_termination=True))
        self.assertEqual(
            _event_of(before, "c1").event_id, _event_of(after, "c1").event_id
        )

    def test_event_id_is_stable_when_a_contract_update_is_added(self) -> None:
        before = _build(*self._fixture(with_termination=True))
        after = _build(*self._fixture(with_update=True, with_termination=True))
        self.assertEqual(
            _event_of(before, "c1").event_id, _event_of(after, "c1").event_id
        )
        self.assertEqual(_event_of(after, "c1").member_count, 3)

    def test_rebuilding_the_same_input_is_idempotent(self) -> None:
        first = _build(*self._fixture(with_update=True, with_termination=True))
        second = _build(*self._fixture(with_update=True, with_termination=True))
        self.assertEqual(
            [item.to_dict() for item in first.events],
            [item.to_dict() for item in second.events],
        )
        self.assertEqual(
            [item.to_dict() for item in first.relations],
            [item.to_dict() for item in second.relations],
        )


class TrustMatchingTests(unittest.TestCase):
    def test_exact_period_resolves(self) -> None:
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust("k1", "2024-01-01", "conclusion"),
                _trust("k2", "2024-07-01", "termination"),
            ],
        )
        found = _event_of(graph, "k2")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(found.lifecycle_status, EventLifecycleStatus.TERMINATED)
        self.assertEqual(sorted(found.doc_ids), ["k1", "k2"])
        self.assertEqual(
            str(found.event_family.value), "treasury_trust_contract"
        )

    def test_opening_outside_corpus_stays_unresolved(self) -> None:
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust(
                    "k1",
                    "2024-01-01",
                    "conclusion",
                    period_start="2024-01-01",
                    period_end="2024-12-31",
                ),
                _trust(
                    "k2",
                    "2024-07-01",
                    "termination",
                    period_start="2022-02-21",
                    period_end="2023-02-20",
                ),
            ],
        )
        found = _event_of(graph, "k2")
        self.assertEqual(found.resolution_status, EventResolutionStatus.UNRESOLVED)
        self.assertEqual(found.doc_ids, ("k2",))

    def test_two_distinct_openings_on_one_period_are_ambiguous(self) -> None:
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion"),
                _trust_record("k1b", "2024-01-02", "conclusion"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust("k1", "2024-01-01", "conclusion"),
                _trust("k1b", "2024-01-02", "conclusion"),
                _trust("k2", "2024-07-01", "termination"),
            ],
        )
        found = _event_of(graph, "k2")
        self.assertEqual(found.resolution_status, EventResolutionStatus.AMBIGUOUS)
        self.assertEqual(found.doc_ids, ("k2",))

    def test_a_corrected_opening_counts_once_not_as_a_tie(self) -> None:
        """P0-A collapses the corrected duplicate into one logical candidate."""

        correction = CorrectionGraph(
            [
                CorrectionGroupMember(
                    doc_id="k1",
                    correction_group_id="g1",
                    root_doc_id="k1",
                    parent_doc_id=None,
                    correction_order=0,
                    is_latest=False,
                    resolution_status=RESOLVED,
                    resolution_source="correction_notice",
                    confidence=0.95,
                ),
                CorrectionGroupMember(
                    doc_id="k1b",
                    correction_group_id="g1",
                    root_doc_id="k1",
                    parent_doc_id="k1",
                    correction_order=1,
                    is_latest=True,
                    resolution_status=RESOLVED,
                    resolution_source="correction_notice",
                    confidence=0.95,
                    is_correction=True,
                ),
            ],
            [],
        )
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion"),
                _trust_record("k1b", "2024-01-02", "conclusion"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust("k1", "2024-01-01", "conclusion"),
                _trust("k1b", "2024-01-02", "conclusion"),
                _trust("k2", "2024-07-01", "termination"),
            ],
            correction_graph=correction,
        )
        found = _event_of(graph, "k2")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual(sorted(found.doc_ids), ["k1b", "k2"])
        self.assertEqual(graph.diagnostics()["duplicate_membership_count"], 0)
        self.assertEqual(_event_of(graph, "k1").event_id, found.event_id)

    def test_no_similarity_fallback_when_the_period_differs(self) -> None:
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust("k1", "2024-01-01", "conclusion", period_end="2024-07-02"),
                _trust("k2", "2024-07-01", "termination", period_end="2024-07-01"),
            ],
        )
        self.assertEqual(_event_of(graph, "k2").doc_ids, ("k2",))

    def test_cross_company_period_match_is_never_produced(self) -> None:
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion", corp_code="00999999"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust("k1", "2024-01-01", "conclusion", corp_code="00999999"),
                _trust("k2", "2024-07-01", "termination"),
            ],
        )
        self.assertEqual(_event_of(graph, "k2").doc_ids, ("k2",))
        self.assertEqual(graph.diagnostics()["cross_company_event_count"], 0)


class AmbiguousCorrectionSafetyTests(unittest.TestCase):
    """The real major_20241108000518 shape: P0-A could not attribute it."""

    def _ambiguous_correction(self) -> CorrectionGraph:
        return CorrectionGraph(
            [
                CorrectionGroupMember(
                    doc_id="k1b",
                    correction_group_id="k1b",
                    root_doc_id="k1b",
                    parent_doc_id=None,
                    correction_order=0,
                    is_latest=True,
                    resolution_status=AMBIGUOUS,
                    resolution_source="multiple_candidates",
                    confidence=0.0,
                    is_correction=True,
                )
            ],
            [],
        )

    def test_ambiguous_correction_is_not_promoted_or_folded(self) -> None:
        graph = _build(
            [
                _trust_record("k1", "2024-01-01", "conclusion"),
                _trust_record("k1b", "2024-01-02", "conclusion"),
                _trust_record("k2", "2024-07-01", "termination"),
            ],
            [
                _trust("k1", "2024-01-01", "conclusion"),
                _trust("k1b", "2024-01-02", "conclusion"),
                _trust("k2", "2024-07-01", "termination"),
            ],
            correction_graph=self._ambiguous_correction(),
        )
        # Two logical openings remain, so the termination is not folded onto
        # whichever one happens to look closest.
        self.assertEqual(
            _event_of(graph, "k2").resolution_status, EventResolutionStatus.AMBIGUOUS
        )
        member = _event_of(graph, "k1b").members[0]
        self.assertEqual(member.canonical_doc_id, "k1b")
        self.assertEqual(str(member.correction_resolution_status.value), "ambiguous")

    def test_ambiguous_correction_state_is_reported(self) -> None:
        graph = _build(
            [_trust_record("k1b", "2024-01-02", "conclusion")],
            [_trust("k1b", "2024-01-02", "conclusion")],
            correction_graph=self._ambiguous_correction(),
        )
        report = graph.diagnostics()
        self.assertEqual(report["ambiguous_correction_member_count"], 1)
        self.assertEqual(report["unverified_correction_member_count"], 1)


# ---------------------------------------------------------------------------
# The P0-A / P0-B membership invariant.
#
# One member row is one logical corporate-event disclosure state. A correction
# group P0-A resolved contributes exactly one row, represented by its verified
# latest filing, with the chain kept as provenance. Emitting a row per raw
# filing would restate the correction graph inside the event graph.
# ---------------------------------------------------------------------------


def _chain_correction_graph(*doc_ids: str, group_id: str = "g1") -> CorrectionGraph:
    """A resolved P0-A chain: doc_ids[0] original, the rest corrections."""

    members = []
    for order, doc_id in enumerate(doc_ids):
        members.append(
            CorrectionGroupMember(
                doc_id=doc_id,
                correction_group_id=group_id,
                root_doc_id=doc_ids[0],
                parent_doc_id=doc_ids[order - 1] if order else None,
                correction_order=order,
                is_latest=(order == len(doc_ids) - 1),
                resolution_status=RESOLVED,
                resolution_source="correction_notice",
                confidence=0.95,
                is_correction=bool(order),
            )
        )
    return CorrectionGraph(members, [])


class LogicalMembershipTests(unittest.TestCase):
    def _chain_only(self):
        """A -> A1 -> A2, no termination."""

        return _build(
            [
                _record("A", "2024-01-10"),
                _record("A1", "2024-03-10", is_correction=True),
                _record("A2", "2024-05-10", is_correction=True),
            ],
            [
                _conclusion("A", "2024-01-10"),
                _conclusion("A1", "2024-03-10"),
                _conclusion("A2", "2024-05-10"),
            ],
            correction_graph=_chain_correction_graph("A", "A1", "A2"),
        )

    def _chain_with_termination(self):
        return _build(
            [
                _record("A", "2024-01-10"),
                _record("A1", "2024-03-10", is_correction=True),
                _record("A2", "2024-05-10", is_correction=True),
                _record("T", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("A", "2024-01-10"),
                _conclusion("A1", "2024-03-10"),
                _conclusion("A2", "2024-05-10"),
                _termination("T", "2025-05-01", (("2024-01-10", REF_TITLE),)),
            ],
            correction_graph=_chain_correction_graph("A", "A1", "A2"),
        )

    # A. resolved correction collapse
    def test_resolved_chain_is_one_logical_contract_member(self) -> None:
        graph = self._chain_only()
        found = _event_of(graph, "A")
        self.assertEqual(len(found.contract_members), 1)
        self.assertEqual(found.member_count, 1)
        self.assertEqual(found.doc_ids, ("A2",))

    # B. correction + termination
    def test_chain_plus_termination_is_two_logical_members(self) -> None:
        graph = self._chain_with_termination()
        found = _event_of(graph, "T")
        self.assertEqual([item.doc_id for item in found.members], ["A2", "T"])
        self.assertEqual(len(found.contract_members), 1)
        self.assertEqual(len(found.termination_members), 1)

    # C. correction chain provenance
    def test_collapsed_member_keeps_root_canonical_and_chain(self) -> None:
        graph = self._chain_with_termination()
        member = _event_of(graph, "T").contract_members[0]
        self.assertEqual(member.doc_id, "A2")
        self.assertEqual(member.canonical_doc_id, "A2")
        self.assertEqual(member.root_doc_id, "A")
        self.assertEqual(list(member.correction_chain), ["A", "A1", "A2"])
        self.assertEqual(
            member.provenance["collapsed_doc_ids"], ["A", "A1", "A2"]
        )
        self.assertEqual(member.provenance["root_doc_id"], "A")
        self.assertEqual(member.provenance["correction_group_id"], "g1")

    def test_every_superseded_filing_still_reaches_its_lifecycle(self) -> None:
        graph = self._chain_with_termination()
        expected = _event_of(graph, "T").event_id
        for doc_id in ("A", "A1", "A2"):
            self.assertEqual(_event_of(graph, doc_id).event_id, expected)
            self.assertEqual(graph.get_member(doc_id).doc_id, "A2")
            self.assertEqual(graph.get_event_state(doc_id).canonical_doc_id, "A2")
        # Reverse traversal from the original reaches the termination.
        self.assertEqual(graph.get_related_documents("A"), ("T",))
        self.assertEqual(graph.get_related_documents("T"), ("A2",))

    # D. relation collapse
    def test_termination_emits_one_relation_per_logical_contract(self) -> None:
        graph = self._chain_with_termination()
        edges = [
            item
            for item in graph.relations
            if str(item.relation_type.value) == "terminates_event"
        ]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_doc_id, "T")
        self.assertEqual(edges[0].target_doc_id, "A2")
        self.assertEqual(graph.diagnostics()["duplicate_relation_count"], 0)

    def test_multi_reference_across_one_chain_still_emits_one_relation(self) -> None:
        """The real 두산퓨얼셀 shape: several references, one logical contract."""

        graph = _build(
            [
                _record("A", "2024-01-10"),
                _record("A1", "2024-06-10", is_correction=True),
                _record("A2", "2024-09-10", is_correction=True),
                _record("T", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("A", "2024-01-10"),
                _conclusion("A1", "2024-06-10"),
                _conclusion("A2", "2024-09-10"),
                _termination(
                    "T",
                    "2025-05-01",
                    (
                        ("2024-01-10", REF_TITLE),
                        ("2024-06-10", REF_TITLE),
                        ("2024-09-10", REF_TITLE),
                    ),
                ),
            ],
            correction_graph=_chain_correction_graph("A", "A1", "A2"),
        )
        found = _event_of(graph, "T")
        self.assertEqual(found.resolution_status, EventResolutionStatus.RESOLVED)
        self.assertEqual([item.doc_id for item in found.members], ["A2", "T"])
        edges = [
            item
            for item in graph.relations
            if str(item.relation_type.value) == "terminates_event"
        ]
        self.assertEqual(len(edges), 1)
        # Three references, one logical contract -> no contract_update role.
        self.assertEqual(len(found.contract_members), 1)

    # E. ambiguous correction is never collapsed
    def test_ambiguous_correction_is_never_collapsed(self) -> None:
        ambiguous = CorrectionGraph(
            [
                CorrectionGroupMember(
                    doc_id="A1",
                    correction_group_id="A1",
                    root_doc_id="A1",
                    parent_doc_id=None,
                    correction_order=0,
                    is_latest=True,
                    resolution_status=AMBIGUOUS,
                    resolution_source="multiple_candidates",
                    confidence=0.0,
                    is_correction=True,
                )
            ],
            [],
        )
        graph = _build(
            [
                _record("A", "2024-01-10"),
                _record("A1", "2024-03-10", is_correction=True),
            ],
            [_conclusion("A", "2024-01-10"), _conclusion("A1", "2024-03-10")],
            correction_graph=ambiguous,
        )
        # Two independent logical contracts, because nothing established that A1
        # is a version of A.
        self.assertEqual(_event_of(graph, "A").doc_ids, ("A",))
        self.assertEqual(_event_of(graph, "A1").doc_ids, ("A1",))
        self.assertNotEqual(
            _event_of(graph, "A").event_id, _event_of(graph, "A1").event_id
        )
        member = _event_of(graph, "A1").members[0]
        self.assertEqual(member.canonical_doc_id, "A1")
        self.assertEqual(str(member.correction_resolution_status.value), "ambiguous")

    # F. diagnostics separate raw from logical
    def test_diagnostics_separate_raw_documents_from_logical_members(self) -> None:
        report = self._chain_with_termination().diagnostics()
        self.assertEqual(report["raw_contract_document_count"], 4)
        self.assertEqual(report["logical_event_member_count"], 2)
        self.assertEqual(report["membership_count"], 2)
        self.assertEqual(report["resolved_correction_documents_collapsed"], 2)
        self.assertEqual(report["resolved_correction_groups_reused"], 1)
        self.assertEqual(report["ambiguous_correction_documents_preserved"], 0)

    def test_multi_contract_needs_two_distinct_logical_contracts(self) -> None:
        """체결 -> 해지 is multi-member but not multi-contract."""

        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _termination("t1", "2025-05-01", (("2024-01-10", REF_TITLE),)),
            ],
        )
        report = graph.diagnostics()
        self.assertEqual(report["multi_member_event_count"], 1)
        self.assertEqual(report["multi_contract_event_count"], 0)

    def test_two_distinct_contracts_do_count_as_multi_contract(self) -> None:
        graph = _build(
            [
                _record("c1", "2024-01-10"),
                _record("c2", "2024-06-10"),
                _record("t1", "2025-05-01", doc_subtype=SUPPLY_TERMINATION),
            ],
            [
                _conclusion("c1", "2024-01-10"),
                _conclusion("c2", "2024-06-10"),
                _termination(
                    "t1",
                    "2025-05-01",
                    (("2024-01-10", REF_TITLE), ("2024-06-10", REF_TITLE)),
                ),
            ],
        )
        report = graph.diagnostics()
        self.assertEqual(report["multi_contract_event_count"], 1)
        roles = [
            str(item.member_role.value) for item in _event_of(graph, "t1").members
        ]
        self.assertEqual(roles, ["contract", "contract_update", "termination"])

    def test_collapse_is_idempotent_across_rebuilds(self) -> None:
        first = self._chain_with_termination()
        second = self._chain_with_termination()
        self.assertEqual(
            [item.to_dict() for item in first.events],
            [item.to_dict() for item in second.events],
        )
        self.assertEqual(
            [item.to_dict() for item in first.relations],
            [item.to_dict() for item in second.relations],
        )


# ---------------------------------------------------------------------------
# Step 3: repository + resolver.
#
# Two layers.  The fake-cursor tests below run everywhere and pin the contract
# the repository must honour.  The live-PostgreSQL tests further down are
# skipped unless FESTIVAL_TEST_DATABASE_URL points at a disposable database, so
# the suite never depends on a server being up.
# ---------------------------------------------------------------------------

import os

from app.reasoning.corporate_event_graph import CorporateEventGraphUnavailable
from app.reasoning.corporate_event_resolver import CorporateEventResolver
from app.retrieval.corporate_event_repository import PostgresCorporateEventRepository


class _RecordingCursor:
    """Captures the SQL a repository issues without touching a database."""

    def __init__(self, store):
        self._store = store
        self._rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=()):
        self._store.queries.append((" ".join(query.split()), list(params or ())))
        self._rows = self._store.rows_for(query, params)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _RecordingConnection:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, row_factory=None):
        return _RecordingCursor(self._store)

    def commit(self):
        self._store.commits += 1


class _RecordingBackend:
    def __init__(self, rows=None, error=None):
        self.queries: list = []
        self.commits = 0
        self._rows = rows or []
        self._error = error

    def rows_for(self, query, params):
        if self._error is not None:
            raise self._error
        return self._rows

    def connection(self):
        return _RecordingConnection(self)


class RepositoryAliasContractTests(unittest.TestCase):
    """A superseded filing must be answerable without a member row of its own."""

    def _repo(self, rows=None, error=None):
        backend = _RecordingBackend(rows=rows, error=error)
        return PostgresCorporateEventRepository(backend), backend

    def test_lookup_matches_chain_and_collapsed_aliases(self) -> None:
        repo, backend = self._repo()
        repo.get_event("A")
        sql, params = backend.queries[0]
        self.assertIn("m.doc_id = %s", sql)
        self.assertIn("m.correction_chain ? %s", sql)
        self.assertIn("m.provenance -> 'collapsed_doc_ids' ? %s", sql)
        # The same identifier is bound to all three alternatives.
        self.assertEqual(params, ["A", "A", "A"])

    def test_event_states_keys_by_the_identifier_the_caller_asked_for(self) -> None:
        repo, backend = self._repo(
            rows=[
                {
                    "asked_doc_id": "A",
                    "doc_id": "A2",
                    "event_id": "evt_1",
                    "member_role": "contract",
                    "canonical_doc_id": "A2",
                    "correction_group_id": "g1",
                    "correction_resolution_status": "resolved",
                    "corp_code": CORP_CODE,
                    "event_family": "supply_contract",
                    "lifecycle_status": "terminated",
                    "resolution_status": "resolved",
                    "member_count": 2,
                }
            ]
        )
        states = repo.event_states(["A"])
        self.assertIn("A", states)
        # Keyed by the asked id, describing the logical member.
        self.assertEqual(states["A"].doc_id, "A2")
        self.assertEqual(states["A"].canonical_doc_id, "A2")

    def test_persist_offers_no_partial_scope(self) -> None:
        """Full rebuild only, so a partial build cannot delete what it never saw."""

        import inspect

        signature = inspect.signature(PostgresCorporateEventRepository.persist_graph)
        self.assertEqual(list(signature.parameters), ["self", "graph"])


class RepositoryErrorHandlingTests(unittest.TestCase):
    """A missing migration degrades; a programming error must not."""

    def test_missing_table_is_reported_as_unavailable(self) -> None:
        import psycopg

        backend = _RecordingBackend(
            error=psycopg.errors.UndefinedTable("relation does not exist")
        )
        repo = PostgresCorporateEventRepository(backend)
        with self.assertRaises(CorporateEventGraphUnavailable):
            repo.get_event("A")

    def test_unreachable_database_is_reported_as_unavailable(self) -> None:
        import psycopg

        backend = _RecordingBackend(error=psycopg.OperationalError("no server"))
        repo = PostgresCorporateEventRepository(backend)
        with self.assertRaises(CorporateEventGraphUnavailable):
            repo.get_event("A")

    def test_undefined_column_propagates_instead_of_degrading(self) -> None:
        """A SQL typo is a defect, not a missing migration."""

        import psycopg

        backend = _RecordingBackend(
            error=psycopg.errors.UndefinedColumn("column m.typo does not exist")
        )
        repo = PostgresCorporateEventRepository(backend)
        with self.assertRaises(psycopg.errors.UndefinedColumn):
            repo.get_event("A")

    def test_programming_error_propagates(self) -> None:
        backend = _RecordingBackend(error=TypeError("bad argument"))
        repo = PostgresCorporateEventRepository(backend)
        with self.assertRaises(TypeError):
            repo.get_event("A")

    def test_resolver_degrades_only_on_unavailable_graph(self) -> None:
        class _Unavailable:
            def get_event(self, doc_id):
                raise CorporateEventGraphUnavailable("db/007 not applied")

        class _Broken:
            def get_event(self, doc_id):
                raise RuntimeError("programming error")

        self.assertFalse(
            CorporateEventResolver(_Unavailable()).get_event_timeline("A").found
        )
        with self.assertRaises(RuntimeError):
            CorporateEventResolver(_Broken()).get_event_timeline("A")

    def test_canonicalizer_propagates_a_broken_correction_graph(self) -> None:
        """P0-A lookup failures other than "unavailable" must surface."""

        from app.reasoning.corporate_event_graph import CorrectionCanonicalizer

        class _Broken:
            def get_correction_group(self, doc_id):
                raise RuntimeError("programming error")

        class _Unavailable:
            def get_correction_group(self, doc_id):
                from app.reasoning.correction_graph import CorrectionGraphUnavailable

                raise CorrectionGraphUnavailable("db/006 not applied")

        with self.assertRaises(RuntimeError):
            CorrectionCanonicalizer(_Broken()).canonical("A")
        # Unavailable degrades to "no correction group".
        canonical = CorrectionCanonicalizer(_Unavailable()).canonical("A")
        self.assertEqual(canonical.canonical_doc_id, "A")
        self.assertIsNone(canonical.correction_group_id)


_LIVE_DSN = os.environ.get("FESTIVAL_TEST_DATABASE_URL")


@unittest.skipUnless(_LIVE_DSN, "FESTIVAL_TEST_DATABASE_URL is not set")
class LivePostgresRoundTripTests(unittest.TestCase):
    """Round-trip against a disposable PostgreSQL that already holds the graph."""

    @classmethod
    def setUpClass(cls) -> None:
        from app.retrieval.postgres_backend import PostgresBackend

        os.environ.setdefault("DATABASE_URL", _LIVE_DSN)
        cls.backend = PostgresBackend()
        cls.repo = PostgresCorporateEventRepository(cls.backend)
        cls.resolver = CorporateEventResolver(cls.repo)
        cls.graph = cls.repo.load_graph()

    def test_only_logical_members_are_persisted(self) -> None:
        report = self.repo.diagnostics()
        self.assertEqual(report["stored_member_count"], report["logical_event_member_count"])
        self.assertLess(
            report["logical_event_member_count"],
            report["raw_contract_document_count"],
        )
        self.assertEqual(
            report["raw_contract_document_count"]
            - report["logical_event_member_count"],
            report["resolved_correction_documents_collapsed"],
        )

    def test_no_duplicate_membership_or_relations(self) -> None:
        report = self.repo.diagnostics()
        for guard in (
            "duplicate_membership_count",
            "duplicate_relation_count",
            "self_relation_count",
            "cross_company_member_count",
            "cross_company_relation_count",
            "cycle_count",
        ):
            self.assertEqual(report[guard], 0, guard)

    def test_every_collapsed_filing_resolves_to_its_representative(self) -> None:
        checked = 0
        for event in self.graph.events:
            for member in event.members:
                collapsed = (member.provenance or {}).get("collapsed_doc_ids") or []
                if len(collapsed) < 2:
                    continue
                for raw in collapsed:
                    found = self.repo.get_event(raw)
                    self.assertIsNotNone(found, raw)
                    self.assertEqual(found.event_id, event.event_id)
                    self.assertEqual(self.repo.get_member(raw).doc_id, member.doc_id)
                checked += 1
                if checked >= 25:
                    return
        self.assertGreater(checked, 0)

    def test_forward_and_reverse_traversal(self) -> None:
        terminated = [
            event
            for event in self.graph.events
            if event.is_terminated and event.resolution_status.value == "resolved"
        ]
        self.assertTrue(terminated)
        for event in terminated[:10]:
            contract = event.contract_members[0].doc_id
            termination = event.termination_members[0].doc_id
            self.assertIn(
                termination,
                [item.doc_id for item in self.resolver.get_terminations(contract)],
            )
            self.assertIn(
                contract,
                [item.doc_id for item in self.resolver.get_contract_documents(termination)],
            )

    def test_correction_provenance_survives_the_round_trip(self) -> None:
        for event in self.graph.events:
            for member in event.members:
                if not member.correction_group_id:
                    continue
                self.assertIn(member.doc_id, member.correction_chain)
                if str(member.correction_resolution_status.value) == "resolved":
                    self.assertEqual(member.canonical_doc_id, member.doc_id)
                    self.assertIn(member.root_doc_id, member.correction_chain)
                else:
                    # Never promoted to somebody else's verified latest.
                    self.assertEqual(member.canonical_doc_id, member.doc_id)
                return
