"""P0-C Step 2: the Evidence Slot domain model.

These tests pin the vocabulary Step 3 (planner) and Step 4 (bounded execution)
will build on, especially the two distinctions that are easy to get wrong:

* "nothing exists" is COMPLETE, "nothing was found" is INCOMPLETE;
* P0-C's SlotStatus is a different axis from P0-A/P0-B ``resolution_status``.
"""

from __future__ import annotations

import unittest

from app.reasoning.multi_document_plan import (
    DATE_FIELD_OPENED_AT,
    MEMBER_ROLE_CONTRACT,
    REASON_ALL_MEMBERS_FOUND,
    REASON_EMPTY_SET,
    REASON_MISSING_MEMBERS,
    REASON_NO_CORP_CODE,
    REASON_NO_DATE_RANGE,
    REASON_NO_FAMILY,
    REASON_UNRESOLVED_MEMBERS,
    EvidenceSlot,
    MultiDocumentPlan,
    SlotStatus,
    SlotType,
)


CORP = "00123456"


def _slot(**overrides):
    values = {
        "slot_id": "contracts_2025",
        "slot_type": SlotType.ENUMERATE_EVENTS,
        "corp_code": CORP,
        "event_family": "supply_contract",
        "member_role": MEMBER_ROLE_CONTRACT,
        "date_field": DATE_FIELD_OPENED_AT,
        "date_from": "2025-01-01",
        "date_to": "2026-01-01",
    }
    values.update(overrides)
    return EvidenceSlot(**values)


class SlotValidationTests(unittest.TestCase):
    def test_rejects_an_empty_slot_id(self) -> None:
        with self.assertRaises(ValueError):
            _slot(slot_id="  ")

    def test_rejects_an_unknown_date_field(self) -> None:
        with self.assertRaises(ValueError):
            _slot(date_field="signed_at")

    def test_rejects_an_inverted_interval(self) -> None:
        with self.assertRaises(ValueError):
            _slot(date_from="2026-01-01", date_to="2025-01-01")

    def test_event_state_slot_must_declare_a_dependency(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceSlot(slot_id="term", slot_type=SlotType.EVENT_STATE)

    def test_ids_are_normalized_and_deduplicated(self) -> None:
        slot = _slot(expected_ids=("b", "a", "a", "", "  "))
        self.assertEqual(slot.expected_ids, ("a", "b"))


class SlotStatusTests(unittest.TestCase):
    def test_all_expected_members_found_is_complete(self) -> None:
        slot = _slot(expected_ids=("a", "b"), found_ids=("a", "b")).resolve_status()
        self.assertIs(slot.status, SlotStatus.COMPLETE)
        self.assertEqual(slot.completeness_reason, REASON_ALL_MEMBERS_FOUND)

    def test_empty_expected_set_is_complete_not_incomplete(self) -> None:
        """"This company signed no contracts that year" is an answer."""

        slot = _slot().resolve_status()
        self.assertIs(slot.status, SlotStatus.COMPLETE)
        self.assertEqual(slot.completeness_reason, REASON_EMPTY_SET)

    def test_missing_member_is_incomplete(self) -> None:
        slot = _slot(expected_ids=("a", "b", "c"), found_ids=("a",)).resolve_status()
        self.assertIs(slot.status, SlotStatus.INCOMPLETE)
        self.assertEqual(slot.completeness_reason, REASON_MISSING_MEMBERS)
        self.assertEqual(slot.missing_ids, ("b", "c"))

    def test_unresolved_member_is_unresolved(self) -> None:
        slot = _slot(
            expected_ids=("a", "b"), found_ids=("a", "b"), unresolved_ids=("b",)
        ).resolve_status()
        self.assertIs(slot.status, SlotStatus.UNRESOLVED)
        self.assertEqual(slot.completeness_reason, REASON_UNRESOLVED_MEMBERS)

    def test_missing_outranks_unresolved(self) -> None:
        """More retrieval can fix a missing member; it cannot fix an
        unresolved one, so the actionable status wins."""

        slot = _slot(
            expected_ids=("a", "b"), found_ids=("a",), unresolved_ids=("a",)
        ).resolve_status()
        self.assertIs(slot.status, SlotStatus.INCOMPLETE)

    def test_termination_absent_is_complete_not_missing(self) -> None:
        """Checked every contract, found no termination: that is COMPLETE."""

        slot = EvidenceSlot(
            slot_id="termination_check",
            slot_type=SlotType.EVENT_STATE,
            depends_on=("contracts_2025",),
            expected_ids=("a", "b", "c"),
            found_ids=("a", "b", "c"),
        ).resolve_status()
        self.assertIs(slot.status, SlotStatus.COMPLETE)


class SlotApplicabilityTests(unittest.TestCase):
    def test_without_a_company_the_slot_is_not_applicable(self) -> None:
        slot = _slot(corp_code=None).resolve_status()
        self.assertIs(slot.status, SlotStatus.NOT_APPLICABLE)
        self.assertEqual(slot.completeness_reason, REASON_NO_CORP_CODE)

    def test_without_a_family_the_slot_is_not_applicable(self) -> None:
        slot = _slot(event_family=None).resolve_status()
        self.assertIs(slot.status, SlotStatus.NOT_APPLICABLE)
        self.assertEqual(slot.completeness_reason, REASON_NO_FAMILY)

    def test_without_a_bounded_range_the_slot_is_not_applicable(self) -> None:
        slot = _slot(date_from=None, date_to=None).resolve_status()
        self.assertIs(slot.status, SlotStatus.NOT_APPLICABLE)
        self.assertEqual(slot.completeness_reason, REASON_NO_DATE_RANGE)

    def test_document_slot_needs_a_group_or_subtype(self) -> None:
        slot = EvidenceSlot(
            slot_id="docs",
            slot_type=SlotType.ENUMERATE_DOCUMENTS,
            corp_code=CORP,
            date_from="2025-01-01",
            date_to="2026-01-01",
        ).resolve_status()
        self.assertIs(slot.status, SlotStatus.NOT_APPLICABLE)

    def test_document_slot_with_a_subtype_is_definable(self) -> None:
        slot = EvidenceSlot(
            slot_id="docs",
            slot_type=SlotType.ENUMERATE_DOCUMENTS,
            corp_code=CORP,
            doc_subtype="단일판매공급계약체결",
            date_from="2025-01-01",
            date_to="2026-01-01",
        )
        self.assertTrue(slot.is_definable)


class PlanTests(unittest.TestCase):
    def test_rejects_duplicate_slot_ids(self) -> None:
        with self.assertRaises(ValueError):
            MultiDocumentPlan(plan_type="enumeration", slots=(_slot(), _slot()))

    def test_rejects_a_dependency_on_an_unknown_slot(self) -> None:
        dependent = EvidenceSlot(
            slot_id="term",
            slot_type=SlotType.EVENT_STATE,
            depends_on=("nonexistent",),
        )
        with self.assertRaises(ValueError):
            MultiDocumentPlan(plan_type="enumeration_plus_event", slots=(dependent,))

    def test_accepts_a_resolvable_dependency(self) -> None:
        enumerate_slot = _slot()
        dependent = EvidenceSlot(
            slot_id="termination_check",
            slot_type=SlotType.EVENT_STATE,
            depends_on=("contracts_2025",),
        )
        plan = MultiDocumentPlan(
            plan_type="enumeration_plus_event", slots=(enumerate_slot, dependent)
        )
        self.assertEqual(len(plan.slots), 2)
        self.assertIsNotNone(plan.slot("termination_check"))

    def test_not_applicable_slots_do_not_block_completeness(self) -> None:
        plan = MultiDocumentPlan(
            plan_type="enumeration",
            slots=(
                _slot(slot_id="a").resolve_status(),
                _slot(slot_id="b", corp_code=None).resolve_status(),
            ),
        )
        self.assertTrue(plan.complete)

    def test_an_incomplete_slot_blocks_completeness(self) -> None:
        plan = MultiDocumentPlan(
            plan_type="enumeration",
            slots=(_slot(expected_ids=("a",), found_ids=()).resolve_status(),),
        )
        self.assertFalse(plan.complete)

    def test_rejects_negative_passes(self) -> None:
        with self.assertRaises(ValueError):
            MultiDocumentPlan(plan_type="enumeration", passes=-1)

    def test_trace_shape_carries_counts_not_ids(self) -> None:
        """The serving response is a fixed 5-field contract; the planner trace
        rides inside think_trace as an execution summary, not as evidence."""

        plan = MultiDocumentPlan(
            plan_type="enumeration_plus_event",
            slots=(_slot(expected_ids=("a", "b"), found_ids=("a", "b")).resolve_status(),),
            passes=2,
            stop_reason="all_slots_complete",
        )
        payload = plan.to_dict()
        self.assertEqual(payload["plan_type"], "enumeration_plus_event")
        self.assertEqual(payload["passes"], 2)
        self.assertTrue(payload["complete"])
        slot_payload = payload["slots"][0]
        self.assertEqual(slot_payload["expected_count"], 2)
        self.assertEqual(slot_payload["found_count"], 2)
        self.assertEqual(slot_payload["status"], "complete")
        self.assertNotIn("expected_ids", slot_payload)
        self.assertNotIn("found_ids", slot_payload)


if __name__ == "__main__":
    unittest.main()
