"""P0-C Step 4: the deterministic completeness executor.

The executor runs a plan; it never re-reads the question.  These tests use
counting repository doubles so the round-trip budget is pinned as a property,
not merely observed:

    Tier 1 enumeration + lifecycle  = 1 call
    Tier 2 enumeration + collapse   = 2 calls
    Tier 2 + lifecycle              = 3 calls

The invariant that matters most is §7: 732 of the corpus's 941 events carry
``resolution_status='unresolved'`` purely because a contract was filed once with
nothing to link to.  Treating that as an evidence gap would mark 79% of the
corpus unresolved, so only ``has_dangling_reference`` counts.
"""

from __future__ import annotations

import csv
import json
import unicodedata
import unittest
from pathlib import Path

from app.reasoning.corporate_event import CorporateEventState
from app.reasoning.corporate_event_graph import CorporateEventGraphUnavailable
from app.reasoning.multi_document_executor import (
    MAX_PLANNER_PASSES,
    STOP_ALL_SLOTS_COMPLETE,
    STOP_NO_DETERMINISTIC_ACTION,
    STOP_REPOSITORY_UNAVAILABLE,
    MultiDocumentExecutor,
)
from app.reasoning.multi_document_plan import (
    PLAN_NOT_APPLICABLE,
    REASON_EMPTY_SET,
    EvidenceSlot,
    MultiDocumentPlan,
    SlotStatus,
    SlotType,
)
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.interfaces import CandidateDocument, MetadataMatch


REPO = Path(__file__).resolve().parents[1]
GOLD60 = (
    REPO / "reports/evaluation/gold60/2026-08-21-agent-90pct/gold60_agent_questions.jsonl"
)
UNIVERSE = REPO / "data/corpus/universe.csv"
CORP = "00126478"


# ------------------------------------------------------------------ doubles


def _state(event_id, *, terminated=False, source="single_document", status="unresolved"):
    return CorporateEventState(
        doc_id=f"doc_{event_id}",
        event_id=event_id,
        corp_code=CORP,
        event_family="supply_contract",
        member_role="contract",
        lifecycle_status="terminated" if terminated else "open",
        resolution_status=status,
        canonical_doc_id=f"doc_{event_id}",
        member_count=1,
        opened_at="2025-03-01",
        resolution_source=source,
    )


class _EventRepo:
    def __init__(self, states=()):
        self.states = tuple(states)
        self.calls = []

    def enumerate_events(self, **kwargs):
        self.calls.append(("enumerate_events", kwargs))
        return self.states

    def event_states(self, doc_ids):
        self.calls.append(("event_states", tuple(doc_ids)))
        return {s.event_id: s for s in self.states if s.event_id in set(doc_ids)}


class _CorrectionRepo:
    def __init__(self, states=None):
        self._states = states or {}
        self.calls = []

    def document_states(self, doc_ids):
        self.calls.append(("document_states", tuple(doc_ids)))
        return {k: v for k, v in self._states.items() if k in set(doc_ids)}


class _Backend:
    def __init__(self, doc_ids=()):
        self.doc_ids = tuple(doc_ids)
        self.calls = []

    def enumerate_disclosures(self, **kwargs):
        self.calls.append(("enumerate_disclosures", kwargs))
        return tuple(
            CandidateDocument(doc_id=d, metadata={}, metadata_match=MetadataMatch())
            for d in self.doc_ids
        )


class _CorrState:
    def __init__(self, group, order, is_latest, status):
        self.correction_group_id = group
        self.correction_order = order
        self.is_latest = is_latest
        self.resolution_status = status

    @property
    def is_resolved_latest(self):
        return self.resolution_status == "resolved" and self.is_latest


def _enumerate_slot(slot_id="contracts", **kw):
    values = dict(
        slot_id=slot_id,
        slot_type=SlotType.ENUMERATE_EVENTS,
        corp_code=CORP,
        event_family="supply_contract",
        member_role="contract",
        date_field="opened_at",
        date_from="2025-01-01",
        date_to="2026-01-01",
    )
    values.update(kw)
    return EvidenceSlot(**values)


def _lifecycle_slot(depends="contracts"):
    return EvidenceSlot(
        slot_id="lifecycle",
        slot_type=SlotType.EVENT_STATE,
        corp_code=CORP,
        event_family="supply_contract",
        depends_on=(depends,),
    )


def _plan(*slots, plan_type="enumeration"):
    return MultiDocumentPlan(plan_type=plan_type, slots=tuple(slots))


# ------------------------------------------------------------ E1-E7 Tier 1


class Tier1ExecutionTests(unittest.TestCase):
    def test_e1_enumeration_of_three_contracts(self) -> None:
        repo = _EventRepo([_state("e1"), _state("e2"), _state("e3")])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot())
        )
        slot = result.plan.slots[0]
        self.assertEqual(slot.expected_count, 3)
        self.assertEqual(slot.found_count, 3)
        self.assertIs(slot.status, SlotStatus.COMPLETE)
        self.assertTrue(result.complete)
        self.assertEqual(result.plan.stop_reason, STOP_ALL_SLOTS_COMPLETE)

    def test_e4_lifecycle_open_terminated_open(self) -> None:
        repo = _EventRepo(
            [_state("A"), _state("B", terminated=True), _state("C")]
        )
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot(), _lifecycle_slot(),
                  plan_type="enumeration_plus_event")
        )
        lifecycle = result.plan.slot("lifecycle")
        self.assertEqual(lifecycle.expected_count, 3)
        self.assertEqual(lifecycle.found_count, 3)
        self.assertIs(lifecycle.status, SlotStatus.COMPLETE)
        outcome = result.outcome("lifecycle")
        self.assertEqual(outcome.terminated_count, 1)
        self.assertEqual(outcome.terminated_ids, ("B",))
        self.assertEqual(outcome.open_count, 2)
        self.assertTrue(result.complete)

    def test_tier1_lifecycle_costs_no_extra_round_trip(self) -> None:
        """§9/§24 -- enumerate_events already returned lifecycle_status."""

        repo = _EventRepo([_state("A"), _state("B", terminated=True)])
        MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot(), _lifecycle_slot(),
                  plan_type="enumeration_plus_event")
        )
        self.assertEqual(len(repo.calls), 1)
        self.assertEqual(repo.calls[0][0], "enumerate_events")

    def test_e5_empty_set_is_complete(self) -> None:
        """"이 회사는 2025년에 계약을 체결하지 않았다" is an answer, not a miss."""

        repo = _EventRepo([])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot(), _lifecycle_slot(),
                  plan_type="enumeration_plus_event")
        )
        enumerate_slot, lifecycle = result.plan.slots
        self.assertEqual(enumerate_slot.expected_count, 0)
        self.assertIs(enumerate_slot.status, SlotStatus.COMPLETE)
        self.assertEqual(enumerate_slot.completeness_reason, REASON_EMPTY_SET)
        self.assertEqual(lifecycle.expected_count, 0)
        self.assertIs(lifecycle.status, SlotStatus.COMPLETE)
        self.assertTrue(result.complete)

    def test_e6_dangling_reference_is_unresolved(self) -> None:
        repo = _EventRepo(
            [_state("A"), _state("B", source="related_reference_not_in_corpus")]
        )
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot())
        )
        slot = result.plan.slots[0]
        self.assertEqual(slot.unresolved_ids, ("B",))
        self.assertIs(slot.status, SlotStatus.UNRESOLVED)
        self.assertFalse(result.complete)
        self.assertEqual(result.plan.stop_reason, STOP_NO_DETERMINISTIC_ACTION)

    def test_e6b_contract_period_not_in_corpus_is_unresolved(self) -> None:
        repo = _EventRepo([_state("A", source="contract_period_not_in_corpus")])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot())
        )
        self.assertEqual(result.plan.slots[0].unresolved_ids, ("A",))

    def test_e7_single_document_is_not_unresolved(self) -> None:
        """732 of 941 real events look like this. Flagging them would mark 79%
        of the corpus as missing evidence."""

        repo = _EventRepo([_state("A", source="single_document", status="unresolved")])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot())
        )
        slot = result.plan.slots[0]
        self.assertEqual(slot.unresolved_ids, ())
        self.assertIs(slot.status, SlotStatus.COMPLETE)

    def test_terminated_does_not_affect_completeness(self) -> None:
        repo = _EventRepo([_state("A", terminated=True), _state("B", terminated=True)])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot(), _lifecycle_slot(),
                  plan_type="enumeration_plus_event")
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.outcome("lifecycle").terminated_count, 2)


# ------------------------------------------------------- E2/E3/E8 Tier 2


class Tier2ExecutionTests(unittest.TestCase):
    def _slot(self):
        return EvidenceSlot(
            slot_id="documents",
            slot_type=SlotType.ENUMERATE_DOCUMENTS,
            corp_code=CORP,
            event_family="supply_contract",
            doc_group="exchange",
            doc_subtype="단일판매공급계약체결",
            date_field="rcept_dt",
            date_from="2025-01-01",
            date_to="2026-01-01",
        )

    def test_e2_resolved_correction_chain_counts_once(self) -> None:
        backend = _Backend(["a0", "a1", "a2"])
        corrections = _CorrectionRepo(
            {
                "a0": _CorrState("g1", 0, False, "resolved"),
                "a1": _CorrState("g1", 1, False, "resolved"),
                "a2": _CorrState("g1", 2, True, "resolved"),
            }
        )
        result = MultiDocumentExecutor(
            disclosure_backend=backend, correction_repository=corrections
        ).execute(_plan(self._slot()))
        slot = result.plan.slots[0]
        self.assertEqual(slot.expected_count, 1)
        self.assertEqual(slot.expected_ids, ("a2",))
        self.assertIs(slot.status, SlotStatus.COMPLETE)

    def test_e8_ambiguous_correction_is_not_folded(self) -> None:
        backend = _Backend(["a0", "a1"])
        corrections = _CorrectionRepo(
            {
                "a0": _CorrState("g1", 0, True, "resolved"),
                "a1": _CorrState("g2", 0, True, "ambiguous"),
            }
        )
        result = MultiDocumentExecutor(
            disclosure_backend=backend, correction_repository=corrections
        ).execute(_plan(self._slot()))
        slot = result.plan.slots[0]
        self.assertEqual(slot.expected_count, 2)
        self.assertIn("a1", slot.unresolved_ids)
        self.assertIs(slot.status, SlotStatus.UNRESOLVED)

    def test_tier2_uses_exactly_two_round_trips(self) -> None:
        backend = _Backend([f"d{i}" for i in range(40)])
        corrections = _CorrectionRepo({})
        MultiDocumentExecutor(
            disclosure_backend=backend, correction_repository=corrections
        ).execute(_plan(self._slot()))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(len(corrections.calls), 1)
        # One batch for all 40, never one per document.
        self.assertEqual(len(corrections.calls[0][1]), 40)

    def test_tier2_with_lifecycle_uses_three_round_trips(self) -> None:
        backend = _Backend(["d0", "d1"])
        corrections = _CorrectionRepo({})
        events = _EventRepo([_state("d0"), _state("d1", terminated=True)])
        result = MultiDocumentExecutor(
            disclosure_backend=backend,
            correction_repository=corrections,
            event_repository=events,
        ).execute(
            _plan(self._slot(), _lifecycle_slot(depends="documents"),
                  plan_type="enumeration_plus_event")
        )
        total = len(backend.calls) + len(corrections.calls) + len(events.calls)
        self.assertEqual(total, 3)
        self.assertEqual(events.calls[0][0], "event_states")
        self.assertEqual(result.outcome("lifecycle").terminated_count, 1)

    def test_empty_enumeration_skips_the_correction_query(self) -> None:
        backend = _Backend([])
        corrections = _CorrectionRepo({})
        result = MultiDocumentExecutor(
            disclosure_backend=backend, correction_repository=corrections
        ).execute(_plan(self._slot()))
        self.assertEqual(len(corrections.calls), 0)
        self.assertIs(result.plan.slots[0].status, SlotStatus.COMPLETE)

    def test_document_without_lifecycle_membership_is_unresolved(self) -> None:
        backend = _Backend(["d0", "d1"])
        corrections = _CorrectionRepo({})
        events = _EventRepo([_state("d0")])   # d1 has no membership
        result = MultiDocumentExecutor(
            disclosure_backend=backend,
            correction_repository=corrections,
            event_repository=events,
        ).execute(
            _plan(self._slot(), _lifecycle_slot(depends="documents"),
                  plan_type="enumeration_plus_event")
        )
        lifecycle = result.plan.slot("lifecycle")
        self.assertIn("d1", lifecycle.unresolved_ids)
        self.assertIs(lifecycle.status, SlotStatus.UNRESOLVED)


# ------------------------------------------------------- E9/E10 structure


class PlanStructureTests(unittest.TestCase):
    def test_e9_missing_dependency_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiDocumentPlan(
                plan_type="enumeration_plus_event",
                slots=(
                    EvidenceSlot(
                        slot_id="lifecycle",
                        slot_type=SlotType.EVENT_STATE,
                        depends_on=("nope",),
                    ),
                ),
            )

    def test_e10_dependency_cycle_is_rejected(self) -> None:
        a = EvidenceSlot(slot_id="a", slot_type=SlotType.EVENT_STATE, depends_on=("b",))
        b = EvidenceSlot(slot_id="b", slot_type=SlotType.EVENT_STATE, depends_on=("a",))
        with self.assertRaises(ValueError):
            MultiDocumentPlan(plan_type="enumeration_plus_event", slots=(a, b))

    def test_self_dependency_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiDocumentPlan(
                plan_type="enumeration_plus_event",
                slots=(
                    EvidenceSlot(
                        slot_id="a", slot_type=SlotType.EVENT_STATE, depends_on=("a",)
                    ),
                ),
            )

    def test_duplicate_slot_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _plan(_enumerate_slot(), _enumerate_slot())

    def test_declaration_order_is_not_assumed(self) -> None:
        """Lifecycle declared first still runs after its dependency."""

        repo = _EventRepo([_state("A", terminated=True)])
        plan = _plan(_lifecycle_slot(), _enumerate_slot(),
                     plan_type="enumeration_plus_event")
        result = MultiDocumentExecutor(event_repository=repo).execute(plan)
        self.assertEqual(result.plan.slot("lifecycle").expected_count, 1)
        self.assertTrue(result.complete)

    def test_pass_count_is_bounded(self) -> None:
        repo = _EventRepo([_state("A")])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot(), _lifecycle_slot(),
                  plan_type="enumeration_plus_event")
        )
        self.assertEqual(result.plan.passes, 2)
        self.assertLessEqual(result.plan.passes, MAX_PLANNER_PASSES)

    def test_enumeration_only_plan_uses_one_pass(self) -> None:
        repo = _EventRepo([_state("A")])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot())
        )
        self.assertEqual(result.plan.passes, 1)

    def test_input_plan_is_not_mutated(self) -> None:
        repo = _EventRepo([_state("A")])
        plan = _plan(_enumerate_slot())
        before = plan.to_dict()
        MultiDocumentExecutor(event_repository=repo).execute(plan)
        self.assertEqual(plan.to_dict(), before)
        self.assertIs(plan.slots[0].status, SlotStatus.PENDING)

    def test_execution_is_deterministic(self) -> None:
        states = [_state("b"), _state("a"), _state("c", terminated=True)]
        plan = _plan(_enumerate_slot(), _lifecycle_slot(),
                     plan_type="enumeration_plus_event")
        first = MultiDocumentExecutor(event_repository=_EventRepo(states)).execute(plan)
        second = MultiDocumentExecutor(event_repository=_EventRepo(states)).execute(plan)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.plan.slots[0].expected_ids, ("a", "b", "c"))


# --------------------------------------------------------- non-applied plan


class NonAppliedPlanTests(unittest.TestCase):
    def test_declined_plan_touches_no_repository(self) -> None:
        repo, corrections, backend = _EventRepo(), _CorrectionRepo(), _Backend()
        plan = MultiDocumentPlan(
            plan_type=PLAN_NOT_APPLICABLE, stop_reason="no_set_intent"
        )
        result = MultiDocumentExecutor(
            event_repository=repo,
            correction_repository=corrections,
            disclosure_backend=backend,
        ).execute(plan)
        self.assertFalse(result.applied)
        self.assertEqual(result.plan.stop_reason, "no_set_intent")
        self.assertEqual(repo.calls + corrections.calls + backend.calls, [])


class UnavailableRepositoryTests(unittest.TestCase):
    def test_graph_unavailable_degrades_instead_of_raising(self) -> None:
        class _Broken:
            def enumerate_events(self, **kwargs):
                raise CorporateEventGraphUnavailable("db/007 not applied")

        result = MultiDocumentExecutor(event_repository=_Broken()).execute(
            _plan(_enumerate_slot())
        )
        self.assertEqual(result.plan.stop_reason, STOP_REPOSITORY_UNAVAILABLE)
        self.assertFalse(result.complete)
        self.assertEqual(result.unavailable_reason, "CorporateEventGraphUnavailable")

    def test_programming_errors_propagate(self) -> None:
        class _Buggy:
            def enumerate_events(self, **kwargs):
                raise TypeError("bad call")

        with self.assertRaises(TypeError):
            MultiDocumentExecutor(event_repository=_Buggy()).execute(
                _plan(_enumerate_slot())
            )


# ------------------------------------------------------- architecture rules


class ExecutorPurityTests(unittest.TestCase):
    SOURCE = (REPO / "app/reasoning/multi_document_executor.py").read_text(
        encoding="utf-8"
    )

    def test_no_sql_and_no_driver(self) -> None:
        for forbidden in ("SELECT", "INSERT", "UPDATE", "DELETE", "psycopg"):
            self.assertNotIn(forbidden, self.SOURCE, forbidden)

    def test_execute_takes_only_a_plan(self) -> None:
        import inspect

        signature = inspect.signature(MultiDocumentExecutor.execute)
        self.assertEqual(list(signature.parameters), ["self", "plan"])

    def test_no_question_text_branching(self) -> None:
        """Step 4 must not re-derive anything the planner already decided.

        Prose may mention the question; code may not read it.  What is checked
        is the absence of any handle on question text and of any Korean literal
        the executor could branch on.
        """

        import re

        code = chr(10).join(
            line for line in self.SOURCE.splitlines() if not line.lstrip().startswith("#")
        )
        for forbidden in ("raw_query", "query_plan", ".question", "question="):
            self.assertNotIn(forbidden, code, forbidden)
        korean = re.findall(r"[가-힣]+", code)
        self.assertEqual(korean, [], f"executor contains Korean literals: {korean[:5]}")

    def test_trace_carries_counts_not_ids(self) -> None:
        repo = _EventRepo([_state("A"), _state("B", terminated=True)])
        result = MultiDocumentExecutor(event_repository=repo).execute(
            _plan(_enumerate_slot(), _lifecycle_slot(),
                  plan_type="enumeration_plus_event")
        )
        payload = result.to_dict()
        text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("expected_ids", text)
        self.assertNotIn("terminated_ids", text)
        self.assertEqual(payload["lifecycle"][0]["terminated_count"], 1)


# ------------------------------------------------- planner -> executor (§22)


def _company_resolver():
    names = {}
    if UNIVERSE.exists():
        with UNIVERSE.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for key in ("corp_name", "listed_name"):
                    value = (row.get(key) or "").replace(" ", "")
                    if value:
                        names[unicodedata.normalize("NFC", value)] = row["corp_code"]
    else:  # pragma: no cover
        names = {"삼성중공업": CORP}

    def resolve(query: str):
        text = unicodedata.normalize("NFC", (query or "").replace(" ", ""))
        best = None
        for name, code in names.items():
            if name in text and (best is None or len(name) > len(best[0])):
                best = (name, code)
        return {"corp_code": best[1], "corp_name": best[0]} if best else None

    return resolve


class PlannerToExecutorTests(unittest.TestCase):
    """§22 -- the official reference question, end to domain (no serving)."""

    def setUp(self) -> None:
        self.understanding = QueryUnderstanding(company_resolver=_company_resolver())
        self.planner = MultiDocumentPlanner()

    def test_official_reference_question(self) -> None:
        question = "삼성중공업이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"
        plan = self.planner.plan(question, self.understanding.understand(question))
        self.assertEqual(plan.plan_type, "enumeration_plus_event")

        states = [_state(f"e{i:02d}") for i in range(12)] + [
            _state("e12", terminated=True),
            _state("e13", terminated=True),
        ]
        repo = _EventRepo(states)
        result = MultiDocumentExecutor(event_repository=repo).execute(plan)

        enumerate_slot = result.plan.slot("contracts")
        lifecycle = result.plan.slot("lifecycle")
        self.assertEqual(enumerate_slot.expected_count, 14)
        self.assertEqual(enumerate_slot.found_count, 14)
        self.assertIs(enumerate_slot.status, SlotStatus.COMPLETE)
        self.assertEqual(lifecycle.expected_count, 14)
        self.assertEqual(lifecycle.found_count, 14)
        self.assertIs(lifecycle.status, SlotStatus.COMPLETE)
        self.assertEqual(result.outcome("lifecycle").terminated_count, 2)
        self.assertEqual(lifecycle.unresolved_ids, ())
        self.assertTrue(result.complete)
        self.assertEqual(result.plan.stop_reason, STOP_ALL_SLOTS_COMPLETE)
        self.assertLessEqual(result.plan.passes, MAX_PLANNER_PASSES)
        self.assertEqual(len(repo.calls), 1)

    def test_official_question_with_no_contracts(self) -> None:
        """§14 -- an empty 2025 set answers "존재하지 않음", not "not found"."""

        question = "삼성중공업이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"
        plan = self.planner.plan(question, self.understanding.understand(question))
        result = MultiDocumentExecutor(event_repository=_EventRepo([])).execute(plan)
        self.assertTrue(result.complete)
        self.assertEqual(result.plan.stop_reason, STOP_ALL_SLOTS_COMPLETE)
        self.assertEqual(result.outcome("lifecycle").terminated_count, 0)


class Gold60ExecutorSafetyTests(unittest.TestCase):
    """§25 -- 60 questions, zero repository calls."""

    def test_no_repository_call_for_any_gold60_question(self) -> None:
        if not GOLD60.exists():
            self.skipTest("gold60 artifact not present")
        understanding = QueryUnderstanding(company_resolver=_company_resolver())
        planner = MultiDocumentPlanner()
        repo, corrections, backend = _EventRepo(), _CorrectionRepo(), _Backend()
        executor = MultiDocumentExecutor(
            event_repository=repo,
            correction_repository=corrections,
            disclosure_backend=backend,
        )
        rows = [
            json.loads(line)
            for line in GOLD60.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 60)
        for row in rows:
            question = row["question"]
            plan = planner.plan(question, understanding.understand(question))
            result = executor.execute(plan)
            self.assertFalse(result.applied, row["question_id"])
        self.assertEqual(repo.calls + corrections.calls + backend.calls, [])


if __name__ == "__main__":
    unittest.main()
