"""P0-C Step 6: semantic preservation from deterministic state to Korean answer.

The counts being right is not the gate.  The gate is that the sentence built
from them still means the same thing, and above all that ``undetermined`` never
renders as ``none``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.reasoning.multi_document_evidence import (
    LIFECYCLE_EXISTS,
    LIFECYCLE_NONE,
    LIFECYCLE_NO_MEMBERS,
    LIFECYCLE_UNDETERMINED,
    MultiDocumentFacts,
)
from app.reasoning.multi_document_executor import (
    MultiDocumentExecution,
    MultiDocumentExecutor,
)
from app.reasoning.multi_document_semantics import (
    check_answer,
    has_confident_negative,
    has_uncertainty,
)

from tests.test_multi_document_serving import (
    COUNT_Q,
    LIFECYCLE_Q,
    _EventRepo,
    _pipeline,
    _state,
)


REPO = Path(__file__).resolve().parents[1]
EVAL_SET = REPO / "data/evaluation/p0c_official_style_v1.json"


def _facts(**kw):
    values = dict(plan_type="enumeration_plus_event", complete=True, logical_count=14)
    values.update(kw)
    return MultiDocumentFacts(**values)


class StatementTests(unittest.TestCase):
    def test_exists_states_the_count(self) -> None:
        text = _facts(lifecycle_answer=LIFECYCLE_EXISTS, terminated_count=2).statement()
        self.assertIn("14", text)
        self.assertIn("2건", text)
        self.assertFalse(has_confident_negative(text))

    def test_none_is_a_negative_without_hedging(self) -> None:
        text = _facts(lifecycle_answer=LIFECYCLE_NONE).statement()
        self.assertIn("확인되지 않았습니다", text)
        self.assertFalse(has_uncertainty(text))

    def test_undetermined_is_never_a_negative(self) -> None:
        text = _facts(
            lifecycle_answer=LIFECYCLE_UNDETERMINED, complete=False, unresolved_count=1
        ).statement()
        self.assertTrue(has_uncertainty(text))
        self.assertFalse(has_confident_negative(text))

    def test_no_members_is_not_a_retrieval_failure(self) -> None:
        text = _facts(lifecycle_answer=LIFECYCLE_NO_MEMBERS, logical_count=0).statement()
        self.assertNotIn("찾지 못", text)
        self.assertNotIn("검색", text)

    def test_incomplete_cardinality_admits_the_gap(self) -> None:
        text = _facts(
            plan_type="enumeration", complete=False, logical_count=18, unresolved_count=1
        ).statement()
        self.assertTrue(has_uncertainty(text))

    def test_counts_are_never_recomputed(self) -> None:
        """Every number in the sentence comes from the facts object."""

        text = _facts(
            lifecycle_answer=LIFECYCLE_EXISTS, logical_count=9, terminated_count=2
        ).statement()
        self.assertIn("9건", text)
        self.assertIn("2건", text)
        self.assertNotIn("11", text)

    def test_summary_alone_is_supported_deterministic_answer_material(self) -> None:
        from app.generation.answer_generator import CitationAwareAnswerGenerator
        from app.reasoning.answer_composer import AnswerDraft, AnswerSection

        facts = _facts(lifecycle_answer=LIFECYCLE_NO_MEMBERS, logical_count=0)
        draft = AnswerDraft(
            question=LIFECYCLE_Q,
            task_type="corporate_event",
            answer_sections=(
                AnswerSection(
                    title="계약 확인 결과",
                    content={"summary": facts.statement(), **facts.to_dict()},
                    supporting_evidence_ids=(),
                ),
            ),
            evidence_references=(),
            citations=(),
            ambiguity={},
            warnings=(),
            confidence={"level": "medium", "score": 0.5},
            answerable=True,
        )
        generated = CitationAwareAnswerGenerator().generate(draft)
        self.assertTrue(generated.answerable)
        self.assertNotIn("확인되지 않은 정보가 있습니다", generated.answer_text)


class GuardTests(unittest.TestCase):
    def test_hedged_wording_is_not_a_confident_negative(self) -> None:
        """The trap: the correct undetermined wording contains 없습니다."""

        text = "해지 여부를 단정할 수 없습니다."
        self.assertFalse(has_confident_negative(text))
        self.assertTrue(has_uncertainty(text))

    def test_real_confident_negatives_are_caught(self) -> None:
        for text in (
            "해지된 계약은 없습니다.",
            "해지된 계약이 없습니다.",
            "해지되지 않았습니다.",
            "해지된 계약은 0건입니다.",
            "해지 사실이 없습니다.",
        ):
            self.assertTrue(has_confident_negative(text), text)

    def test_undetermined_rejects_a_confident_negative(self) -> None:
        verdict = check_answer(LIFECYCLE_UNDETERMINED, "해지된 계약은 없습니다.")
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "confident_negative_for_undetermined")

    def test_undetermined_requires_an_uncertainty_marker(self) -> None:
        verdict = check_answer(LIFECYCLE_UNDETERMINED, "계약 14건을 확인했습니다.")
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.reason, "no_uncertainty_marker")

    def test_exists_rejects_a_negative(self) -> None:
        self.assertFalse(check_answer(LIFECYCLE_EXISTS, "해지된 계약은 없습니다.").ok)

    def test_none_rejects_hedging(self) -> None:
        self.assertFalse(
            check_answer(LIFECYCLE_NONE, "해지 여부를 확정할 수 없습니다.").ok
        )

    def test_every_generated_statement_passes_its_own_guard(self) -> None:
        cases = [
            (LIFECYCLE_EXISTS, dict(terminated_count=2)),
            (LIFECYCLE_NONE, {}),
            (LIFECYCLE_UNDETERMINED, dict(complete=False, unresolved_count=1)),
            (LIFECYCLE_NO_MEMBERS, dict(logical_count=0)),
        ]
        for answer, extra in cases:
            facts = _facts(lifecycle_answer=answer, **extra)
            verdict = check_answer(answer, facts.statement())
            self.assertTrue(verdict.ok, f"{answer}: {verdict.reason}")


class EndToEndSemanticTests(unittest.TestCase):
    """The rendered /answer payload, not just the facts object."""

    def _answer(self, contracts, question=LIFECYCLE_Q, terminations=()):
        return _pipeline(_EventRepo(contracts, terminations)).answer("S", question)

    def test_undetermined_never_reaches_the_reader_as_none(self) -> None:
        contracts = [_state(f"e{i:02d}") for i in range(13)] + [
            _state("e13", source="related_reference_not_in_corpus")
        ]
        payload = self._answer(contracts)
        trace = payload["think_trace"]["multi_document_planner"]
        self.assertEqual(trace["lifecycle_answer"], LIFECYCLE_UNDETERMINED)
        verdict = check_answer(LIFECYCLE_UNDETERMINED, payload["answer"])
        self.assertTrue(verdict.ok, verdict.reason)
        self.assertFalse(has_confident_negative(payload["answer"]))

    def test_exists_reaches_the_reader(self) -> None:
        contracts = [_state("e0"), _state("e1", terminated=True)]
        payload = self._answer(
            contracts, terminations=[_state("e1", role="termination", doc_id="term_e1")]
        )
        self.assertEqual(
            payload["think_trace"]["multi_document_planner"]["lifecycle_answer"],
            LIFECYCLE_EXISTS,
        )
        self.assertTrue(check_answer(LIFECYCLE_EXISTS, payload["answer"]).ok)

    def test_none_reaches_the_reader(self) -> None:
        payload = self._answer([_state("e0"), _state("e1")])
        self.assertTrue(check_answer(LIFECYCLE_NONE, payload["answer"]).ok)

    def test_no_members_reaches_the_reader(self) -> None:
        payload = self._answer([])
        self.assertTrue(check_answer(LIFECYCLE_NO_MEMBERS, payload["answer"]).ok)
        self.assertIn(
            "해당 기간에 조건에 맞는 계약이 확인되지 않았습니다.",
            payload["answer"],
        )
        self.assertNotIn("확인되지 않은 정보가 있습니다", payload["answer"])

    def test_statement_leads_the_answer(self) -> None:
        payload = self._answer([_state("e0")], question=COUNT_Q)
        first_block = payload["answer"].split("\n\n")[0]
        self.assertIn("계약 확인 결과", first_block)
        self.assertIn("건", first_block)

    def test_internal_labels_never_surface(self) -> None:
        payload = self._answer([_state("e0")], question=COUNT_Q)
        for leaked in ("lifecycle_answer", "bare_contract_fallback", "enumerate_events",
                       "slot_id", "expected_count"):
            self.assertNotIn(leaked, payload["answer"], leaked)


class VerbalizerInvarianceTests(unittest.TestCase):
    """HCX cannot rewrite a P0-C answer, configured or not.

    The verbalizer only speaks when ``build_compact_claim`` yields a single
    verified event claim.  A multi-document answer never produces one, so the
    deterministic sentence is what the reader sees -- which is why the semantic
    guarantee does not depend on model behaviour.
    """

    def test_p0c_answer_yields_no_compact_claim(self) -> None:
        from app.generation.compact_claim import build_compact_claim

        pipeline = _pipeline(_EventRepo([_state("e0"), _state("e1", terminated=True)]))
        plan, execution = pipeline._retrieve(LIFECYCLE_Q)
        multi = pipeline._multi_document(LIFECYCLE_Q, plan, execution)
        result = pipeline.orchestrator.run(
            LIFECYCLE_Q, plan, execution, multi_document=multi
        )
        claim = build_compact_claim(
            result.answer_draft, result.resolution,
            task_type=result.task_decision.task_type,
        )
        self.assertIsNone(claim)

    def test_verbalizer_returns_the_deterministic_text(self) -> None:
        from app.generation.hcx_verbalizer import HcxVerbalizer

        pipeline = _pipeline(_EventRepo([_state("e0")]))
        payload = pipeline.answer("V", COUNT_Q)
        self.assertIn("건", payload["answer"])
        # Whatever the verbalizer status, the statement survives verbatim.
        self.assertIn("조건에 해당하는 계약", payload["answer"])

    def test_final_guard_rejects_confident_negative_for_undetermined(self) -> None:
        from app.generation.hcx_verbalizer import VerbalizationOutcome

        class _SemanticDrift:
            def verbalize(self, generated, **kwargs):
                return VerbalizationOutcome("해지된 계약은 없습니다.", "success")

        contracts = [_state(f"e{i:02d}") for i in range(13)] + [
            _state("e13", source="related_reference_not_in_corpus")
        ]
        pipeline = _pipeline(_EventRepo(contracts))
        pipeline.verbalizer = _SemanticDrift()
        payload = pipeline.answer("G", LIFECYCLE_Q)

        self.assertEqual(payload["think_trace"]["hcx_status"], "fallback_semantic_guard")
        self.assertTrue(check_answer(LIFECYCLE_UNDETERMINED, payload["answer"]).ok)
        self.assertFalse(has_confident_negative(payload["answer"]))


class OpeningMappingTests(unittest.TestCase):
    """§10 -- the event->opening link must be explicit, not positional."""

    def test_shuffled_order_still_pairs_event_with_its_opening(self) -> None:
        # event ids and document ids sort in *opposite* orders, so any
        # positional pairing would cross-link every one of them.
        states = [
            _state("e01", doc_id="contract_z"),
            _state("e02", doc_id="contract_y"),
            _state("e03", doc_id="contract_x", terminated=True),
        ]
        execution = MultiDocumentExecutor(
            event_repository=_EventRepo(list(reversed(states)))
        ).execute(_lifecycle_plan())
        mapping = execution.opening_documents
        self.assertEqual(mapping["e01"], "contract_z")
        self.assertEqual(mapping["e02"], "contract_y")
        self.assertEqual(mapping["e03"], "contract_x")

    def test_terminated_event_hydrates_its_own_opening(self) -> None:
        states = [
            _state("e01", doc_id="contract_z"),
            _state("e02", doc_id="contract_y"),
            _state("e03", doc_id="contract_x", terminated=True),
        ]
        payload = _pipeline(
            _EventRepo(states, [_state("e03", role="termination", doc_id="term_e03")])
        ).answer("M", LIFECYCLE_Q)
        doc_ids = [row["doc_id"] for row in payload["retrieved_context"]]
        self.assertIn("term_e03", doc_ids)
        # The opening that belongs to e03 -- not whichever sorted first.
        self.assertIn("contract_x", doc_ids)
        self.assertLess(doc_ids.index("term_e03"), doc_ids.index("contract_x"))

    def test_positional_zip_is_absent_from_opening_mapping(self) -> None:
        source = (REPO / "app/reasoning/multi_document_evidence.py").read_text(
            encoding="utf-8"
        )
        method = source.split("def _opening_documents", 1)[1].split(
            "def _termination_documents", 1
        )[0]
        self.assertNotIn("zip(", method)


class CitationCompletenessTests(unittest.TestCase):
    """The summary cites the filings behind its actual lifecycle claim."""

    def test_exists_summary_cites_termination_and_opening_filings(self) -> None:
        terminations = [_state("e01", role="termination", doc_id="term_e01")]
        pipeline = _pipeline(
            _EventRepo(
                [_state("e00"), _state("e01", doc_id="contract_e01", terminated=True)],
                terminations,
            )
        )
        plan, execution = pipeline._retrieve(LIFECYCLE_Q)
        multi = pipeline._multi_document(LIFECYCLE_Q, plan, execution)
        result = pipeline.orchestrator.run(
            LIFECYCLE_Q, plan, execution, multi_document=multi
        )
        generated = pipeline.generator.generate(result.answer_draft)
        summary = next(
            section for section in generated.sections if section.title == "계약 확인 결과"
        )
        cited_docs = {
            citation.doc_id
            for citation in generated.citations
            if citation.citation_id in summary.citations
        }
        self.assertIn("term_e01", cited_docs)
        self.assertIn("contract_e01", cited_docs)

    def test_unresolved_opening_is_prioritized_inside_presentation_budget(self) -> None:
        contracts = [_state(f"e{i:02d}") for i in range(17)] + [
            _state("e17", source="related_reference_not_in_corpus")
        ]
        pipeline = _pipeline(_EventRepo(contracts))
        plan, execution = pipeline._retrieve(LIFECYCLE_Q)
        multi = pipeline._multi_document(LIFECYCLE_Q, plan, execution)
        self.assertIn("contract_e17", multi.added_doc_ids)

    def test_tier2_unresolved_document_is_prioritized(self) -> None:
        from app.reasoning.multi_document_evidence import MultiDocumentEvidenceBuilder
        from app.reasoning.multi_document_plan import (
            EvidenceSlot,
            MultiDocumentPlan,
            SlotType,
        )

        doc_ids = tuple(f"d{i:02d}" for i in range(18))
        enumeration = EvidenceSlot(
            slot_id="documents",
            slot_type=SlotType.ENUMERATE_DOCUMENTS,
            corp_code="00126478",
            event_family="supply_contract",
            doc_group="exchange",
            doc_subtype="단일판매공급계약체결",
            date_field="rcept_dt",
            date_from="2025-01-01",
            date_to="2026-01-01",
        ).resolve_status_with(
            expected_ids=doc_ids,
            found_ids=doc_ids,
            unresolved_ids=("d17",),
        )
        lifecycle = EvidenceSlot(
            slot_id="lifecycle",
            slot_type=SlotType.EVENT_STATE,
            corp_code="00126478",
            event_family="supply_contract",
            depends_on=("documents",),
        ).resolve_status_with(
            expected_ids=doc_ids,
            found_ids=doc_ids,
            unresolved_ids=("d17",),
        )
        execution = MultiDocumentExecution(
            plan=MultiDocumentPlan(
                plan_type="enumeration_plus_event",
                slots=(enumeration, lifecycle),
            ),
            document_ids={"documents": doc_ids},
        )
        ordered = MultiDocumentEvidenceBuilder()._ordered_doc_ids(execution)
        self.assertEqual(ordered[0], "d17")

    def test_many_terminations_each_keep_an_opening_citation(self) -> None:
        contracts = [
            _state(f"e{i:02d}", doc_id=f"contract_e{i:02d}", terminated=True)
            for i in range(5)
        ]
        terminations = [
            _state(f"e{i:02d}", role="termination", doc_id=f"term_e{i:02d}")
            for i in range(5)
        ]
        pipeline = _pipeline(_EventRepo(contracts, terminations))
        plan, execution = pipeline._retrieve(LIFECYCLE_Q)
        multi = pipeline._multi_document(LIFECYCLE_Q, plan, execution)
        result = pipeline.orchestrator.run(
            LIFECYCLE_Q, plan, execution, multi_document=multi
        )
        generated = pipeline.generator.generate(result.answer_draft)
        summary = next(
            section for section in generated.sections if section.title == "계약 확인 결과"
        )
        cited_docs = {
            citation.doc_id
            for citation in generated.citations
            if citation.citation_id in summary.citations
        }
        self.assertTrue({f"term_e{i:02d}" for i in range(5)} <= cited_docs)
        self.assertTrue({f"contract_e{i:02d}" for i in range(5)} <= cited_docs)


def _lifecycle_plan():
    from app.reasoning.multi_document_plan import (
        EvidenceSlot,
        MultiDocumentPlan,
        SlotType,
    )

    enumerate_slot = EvidenceSlot(
        slot_id="contracts", slot_type=SlotType.ENUMERATE_EVENTS,
        corp_code="00126478", event_family="supply_contract",
        member_role="contract", date_field="opened_at",
        date_from="2025-01-01", date_to="2026-01-01",
    )
    lifecycle = EvidenceSlot(
        slot_id="lifecycle", slot_type=SlotType.EVENT_STATE,
        corp_code="00126478", event_family="supply_contract",
        depends_on=("contracts",),
    )
    return MultiDocumentPlan(
        plan_type="enumeration_plus_event", slots=(enumerate_slot, lifecycle)
    )


class EvaluationSetTests(unittest.TestCase):
    """The frozen official-style set is a checked-in artifact."""

    def test_set_is_present_and_covers_every_branch(self) -> None:
        if not EVAL_SET.exists():
            self.skipTest("evaluation set not present")
        cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]
        self.assertGreaterEqual(len(cases), 20)
        groups = {case["group"] for case in cases}
        for required in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "N"):
            self.assertIn(required, groups, required)

    def test_every_case_declares_an_expected_plan(self) -> None:
        if not EVAL_SET.exists():
            self.skipTest("evaluation set not present")
        required = {
            "question",
            "expected_plan_type",
            "expected_logical_count",
            "expected_terminated_count",
            "expected_unresolved_count",
            "expected_lifecycle_answer",
            "expected_complete",
        }
        for case in json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]:
            self.assertTrue(case.get("question"))
            self.assertTrue(case.get("expected_plan_type"))
            self.assertTrue(case.get("case_id"))
            self.assertFalse(required - set(case), case["case_id"])


if __name__ == "__main__":
    unittest.main()
