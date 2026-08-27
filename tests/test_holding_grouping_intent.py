"""P1-A2: the grouping EvidenceBuilder performs must match the chosen resolver.

TaskRouter promotes a ``disclosure_lookup`` plan on the holding route to a
``holding_event`` execution and picks ``holding_event_resolver``.  That resolver
reads only ``holding_event`` groups, but EvidenceBuilder grouped from
``plan.task_type`` and so produced document/standalone groups instead, leaving
the resolver with nothing to consume.  These tests pin the repair: the router's
execution shape now selects the grouping, in its own vocabulary, without the
plan being rewritten.
"""

import unittest

from app.agent.orchestrator import _EXECUTION_GROUPING_INTENT, AgentOrchestrator
from app.agent.task_router import route_task
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.evidence_builder import EvidenceBuilder, build_evidence_set
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult
from tests.test_agent_end_to_end_smoke import _execution


REPORTER = "국민연금기금"


def _projection(chunk_id: str, doc_id: str, *, reporter: str = REPORTER,
                reference_date: str = "2022년 12월 05일", before: str = "613,758",
                change: str = "106,281", after: str = "720,039", rank: int = 1):
    """A holding_detail_row projection shaped like the frozen corpus."""

    fields = {
        "보고자/보유자": reporter,
        "기준일/보고일": reference_date,
        "직전 보유주식수": before,
        "증감주식수": change,
        "보유주식수": after,
        "보유 목적": "단순투자",
    }
    ref = {"table_id": "t0019", "row_start": 2, "row_end": 2}
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "corp_code": "00970453",
        "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서(약식)",
        "rcept_dt": "2023-01-03",
        "chunk_type": "table_projection",
        "projection_type": "holding_detail_row",
        "projection_fields": fields,
        "projection_field_refs": {label: [dict(ref)] for label in fields},
        "source_refs": [dict(ref)],
        "section_path": ["제3부 직전보고일 이후 대량변동 내역", "2. 세부변동내역"],
        "content": " ".join(f"[{k}] {v}" for k, v in fields.items()),
        "retrieval_text": " ".join(f"[{k}] {v}" for k, v in fields.items()),
    }
    candidate = CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch())
    result = RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict())
    return candidate, result


def _table(chunk_id: str, doc_id: str, *, rank: int = 1):
    """A raw holding table chunk: no structured holding fields at all."""

    text = ("| 성명(명칭) | 변동일* | 변동 내역 / 변동전 | 변동 내역 / 증감 | "
            "변동 내역 / 변동후 |")
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "corp_code": "00970453",
        "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서(약식)",
        "rcept_dt": "2023-01-03",
        "chunk_type": "table",
        "projection_type": None,
        "source_refs": [{"table_id": "t0019", "row_start": rank, "row_end": rank}],
        "section_path": ["제3부 직전보고일 이후 대량변동 내역", "2. 세부변동내역"],
        "content": text,
        "retrieval_text": text,
    }
    candidate = CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch())
    result = RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict())
    return candidate, result


def _plan(task_type: str, *, route=("holding",), metric=None,
          raw_query: str = "테스트회사 국민연금기금 변동일 변동전 변동후") -> QueryPlan:
    return QueryPlan(
        query="국민연금기금 변동일 변동전 변동후",
        raw_query=raw_query,
        company="테스트회사",
        task_type=task_type,
        metric=metric,
        reporter=REPORTER,
        disclosure_route=route,
        route_confidence={f"disclosure_route.{route[0]}": 0.99} if route else {},
    )


def _groups(evidence, group_type: str):
    return [g for g in evidence.evidence_groups if g.group_type == group_type]


class MappingTests(unittest.TestCase):
    """B: the vocabulary conversion is explicit and testable."""

    def test_the_router_shape_maps_onto_the_grouping_name(self) -> None:
        self.assertEqual(_EXECUTION_GROUPING_INTENT["holding_event"], "holding_change")

    def test_only_the_holding_execution_shape_is_mapped(self) -> None:
        # Every other route keeps the plan's own task type as the grouping.
        self.assertEqual(set(_EXECUTION_GROUPING_INTENT), {"holding_event"})

    def test_passing_the_router_vocabulary_verbatim_would_not_group(self) -> None:
        """Guards the exact trap: "holding_event" is not the grouping name."""

        candidate, result = _projection("c1", "d1")
        verbatim = build_evidence_set(
            question="q", query_plan=_plan("disclosure_lookup").to_dict(),
            candidates=[candidate], results=[result],
            grouping_intent="holding_event",
        )
        mapped = build_evidence_set(
            question="q", query_plan=_plan("disclosure_lookup").to_dict(),
            candidates=[candidate], results=[result],
            grouping_intent=_EXECUTION_GROUPING_INTENT["holding_event"],
        )
        self.assertEqual(_groups(verbatim, "holding_event"), [])
        self.assertTrue(_groups(mapped, "holding_event"))


class GroupingIntentTests(unittest.TestCase):
    """A: a promoted disclosure_lookup now gets holding groups."""

    def setUp(self) -> None:
        self.candidate, self.result = _projection("c1", "d1")
        self.plan = _plan("disclosure_lookup")

    def test_the_router_promotes_this_plan(self) -> None:
        decision = route_task(self.plan.raw_query, self.plan)
        self.assertEqual(self.plan.task_type, "disclosure_lookup")
        self.assertEqual(decision.task_type, "holding_event")
        self.assertEqual(decision.resolver_type, "holding_event_resolver")

    def test_without_the_intent_no_holding_group_is_built(self) -> None:
        evidence = build_evidence_set(
            question="q", query_plan=self.plan.to_dict(),
            candidates=[self.candidate], results=[self.result])
        self.assertEqual(_groups(evidence, "holding_event"), [])

    def test_with_the_intent_a_holding_group_is_built(self) -> None:
        evidence = build_evidence_set(
            question="q", query_plan=self.plan.to_dict(),
            candidates=[self.candidate], results=[self.result],
            grouping_intent="holding_change")
        self.assertEqual(len(_groups(evidence, "holding_event")), 1)

    def test_the_plan_semantics_are_not_rewritten(self) -> None:
        """The EvidenceSet still reports what query understanding decided."""

        evidence = build_evidence_set(
            question="q", query_plan=self.plan.to_dict(),
            candidates=[self.candidate], results=[self.result],
            grouping_intent="holding_change")
        self.assertEqual(evidence.task_type, "disclosure_lookup")
        self.assertEqual(self.plan.task_type, "disclosure_lookup")

    def test_the_orchestrator_wires_the_intent_end_to_end(self) -> None:
        execution = _execution(self.plan, (self.candidate, self.result))
        result = AgentOrchestrator().run(self.plan.raw_query, self.plan, execution)

        self.assertEqual(result.task_decision.task_type, "holding_event")
        self.assertEqual(len(_groups(result.evidence_set, "holding_event")), 1)
        self.assertEqual(result.resolution.matching_event_count, 1)


class BackwardCompatibilityTests(unittest.TestCase):
    """C + D: omitting the intent reproduces the previous behaviour exactly."""

    CASES = {
        "holding_change": _plan("holding_change", metric="holding_shares"),
        "disclosure_lookup": _plan("disclosure_lookup"),
        "financial_metric": _plan("financial_metric", route=("periodic",),
                                  metric="매출액",
                                  raw_query="테스트회사 2024년 매출액"),
        "corporate_event": _plan("corporate_event", route=("major",),
                                 raw_query="테스트회사 유상증자 결정"),
    }

    def test_omitting_the_intent_equals_passing_the_plan_task_type(self) -> None:
        candidate, result = _projection("c1", "d1")
        for name, plan in self.CASES.items():
            with self.subTest(plan=name):
                default = build_evidence_set(
                    question="q", query_plan=plan.to_dict(),
                    candidates=[candidate], results=[result])
                explicit = build_evidence_set(
                    question="q", query_plan=plan.to_dict(),
                    candidates=[candidate], results=[result],
                    grouping_intent=plan.task_type)
                self.assertEqual(default.to_dict(), explicit.to_dict())

    def test_an_already_working_holding_change_plan_is_unchanged(self) -> None:
        candidate, result = _projection("c1", "d1")
        plan = self.CASES["holding_change"]
        before = build_evidence_set(
            question="q", query_plan=plan.to_dict(),
            candidates=[candidate], results=[result])
        after = build_evidence_set(
            question="q", query_plan=plan.to_dict(),
            candidates=[candidate], results=[result],
            grouping_intent=_EXECUTION_GROUPING_INTENT["holding_event"])
        self.assertEqual(before.to_dict(), after.to_dict())
        self.assertEqual(len(_groups(after, "holding_event")), 1)

    def test_the_builder_object_also_defaults_to_the_plan(self) -> None:
        candidate, result = _projection("c1", "d1")
        plan = self.CASES["disclosure_lookup"]
        execution = _execution(plan, (candidate, result))
        self.assertEqual(
            EvidenceBuilder().build(execution, question="q").to_dict(),
            build_evidence_set(question="q", query_plan=plan.to_dict(),
                               candidates=[candidate], results=[result]).to_dict(),
        )


class UnrelatedRouteIdentityTests(unittest.TestCase):
    """E, F, G, H, I: nothing outside the holding execution shape moves."""

    ROUTES = {
        "financial_metric": ("테스트회사 2024년 매출액", "financial_metric",
                             ("periodic",), "매출액"),
        "periodic_disclosure_lookup": ("테스트회사 2024년 사업보고서 주요 제품",
                                       "disclosure_lookup", ("periodic",), None),
        "corporate_event": ("테스트회사 유상증자 결정 내용", "corporate_event",
                            ("major",), None),
        "correction_lookup": ("테스트회사 정정 공시만 계약금액", "disclosure_lookup",
                              ("exchange",), None),
        "general_lookup": ("테스트회사 공시 내용 알려줘", "disclosure_lookup",
                           ("exchange",), None),
    }

    def test_no_unrelated_route_is_promoted_to_holding_grouping(self) -> None:
        for name, (raw, task, route, metric) in self.ROUTES.items():
            with self.subTest(route=name):
                plan = _plan(task, route=route, metric=metric, raw_query=raw)
                decision = route_task(raw, plan)
                self.assertNotEqual(decision.task_type, "holding_event")
                self.assertIsNone(
                    _EXECUTION_GROUPING_INTENT.get(decision.task_type))

    def test_unrelated_routes_build_identical_evidence(self) -> None:
        candidate, result = _table("c1", "d1")
        for name, (raw, task, route, metric) in self.ROUTES.items():
            with self.subTest(route=name):
                plan = _plan(task, route=route, metric=metric, raw_query=raw)
                decision = route_task(raw, plan)
                default = build_evidence_set(
                    question=raw, query_plan=plan.to_dict(),
                    candidates=[candidate], results=[result])
                routed = build_evidence_set(
                    question=raw, query_plan=plan.to_dict(),
                    candidates=[candidate], results=[result],
                    grouping_intent=_EXECUTION_GROUPING_INTENT.get(
                        decision.task_type))
                self.assertEqual(default.to_dict(), routed.to_dict())

    def test_a_disclosure_lookup_off_the_holding_route_stays_document_evidence(self) -> None:
        """I: the holding route, not the plan task, is what the router keys on."""

        raw = "테스트회사 공시 내용 알려줘"
        plan = _plan("disclosure_lookup", route=("exchange",), raw_query=raw)
        decision = route_task(raw, plan)
        self.assertEqual(decision.task_type, "general_evidence")

        candidate, result = _table("c1", "d1")
        evidence = build_evidence_set(
            question=raw, query_plan=plan.to_dict(),
            candidates=[candidate], results=[result],
            grouping_intent=_EXECUTION_GROUPING_INTENT.get(decision.task_type))
        self.assertEqual(_groups(evidence, "holding_event"), [])
        self.assertTrue(
            _groups(evidence, "document_evidence")
            or _groups(evidence, "standalone_evidence"))


class RetrievalStillOutOfScopeTests(unittest.TestCase):
    """J + K: the repair fixes grouping, and only grouping."""

    def _answerability(self, plan, pairs):
        execution = _execution(plan, *pairs)
        result = AgentOrchestrator().run(plan.raw_query, plan, execution)
        generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
        answerability = AnswerabilityGuard().evaluate(
            generated, plan=plan, agent_result=result, execution=execution)
        return result, answerability

    def test_without_a_structured_projection_it_stays_unanswerable(self) -> None:
        """J: raw tables carry no holding fields, so no event can be seeded.

        This is the live HX12 shape and it is expected to stay unanswerable.
        The grouping repair must not be mistaken for a retrieval fix.
        """

        plan = _plan("disclosure_lookup")
        pairs = [_table(f"t{i}", "d1", rank=i + 1) for i in range(3)]
        result, answerability = self._answerability(plan, pairs)

        self.assertEqual(result.task_decision.task_type, "holding_event")
        self.assertEqual(_groups(result.evidence_set, "holding_event"), [])
        self.assertEqual(result.resolution.matching_event_count, 0)
        self.assertFalse(answerability.model_answer_allowed)
        self.assertEqual(answerability.citation_count, 0)

    def test_with_a_structured_projection_the_resolver_consumes_it(self) -> None:
        """K: the injected control -- grouping repair proven correct."""

        plan = _plan("disclosure_lookup")
        pairs = [_projection("p1", "d1", rank=1)]
        pairs += [_table(f"t{i}", "d1", rank=i + 2) for i in range(3)]
        result, answerability = self._answerability(plan, pairs)

        self.assertEqual(len(_groups(result.evidence_set, "holding_event")), 1)
        self.assertEqual(result.resolution.matching_event_count, 1)
        event = result.resolution.events[0]
        self.assertEqual(event.reporter, REPORTER)
        self.assertEqual(event.reference_date, "2022-12-05")
        self.assertEqual(event.before_shares.normalized, 613758)
        self.assertEqual(event.after_shares.normalized, 720039)
        self.assertTrue(event.source_refs)
        self.assertTrue(answerability.model_answer_allowed)
        self.assertGreater(answerability.citation_count, 0)
        self.assertEqual(list(answerability.missing_fields), [])


if __name__ == "__main__":
    unittest.main()
