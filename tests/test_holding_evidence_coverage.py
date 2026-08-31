"""P1-A3: served holding evidence must be able to answer what was asked.

Sufficiency is not "is a projection present" -- that was the mistake of the
reverted rescue, which let an unrelated event's projection suppress a top-up.
It is whether the union of the served, citable, reporter-compatible projections
covers every field the question asked for.  These tests pin that rule, the
minimal selection that closes a gap, and the cases where the lane must decline.
"""

import unittest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.holding_evidence_coverage import (
    ACQUISITION_PROOF,
    ANCHOR_MEDIUM,
    ANCHOR_STRONG,
    PROVENANCE_KEY,
    RESCUE_MODE_ACQUISITION_PROOF,
    STATUS_NO_ACQUISITION_PROOF,
    STATUS_NO_ANCHOR,
    anchor_tier,
    STATUS_COVERED,
    STATUS_NO_CANDIDATE,
    STATUS_NOT_HOLDING,
    STATUS_NO_SAFE_DISPLACEMENT,
    STATUS_RESCUED,
    _assess_field_coverage,
    _minimum_subset,
    assess,
    proves_acquisition,
    reporter_compatible,
    strict_reporter_identity,
)
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult
from tests.test_agent_end_to_end_smoke import _execution


FUND = "국민연금기금"          # a specific member of a holder family
AGENCY = "국민연금공단"        # a different member of the same family
FAMILY = "국민연금"            # the family name


def _projection(
    chunk_id,
    doc_id="d1",
    *,
    kind="holding_detail_row",
    reporter=FUND,
    reference_date="2022년 12월 05일",
    before="613,758",
    change="106,281",
    after="720,039",
    before_ratio=None,
    after_ratio=None,
    citable=True,
    rank=1,
    event_row=2,
):
    """A structured holding projection shaped like the frozen corpus.

    A detail row carries shares and the reference date; a report additionally
    carries the ratios. That asymmetry is the second failure mode this lane
    exists for.
    """

    fields = {"보고자/보유자": reporter, "기준일/보고일": reference_date}
    if before is not None:
        fields["직전 보유주식수"] = before
    if change is not None:
        fields["증감주식수"] = change
    if after is not None:
        fields["보유주식수"] = after
    if before_ratio is not None:
        fields["직전 보유비율"] = before_ratio
    if after_ratio is not None:
        fields["보유비율"] = after_ratio
    ref = {"table_id": "t0019", "row_start": event_row, "row_end": event_row}
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "corp_code": "00970453",
        "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서(약식)",
        "rcept_dt": "2023-01-03",
        "chunk_type": "table_projection",
        "projection_type": kind,
        "projection_fields": fields,
        "projection_field_refs": (
            {label: [dict(ref)] for label in fields} if citable else {}
        ),
        "source_refs": [dict(ref)] if citable else [],
        "section_path": ["제3부 직전보고일 이후 대량변동 내역", "2. 세부변동내역"],
        "content": " ".join(f"[{k}] {v}" for k, v in fields.items()),
        "retrieval_text": " ".join(f"[{k}] {v}" for k, v in fields.items()),
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()))


def _table(chunk_id, doc_id="d1", *, rank=1, doc_group="holding", event_row=2):
    """A raw table chunk: the header vocabulary, none of the structured fields.

    ``event_row`` is the table row it covers, which is what a projection of the
    same filing shares when the two describe one event.
    """

    text = "| 성명(명칭) | 변동일* | 변동 내역 / 변동전 | 변동 내역 / 변동후 |"
    chunk = {
        "chunk_id": chunk_id, "doc_id": doc_id, "doc_group": doc_group,
        "corp_code": "00970453", "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서(약식)", "rcept_dt": "2023-01-03",
        "chunk_type": "table", "projection_type": None,
        "source_refs": [
            {"table_id": "t0019", "row_start": event_row, "row_end": event_row}
        ],
        "section_path": ["제3부 직전보고일 이후 대량변동 내역", "2. 세부변동내역"],
        "content": text, "retrieval_text": text,
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()))


def _served_tables(count, *, first_doc="d0"):
    """One raw table per disclosure, as the corpus actually serves them."""

    return [_table(f"t{i}", f"d{i}" if i else first_doc, rank=i + 1)
            for i in range(count)]


def _plan(raw_query, *, task_type="disclosure_lookup", reporter=FUND,
          metric=None, route=("holding",)):
    return QueryPlan(
        query=raw_query, raw_query=raw_query, company="테스트회사",
        task_type=task_type, metric=metric, reporter=reporter,
        disclosure_route=route,
        route_confidence={f"disclosure_route.{route[0]}": 0.99} if route else {},
    )


def _assess(plan, pool, served, routed="holding_event"):
    return assess(plan.raw_query, plan, [c for c, _ in pool],
                  [r for _, r in served], routed_task_type=routed)


HX12_Q = "테스트회사 국민연금기금 변동일 변동전 변동후"          # -> reference_date
HX10_Q = "테스트회사 국민연금 직전보고 수량 지분율"               # -> before/after ratio
HX08_Q = "테스트회사 국민연금기금 변동일 변동후 주식수"           # -> reference_date+after_shares
HX16_Q = "테스트회사 국민연금기금 변동일 감소 후 주식수"          # -> +change_direction


class RequestShapeTests(unittest.TestCase):
    """The requested fields come from the resolver, not from a second list."""

    def test_the_fixture_questions_request_what_diagnosis_measured(self) -> None:
        cases = {
            HX12_Q: {"reference_date"},
            HX10_Q: {"before_ratio", "after_ratio"},
            HX08_Q: {"reference_date", "after_shares"},
            HX16_Q: {"reference_date", "after_shares", "change_direction"},
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                a = _assess(_plan(question, reporter=FAMILY), [], [])
                self.assertEqual(set(a.requested), expected)


class GapClosingTests(unittest.TestCase):
    """A + B: the two real shapes diagnosis found."""

    def test_no_structured_evidence_served_is_topped_up(self) -> None:
        """A: HX12 shape -- ten raw tables, one compatible detail row in pool."""

        tables = _served_tables(10)
        wanted = _projection("p_gap", "d0")          # sibling of served rank 1
        pool = [*tables, wanted]
        plan = _plan(HX12_Q)
        a = _assess(plan, pool, tables)

        self.assertEqual(a.status, STATUS_RESCUED)
        self.assertEqual(list(a.unresolved), ["reference_date"])
        self.assertEqual([r.chunk_id for r in a.selected], ["p_gap"])
        self.assertEqual(len(a.selected), 1)
        self.assertEqual(len(a.results), len(tables))      # count preserved
        self.assertEqual(list(a.remaining_unresolved), [])
        ids = [r.chunk_id for r in a.results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("p_gap", ids)

    def test_wrong_projection_type_for_the_request_is_topped_up(self) -> None:
        """B: HX10 shape -- detail rows served, but ratios were asked for."""

        served = [_projection(f"row{i}", f"d{i}", rank=i + 1, reporter=AGENCY)
                  for i in range(10)]
        report = _projection("rep", "d9", kind="holding_report", reporter=AGENCY,
                             before_ratio="6.07", after_ratio="7.12")
        plan = _plan(HX10_Q, reporter=FAMILY)
        a = _assess(plan, [*served, report], served)

        self.assertEqual(a.status, STATUS_RESCUED)
        self.assertEqual(sorted(a.unresolved), ["after_ratio", "before_ratio"])
        self.assertEqual([r.chunk_id for r in a.selected], ["rep"])
        self.assertEqual(list(a.remaining_unresolved), [])

    def test_only_the_missing_half_is_topped_up(self) -> None:
        """C: partial union coverage -- one field served, one field added."""

        served = [_projection("row", "d1", after_ratio=None, rank=1)]
        report = _projection("rep", "d1", kind="holding_report",
                             before_ratio="6.07", after_ratio="7.12")
        filler = [_table(f"t{i}", f"f{i}", rank=i + 2) for i in range(3)]
        plan = _plan("테스트회사 국민연금기금 변동후 주식수와 보유 비율")
        a = _assess(plan, [*served, *filler, report], [*served, *filler])

        self.assertIn("after_shares", a.served_coverage)
        self.assertIn("after_ratio", a.unresolved)
        self.assertEqual([r.chunk_id for r in a.selected], ["rep"])


class DeclineTests(unittest.TestCase):
    """D + E + F: the lane must not fire when the request is already answerable."""

    def test_a_complete_union_declines(self) -> None:
        """D: two served projections jointly cover the request."""

        shares = _projection("row", "d1", rank=1)
        ratios = _projection("rep", "d2", kind="holding_report", rank=2,
                             before_ratio="6.07", after_ratio="7.12")
        spare = _projection("spare", "d3", kind="holding_report", rank=9,
                            before_ratio="1.0", after_ratio="2.0")
        plan = _plan("테스트회사 국민연금기금 변동후 주식수와 보유 비율")
        a = _assess(plan, [shares, ratios, spare], [shares, ratios])

        self.assertEqual(a.status, STATUS_COVERED)
        self.assertFalse(a.rescued)
        self.assertEqual(list(a.unresolved), [])

    def test_it_declines_even_when_another_event_is_absent(self) -> None:
        """E: HX08 shape -- the served event answers, a different one is not fetched."""

        served = [_projection("served", "d1", rank=1)]
        absent = _projection("absent", "d1", reference_date="2023년 03월 07일",
                             after="655,490", rank=9)
        a = _assess(_plan(HX08_Q), [*served, absent], served)

        self.assertEqual(a.status, STATUS_COVERED)
        self.assertFalse(a.rescued)
        self.assertNotIn("absent", [r.chunk_id for r in a.results])

    def test_derived_change_direction_counts_as_covered(self) -> None:
        """F: HX16/HX20 shape -- direction comes from the share change's sign."""

        served = [_projection("served", "d1", change="-283,151", rank=1)]
        a = _assess(_plan(HX16_Q), served, served)

        self.assertIn("change_direction", a.requested)
        self.assertIn("change_direction", a.served_coverage)
        self.assertEqual(a.status, STATUS_COVERED)

    def test_a_projection_without_a_share_change_leaves_direction_open(self) -> None:
        served = [_projection("served", "d1", change=None, rank=1)]
        a = _assess(_plan(HX16_Q), served, served)
        self.assertIn("change_direction", a.unresolved)


class ReporterTests(unittest.TestCase):
    """G + H + I: containment, generically."""

    def test_a_family_name_matches_a_member(self) -> None:
        row = _projection("p", "d1", reporter=FUND)
        self.assertTrue(reporter_compatible(row[0].chunk, FAMILY))

    def test_a_specific_member_does_not_match_a_different_member(self) -> None:
        row = _projection("p", "d1", reporter=AGENCY)
        self.assertFalse(reporter_compatible(row[0].chunk, FUND))

    def test_containment_holds_for_an_unrelated_holder(self) -> None:
        """I: parameterised away from the domain, proving nothing is hardcoded."""

        cases = [
            ("한국투자신탁운용", "한국투자신탁운용 제1호", True),
            ("한국투자신탁운용 제1호", "한국투자신탁운용", True),
            ("한국투자신탁운용", "미래에셋자산운용", False),
            ("BlackRock Fund Advisors", "BlackRock", True),
            ("BlackRock", "Vanguard Group", False),
        ]
        for holder, asked, expected in cases:
            with self.subTest(holder=holder, asked=asked):
                row = _projection("p", "d1", reporter=holder)
                self.assertIs(reporter_compatible(row[0].chunk, asked), expected)

    def test_an_incompatible_holder_is_never_promoted(self) -> None:
        tables = _served_tables(3)
        foreign = _projection("foreign", "d0", reporter=AGENCY)
        a = _assess(_plan(HX12_Q, reporter=FUND), [*tables, foreign], tables)

        self.assertEqual(a.status, STATUS_NO_CANDIDATE)
        self.assertFalse(a.rescued)

    def test_without_a_parsed_reporter_nothing_is_invented(self) -> None:
        tables = _served_tables(3)
        row = _projection("p", "d0", reporter="아무개자산운용")
        a = _assess(_plan(HX12_Q, reporter=None), [*tables, row], tables)
        self.assertEqual([r.chunk_id for r in a.selected], ["p"])


class SafetyTests(unittest.TestCase):
    """J + N + O + P + Q: what the lane must never do."""

    def test_an_uncitable_projection_is_never_promoted(self) -> None:
        tables = _served_tables(3)
        row = _projection("p", "d0", citable=False)
        a = _assess(_plan(HX12_Q), [*tables, row], tables)
        self.assertEqual(a.status, STATUS_NO_CANDIDATE)

    def test_an_already_served_candidate_is_never_duplicated(self) -> None:
        row = _projection("p", "d1", rank=1)
        a = _assess(_plan(HX12_Q), [row], [row])
        ids = [r.chunk_id for r in a.results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(a.status, STATUS_COVERED)

    def test_the_served_count_is_preserved(self) -> None:
        for size in (1, 3, 10):
            with self.subTest(size=size):
                tables = _served_tables(size)
                row = _projection("p", "d0")
                a = _assess(_plan(HX12_Q), [*tables, row], tables)
                self.assertEqual(len(a.results), size)

    def test_a_contributing_row_is_never_displaced(self) -> None:
        """P: make room from a non-contributor, never from a contributor."""

        contributor = _projection("keep", "d1", after_ratio=None, rank=1)
        filler = _table("drop", "d2", rank=2)
        report = _projection("rep", "d1", kind="holding_report",
                             before_ratio="6.07", after_ratio="7.12")
        plan = _plan("테스트회사 국민연금기금 변동후 주식수와 보유 비율")
        a = _assess(plan, [contributor, filler, report], [contributor, filler])

        self.assertEqual(list(a.displaced), ["drop"])
        self.assertIn("keep", [r.chunk_id for r in a.results])

    def test_it_declines_when_every_served_row_is_a_contributor(self) -> None:
        """P/Q: no safe displacement -> decline rather than damage evidence."""

        served = [_projection("c1", "d1", after_ratio=None, rank=1)]
        report = _projection("rep", "d1", kind="holding_report",
                             before_ratio="6.07", after_ratio="7.12")
        plan = _plan("테스트회사 국민연금기금 변동후 주식수와 보유 비율")
        a = _assess(plan, [*served, report], served)

        self.assertEqual(a.status, STATUS_NO_SAFE_DISPLACEMENT)
        self.assertFalse(a.rescued)
        self.assertEqual([r.chunk_id for r in a.results],
                         [r.chunk_id for _c, r in served])

    def test_an_unclosable_gap_is_reported_not_faked(self) -> None:
        """Q: nothing in the pool supplies the field."""

        tables = [_table(f"t{i}", rank=i + 1) for i in range(3)]
        a = _assess(_plan(HX10_Q, reporter=FAMILY), tables, tables)

        self.assertEqual(a.status, STATUS_NO_CANDIDATE)
        self.assertEqual(sorted(a.unresolved), ["after_ratio", "before_ratio"])
        self.assertFalse(a.rescued)

    def test_selected_rows_are_marked(self) -> None:
        tables = _served_tables(3)
        row = _projection("p", "d0")
        a = _assess(_plan(HX12_Q), [*tables, row], tables)
        promoted = next(r for r in a.results if r.chunk_id == "p")
        self.assertIn(PROVENANCE_KEY, promoted.metadata_match)
        for result in a.results:
            if result.chunk_id != "p":
                self.assertNotIn(PROVENANCE_KEY, result.metadata_match)


class NonHoldingTests(unittest.TestCase):
    """K: any other routed execution is untouched."""

    ROUTES = ("periodic_fact", "corporate_event", "general_evidence", "unknown", None)

    def test_no_other_execution_shape_is_assessed(self) -> None:
        tables = _served_tables(3)
        row = _projection("p", "d0")
        for routed in self.ROUTES:
            with self.subTest(routed=routed):
                a = _assess(_plan(HX12_Q), [*tables, row], tables, routed=routed)
                self.assertEqual(a.status, STATUS_NOT_HOLDING)
                self.assertFalse(a.evaluated)
                self.assertFalse(a.rescued)
                self.assertEqual([r.chunk_id for r in a.results],
                                 [r.chunk_id for _c, r in tables])


class _RecordingExecution:
    """M: proves the lane fetches nothing -- any access would be recorded."""

    def __init__(self, plan, pairs) -> None:
        self.plan = plan
        self._chunks = [c for c, _ in pairs]
        self._results = [r for _, r in pairs]
        self.calls: list[str] = []

    @property
    def chunks(self):
        self.calls.append("chunks")
        return self._chunks

    @property
    def results(self):
        self.calls.append("results")
        return self._results

    def retrieve(self, *a, **k):            # pragma: no cover - must never run
        raise AssertionError("the coverage lane must not search")

    def vector_search(self, *a, **k):       # pragma: no cover
        raise AssertionError("the coverage lane must not search")

    def get_candidate_chunks(self, *a, **k):  # pragma: no cover
        raise AssertionError("the coverage lane must not fetch")

    def get_candidate_documents(self, *a, **k):  # pragma: no cover
        raise AssertionError("the coverage lane must not fetch")


class NoSecondSearchTests(unittest.TestCase):
    def test_the_lane_reads_the_pool_and_fetches_nothing(self) -> None:
        tables = _served_tables(3)
        row = _projection("p", "d0")
        plan = _plan(HX12_Q)
        execution = _RecordingExecution(plan, [*tables, row])
        a = assess(plan.raw_query, plan, execution.chunks,
                   [r for _c, r in tables], routed_task_type="holding_event")

        self.assertEqual(a.status, STATUS_RESCUED)
        self.assertEqual(set(execution.calls), {"chunks"})


class OrchestratorIntegrationTests(unittest.TestCase):
    """End to end through the real orchestrator, including P1-A2 (test R)."""

    def _run(self, plan, pairs, served=None):
        """``pairs`` is the candidate pool; ``served`` is what retrieval ranked.

        They differ in the shape this lane exists for: the projection sits in
        the pool but never reached the served list.
        """

        from types import SimpleNamespace

        served = pairs if served is None else served
        execution = SimpleNamespace(
            plan=plan,
            chunks=[c for c, _ in pairs],
            results=[r for _, r in served],
        )
        result = AgentOrchestrator().run(plan.raw_query, plan, execution)
        generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
        answerability = AnswerabilityGuard().evaluate(
            generated, plan=plan, agent_result=result, execution=execution)
        return execution, result, answerability

    def test_a_gap_is_closed_end_to_end(self) -> None:
        tables = _served_tables(3)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        execution, result, answerability = self._run(
            plan, [*tables, wanted], served=tables)

        self.assertEqual(result.task_decision.task_type, "holding_event")
        self.assertTrue(result.holding_coverage.rescued)
        self.assertEqual(result.holding_coverage.status, STATUS_RESCUED)
        self.assertIn("holding_evidence_coverage", result.execution_trace)
        # P1-A2 still supplies the grouping the resolver consumes.
        self.assertTrue([g for g in result.evidence_set.evidence_groups
                         if g.group_type == "holding_event"])
        self.assertEqual(result.resolution.matching_event_count, 1)
        event = result.resolution.events[0]
        self.assertEqual(event.reference_date, "2022-12-05")
        self.assertTrue(event.source_refs)
        self.assertTrue(answerability.model_answer_allowed)
        self.assertGreater(answerability.citation_count, 0)
        self.assertEqual(list(answerability.missing_fields), [])

    def test_the_original_execution_is_never_mutated(self) -> None:
        tables = _served_tables(3)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        execution, result, _a = self._run(plan, [*tables, wanted], served=tables)

        # run() enforces read-only invariants internally; assert it explicitly:
        # the served list handed in is unchanged, the top-up went to a copy.
        self.assertEqual([r.chunk_id for r in execution.results],
                         [r.chunk_id for _c, r in tables])
        self.assertTrue(result.holding_coverage.rescued)
        self.assertIn("p_gap",
                      [r.chunk_id for r in result.holding_coverage.results])

    def test_an_already_covered_request_is_left_alone(self) -> None:
        served = [_projection("served", "d1", rank=1)]
        absent = _projection("absent", "d1", reference_date="2023년 03월 07일", rank=9)
        plan = _plan(HX08_Q)
        _e, result, answerability = self._run(
            plan, [*served, absent], served=served)

        self.assertFalse(result.holding_coverage.rescued)
        self.assertEqual(result.holding_coverage.status, STATUS_COVERED)
        self.assertNotIn("holding_evidence_coverage", result.execution_trace)
        self.assertTrue(answerability.model_answer_allowed)

    def test_a_non_holding_execution_is_byte_identical(self) -> None:
        """K, end to end."""

        pairs = [_table(f"t{i}", f"d{i}", rank=i + 1, doc_group="periodic")
                 for i in range(3)]
        plan = _plan("테스트회사 2024년 매출액", task_type="financial_metric",
                     reporter=None, metric="매출액", route=("periodic",))
        _e, result, _a = self._run(plan, pairs)

        self.assertNotEqual(result.task_decision.task_type, "holding_event")
        self.assertFalse(result.holding_coverage.evaluated)
        self.assertNotIn("holding_evidence_coverage", result.execution_trace)

    def test_p1a2_grouping_intent_is_unchanged(self) -> None:
        """R: the frozen P1-A2 mapping still drives grouping."""

        from app.agent.orchestrator import _EXECUTION_GROUPING_INTENT

        self.assertEqual(_EXECUTION_GROUPING_INTENT, {"holding_event": "holding_change"})
        served = [_projection("served", "d1", rank=1)]
        plan = _plan(HX08_Q)                      # disclosure_lookup + holding route
        _e, result, _a = self._run(plan, served)

        self.assertEqual(plan.task_type, "disclosure_lookup")
        self.assertEqual(result.evidence_set.task_type, "disclosure_lookup")
        self.assertTrue([g for g in result.evidence_set.evidence_groups
                         if g.group_type == "holding_event"])


class BlockedRequestTests(unittest.TestCase):
    """L: a request P0-D refuses never reaches this lane."""

    def test_a_blocked_query_never_reaches_the_orchestrator(self) -> None:
        from app.api.pipeline import AnswerPipeline
        from tests.test_p0d_pipeline import _Understanding, _validator

        class _RecordingExecutor:
            def __init__(self) -> None:
                self.calls = 0

            def execute(self, plan):
                del plan
                self.calls += 1
                raise AssertionError("the coverage lane must be unreachable")

        blocked = QueryPlan(
            query="변동일", raw_query="애플 국민연금기금 변동일 변동전 변동후",
            company="애플", task_type="disclosure_lookup", reporter=FUND,
            disclosure_route=("holding",), evidence={"operation": "lookup_holding"},
        )
        executor = _RecordingExecutor()
        pipeline = AnswerPipeline(
            understanding=_Understanding(blocked), executor=executor,
            query_validator=_validator(), answerability_guard=AnswerabilityGuard())
        payload = pipeline.answer("P1A3-blocked", blocked.raw_query)

        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertFalse(payload["think_trace"]["query_validation"]["retrieval_allowed"])


class PublicContractTests(unittest.TestCase):
    """Diagnostics stay internal; the response contract does not move."""

    def test_the_response_keeps_exactly_its_five_top_level_fields(self) -> None:
        from tests.test_answer_api import _ask, _client

        payload = _ask(_client()).json()
        self.assertEqual(
            set(payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )
        self.assertNotIn("holding_coverage", payload["think_trace"])


def _field_projection(chunk_id, doc_id, *, fields, reporter=FUND,
                      reference_date="2022년 12월 05일", kind="holding_detail_row",
                      event_row=2, rank=1):
    """A projection carrying exactly the named canonical fields.

    ``fields`` maps canonical field name -> value, so a fixture can say "this
    one supplies before_shares only" without restating the label table.
    """

    LABEL = {
        "reference_date": "기준일/보고일", "before_shares": "직전 보유주식수",
        "change_shares": "증감주식수", "after_shares": "보유주식수",
        "before_ratio": "직전 보유비율", "after_ratio": "보유비율",
        "change_ratio": "증감비율",
    }
    pf = {"보고자/보유자": reporter, "기준일/보고일": reference_date}
    for name, value in fields.items():
        pf[LABEL[name]] = value
    ref = {"table_id": "t0019", "row_start": event_row, "row_end": event_row}
    meta = {"table_id": "t0002", "row_start": 8, "row_end": 8}
    chunk = {
        "chunk_id": chunk_id, "doc_id": doc_id, "doc_group": "holding",
        "corp_code": "00970453", "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서(약식)", "rcept_dt": "2023-01-03",
        "chunk_type": "table_projection", "projection_type": kind,
        "projection_fields": pf,
        "projection_field_refs": {
            **{label: [dict(ref)] for label in pf},
            "보유 목적": [dict(meta)],
        },
        "source_refs": [dict(ref)],
        "section_path": ["제3부 직전보고일 이후 대량변동 내역", "2. 세부변동내역"],
        "content": " ".join(f"[{k}] {v}" for k, v in pf.items()),
        "retrieval_text": " ".join(f"[{k}] {v}" for k, v in pf.items()),
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()))


class AnchorContractTests(unittest.TestCase):
    """C, D, E: what does and does not tie a candidate to served evidence."""

    def test_a_shared_event_row_is_a_strong_anchor(self) -> None:
        served = _table("t", "d1", rank=1)
        cand = _projection("p", "d1")
        self.assertEqual(
            anchor_tier(cand[0].chunk, served[0].chunk, FUND), ANCHOR_STRONG)

    def test_a_metadata_only_overlap_is_not_a_strong_anchor(self) -> None:
        """C: every projection of a filing points at the holding-purpose row."""

        meta = {"table_id": "t0002", "row_start": 8, "row_end": 8}
        served = _projection("s", "d1", reference_date="2023년 09월 13일")
        served[0].chunk["projection_field_refs"] = {"보유 목적": [dict(meta)]}
        served[0].chunk["source_refs"] = [dict(meta)]
        cand = _projection("c", "d1", kind="holding_report",
                           reference_date="2024년 01월 01일",
                           before_ratio="6.07", after_ratio="7.12")
        cand[0].chunk["projection_field_refs"]["보유 목적"] = [dict(meta)]
        # Metadata overlap only, and the dates differ: no anchor at all.
        self.assertIsNone(anchor_tier(cand[0].chunk, served[0].chunk, FUND))

    def test_same_document_alone_is_not_an_anchor(self) -> None:
        """D: a disclosure can report several holding events."""

        served = _projection("s", "d1", reference_date="2023년 09월 13일",
                             event_row=2)
        cand = _projection("c", "d1", reference_date="2024년 06월 27일",
                           event_row=5)
        self.assertIsNone(anchor_tier(cand[0].chunk, served[0].chunk, FUND))

    def test_medium_needs_doc_date_and_reporter(self) -> None:
        """E: all three conditions are required."""

        served = _projection("s", "d1", reference_date="2023년 09월 13일",
                             event_row=2)
        same = _projection("c", "d1", kind="holding_report", event_row=7,
                           reference_date="2023년 09월 13일",
                           before_ratio="8.41", after_ratio="9.46")
        self.assertEqual(
            anchor_tier(same[0].chunk, served[0].chunk, FUND), ANCHOR_MEDIUM)

        other_doc = _projection("c", "d2", kind="holding_report", event_row=7,
                                reference_date="2023년 09월 13일",
                                before_ratio="8.41", after_ratio="9.46")
        self.assertIsNone(anchor_tier(other_doc[0].chunk, served[0].chunk, FUND))

        other_date = _projection("c", "d1", kind="holding_report", event_row=7,
                                 reference_date="2022년 12월 05일",
                                 before_ratio="8.41", after_ratio="9.46")
        self.assertIsNone(anchor_tier(other_date[0].chunk, served[0].chunk, FUND))

        other_holder = _projection("c", "d1", kind="holding_report", event_row=7,
                                   reporter="미래에셋자산운용",
                                   reference_date="2023년 09월 13일",
                                   before_ratio="8.41", after_ratio="9.46")
        self.assertIsNone(anchor_tier(other_holder[0].chunk, served[0].chunk, FUND))


class AnchorRankPriorityTests(unittest.TestCase):
    """A, B, G, I, J, K: served rank decides which evidence gets completed."""

    def test_pool_order_does_not_beat_served_rank(self) -> None:
        """A: the rank-2 sibling is first in the pool and must still lose."""

        served = [_table("t_rank1", "dA", rank=1), _table("t_rank2", "dB", rank=2)]
        rank2_sibling = _projection("p_rank2", "dB")
        rank1_sibling = _projection("p_rank1", "dA")
        pool = [*served, rank2_sibling, rank1_sibling]   # rank-2 candidate first
        a = _assess(_plan(HX12_Q), pool, served)

        self.assertEqual([r.chunk_id for r in a.selected], ["p_rank1"])
        self.assertEqual(a.anchors[0][1], ANCHOR_STRONG)
        self.assertEqual(a.anchors[0][2], 1)

    def test_the_rank_one_same_date_sibling_wins(self) -> None:
        """B: HX10 shape -- an earlier pool candidate anchors to rank 2."""

        served = [
            _projection("s_rank1", "dA", reference_date="2023년 09월 13일", rank=1),
            _projection("s_rank2", "dB", reference_date="2022년 12월 05일", rank=2),
        ]
        # A report draws on its own table, so it shares no event row with the
        # detail row -- the tie is the shared disclosure and reference date.
        rank2_report = _projection("rep_rank2", "dB", kind="holding_report",
                                   reference_date="2022년 12월 05일", event_row=7,
                                   before_ratio="6.07", after_ratio="7.12")
        rank1_report = _projection("rep_rank1", "dA", kind="holding_report",
                                   reference_date="2023년 09월 13일", event_row=7,
                                   before_ratio="8.41", after_ratio="9.46")
        pool = [*served, rank2_report, rank1_report]     # rank-2 candidate first
        a = _assess(_plan(HX10_Q, reporter=FAMILY), pool, served)

        self.assertEqual([r.chunk_id for r in a.selected], ["rep_rank1"])
        self.assertEqual(a.anchors[0][1], ANCHOR_MEDIUM)
        self.assertEqual(a.anchors[0][2], 1)

    def test_anchor_rank_dominates_global_candidate_minimum(self) -> None:
        """G: completing rank 1 with two beats completing rank 2 with one."""

        served = [_table("t_rank1", "dA", rank=1), _table("t_rank2", "dB", rank=2)]
        a_only = _field_projection("A_field1", "dA",
                                   fields={"before_shares": "613,758"})
        b_only = _field_projection("B_field2", "dA", event_row=2,
                                   kind="holding_report",
                                   fields={"after_ratio": "7.12"})
        c_both = _field_projection("C_both", "dB", kind="holding_report",
                                   fields={"before_shares": "613,758",
                                           "after_ratio": "7.12"})
        plan = _plan("테스트회사 국민연금기금 변동전 주식수와 보유 비율")
        a = _assess(plan, [*served, c_both, a_only, b_only], served)

        picked = sorted(r.chunk_id for r in a.selected)
        self.assertEqual(picked, ["A_field1", "B_field2"])
        self.assertNotIn("C_both", picked)
        self.assertTrue(all(anchor[2] == 1 for anchor in a.anchors))

    def test_strong_is_preferred_over_medium_on_the_same_anchor(self) -> None:
        """I: same served item, both tiers close the gap."""

        served = [_projection("s", "dA", reference_date="2023년 09월 13일", rank=1)]
        strong = _field_projection("strong", "dA", event_row=2,
                                   reference_date="2023년 09월 13일",
                                   kind="holding_report",
                                   fields={"after_ratio": "9.46",
                                           "before_ratio": "8.41"})
        medium = _field_projection("medium", "dA", event_row=9,
                                   reference_date="2023년 09월 13일",
                                   kind="holding_report",
                                   fields={"after_ratio": "9.46",
                                           "before_ratio": "8.41"})
        a = _assess(_plan(HX10_Q, reporter=FAMILY),
                    [*served, medium, strong], served)

        self.assertEqual([r.chunk_id for r in a.selected], ["strong"])
        self.assertEqual(a.anchors[0][1], ANCHOR_STRONG)

    def test_medium_is_used_before_dropping_to_a_lower_anchor(self) -> None:
        """J: only MEDIUM can close it at rank 1, so MEDIUM is used."""

        served = [
            _projection("s_rank1", "dA", reference_date="2023년 09월 13일", rank=1),
            _projection("s_rank2", "dB", reference_date="2022년 12월 05일", rank=2),
        ]
        medium_rank1 = _projection("m_rank1", "dA", kind="holding_report",
                                   reference_date="2023년 09월 13일", event_row=7,
                                   before_ratio="8.41", after_ratio="9.46")
        strong_rank2 = _projection("s_rank2_sib", "dB", kind="holding_report",
                                   reference_date="2022년 12월 05일",
                                   before_ratio="6.07", after_ratio="7.12")
        a = _assess(_plan(HX10_Q, reporter=FAMILY),
                    [*served, strong_rank2, medium_rank1], served)

        self.assertEqual([r.chunk_id for r in a.selected], ["m_rank1"])
        self.assertEqual(a.anchors[0][1], ANCHOR_MEDIUM)
        self.assertEqual(a.anchors[0][2], 1)

    def test_a_lower_anchor_supplies_only_what_the_higher_cannot(self) -> None:
        """K: rank 1 closes one field, rank 2 closes the remaining one."""

        served = [_table("t_rank1", "dA", rank=1), _table("t_rank2", "dB", rank=2)]
        rank1_part = _field_projection("rank1_shares", "dA",
                                       fields={"before_shares": "613,758"})
        rank2_part = _field_projection("rank2_ratio", "dB", kind="holding_report",
                                       fields={"after_ratio": "7.12"})
        plan = _plan("테스트회사 국민연금기금 변동전 주식수와 보유 비율")
        a = _assess(plan, [*served, rank1_part, rank2_part], served)

        self.assertEqual(sorted(r.chunk_id for r in a.selected),
                         ["rank1_shares", "rank2_ratio"])
        by_id = {chunk_id: rank for chunk_id, _tier, rank in a.anchors}
        self.assertEqual(by_id["rank1_shares"], 1)
        self.assertEqual(by_id["rank2_ratio"], 2)
        self.assertEqual(list(a.remaining_unresolved), [])


class SameAnchorMinimumTests(unittest.TestCase):
    """H: inside one anchor the selection is an exact minimum."""

    def test_one_candidate_covering_both_beats_two_covering_one_each(self) -> None:
        served = [_table("t_rank1", "dA", rank=1)]
        a_only = _field_projection("A_field1", "dA",
                                   fields={"before_shares": "613,758"})
        b_only = _field_projection("B_field2", "dA", kind="holding_report",
                                   fields={"after_ratio": "7.12"})
        c_both = _field_projection("C_both", "dA", kind="holding_report",
                                   fields={"before_shares": "613,758",
                                           "after_ratio": "7.12"})
        plan = _plan("테스트회사 국민연금기금 변동전 주식수와 보유 비율")
        # A and B come first in pool order; the exact minimum is C alone.
        a = _assess(plan, [*served, a_only, b_only, c_both], served)

        self.assertEqual([r.chunk_id for r in a.selected], ["C_both"])
        self.assertEqual(len(a.selected), 1)


class NoAnchorDeclineTests(unittest.TestCase):
    """F: Policy D -- a coverage candidate without an anchor is refused."""

    def test_an_unanchored_coverage_candidate_is_declined(self) -> None:
        served = _served_tables(3)
        # Same company and holder, closes the gap, but completes nothing served.
        stranger = _projection("stranger", "d_other")
        a = _assess(_plan(HX12_Q), [*served, stranger], served)

        self.assertEqual(a.status, STATUS_NO_ANCHOR)
        self.assertFalse(a.rescued)
        self.assertEqual(a.anchored_candidate_count, 0)
        self.assertEqual([r.chunk_id for r in a.results],
                         [r.chunk_id for _c, r in served])

    def test_the_decline_is_reported_distinctly_from_having_no_candidate(self) -> None:
        served = _served_tables(3)
        a = _assess(_plan(HX12_Q), served, served)
        self.assertEqual(a.status, STATUS_NO_CANDIDATE)


class NoDomainLiteralTests(unittest.TestCase):
    """L: production logic carries no Gold, company, or question identifiers."""

    def test_production_module_has_no_domain_literals(self) -> None:
        import re
        from pathlib import Path

        source = Path("app/reasoning/holding_evidence_coverage.py").read_text(
            encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#"))
        for pattern in (r"HX\d", r"holding_20\d", r"파마리서치", r"국민연금",
                        r"이마트", r"효성", r"t0019", r"t0002"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, code))

    def test_anchoring_works_for_an_unrelated_holder(self) -> None:
        served = [_table("t", "dA", rank=1)]
        cand = _projection("p", "dA", reporter="BlackRock Fund Advisors")
        plan = _plan(HX12_Q, reporter="BlackRock")
        a = _assess(plan, [*served, cand], served)

        self.assertEqual([r.chunk_id for r in a.selected], ["p"])
        self.assertEqual(a.anchors[0][1], ANCHOR_STRONG)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# P1-A3.1: one public evidence set
#
# The lane cannot write its top-up back into retrieval, so it carries its own
# list forward.  The defect these tests pin is what that costs if only half the
# pipeline reads it: the resolver answers from an enriched set while the caller
# is shown the original, so the answer cites a chunk the response never
# contains.  Everything below asserts one evidence set reaches both.
# --------------------------------------------------------------------------


class _CapturingOrchestrator(AgentOrchestrator):
    """The real orchestrator, keeping the result the pipeline consumed."""

    def run(self, *args, **kwargs):
        self.result = super().run(*args, **kwargs)
        return self.result


class _CapturingGenerator(CitationAwareAnswerGenerator):
    """The real generator, keeping the citations the answer was built from."""

    def generate(self, draft):
        self.generated = super().generate(draft)
        return self.generated


def _serve(plan, pairs, served=None):
    """Answer through the real public pipeline over a fixture execution.

    Returns the execution handed to the pipeline, the captured orchestrator and
    generator, and the public payload -- so a test can compare what the
    reasoning path used against what the caller was shown.
    """

    from types import SimpleNamespace

    from app.api.pipeline import AnswerPipeline
    from tests.test_p0d_pipeline import _Understanding

    served = pairs if served is None else served
    execution = SimpleNamespace(
        plan=plan,
        chunks=[chunk for chunk, _result in pairs],
        results=[result for _chunk, result in served],
    )

    class _Executor:
        def execute(self, _plan):
            return execution

    orchestrator = _CapturingOrchestrator()
    generator = _CapturingGenerator()
    pipeline = AnswerPipeline(
        understanding=_Understanding(plan),
        executor=_Executor(),
        orchestrator=orchestrator,
        generator=generator,
        answerability_guard=AnswerabilityGuard(),
    )
    payload = pipeline.answer("P1A3.1", plan.raw_query)
    return execution, orchestrator, generator, payload


def _public_ids(payload):
    return [row["chunk_id"] for row in payload["retrieved_context"]]


def _evidence_ids(result):
    """Every chunk the built evidence set actually carries."""

    ids = set(result.evidence_set.retrieval_order)
    for group in result.evidence_set.evidence_groups:
        ids.update(group.member_chunk_ids)
        ids.update(item.chunk_id for item in group.items)
    return ids


class PublicEvidenceSynchronizationTests(unittest.TestCase):
    """A: the enriched set reaches the resolver and the caller alike."""

    def _rescued(self):
        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")       # sibling of served rank 1
        plan = _plan(HX12_Q)
        return _serve(plan, [*tables, wanted], served=tables)

    def test_the_rescue_is_visible_in_the_public_context(self) -> None:
        execution, orchestrator, _gen, payload = self._rescued()
        result = orchestrator.result

        # Retrieval served no projection at all: the gap is real.
        self.assertEqual([r.chunk_id for r in execution.results],
                         ["t0", "t1", "t2", "t3"])
        self.assertTrue(result.holding_coverage.rescued)
        self.assertEqual([r.chunk_id for r in result.holding_coverage.selected],
                         ["p_gap"])

        # The evidence the resolver worked from, and the public list, agree.
        self.assertIn("p_gap", _evidence_ids(result))
        self.assertIn("p_gap", _public_ids(payload))

        displaced = list(result.holding_coverage.displaced)
        self.assertEqual(displaced, ["t3"])
        self.assertNotIn("t3", _public_ids(payload))
        self.assertEqual(len(payload["retrieved_context"]), len(execution.results))

    def test_the_final_evidence_and_the_public_context_are_the_same_set(self) -> None:
        """The strict identity requirement, asserted as sets of chunk ids."""

        _execution, orchestrator, _gen, payload = self._rescued()
        result = orchestrator.result

        self.assertEqual(
            {r.chunk_id for r in result.evidence_results},
            set(_public_ids(payload)),
        )
        self.assertEqual(
            [r.chunk_id for r in result.evidence_results],
            _public_ids(payload),
        )

    def test_ranks_are_renumbered_without_gaps(self) -> None:
        _e, _o, _g, payload = self._rescued()
        ranks = [row["rank"] for row in payload["retrieved_context"]]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))


class CitationCompletenessTests(unittest.TestCase):
    """B: nothing the answer cites may be missing from the response."""

    def _rescued(self):
        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        return _serve(plan, [*tables, wanted], served=tables)

    def test_every_cited_chunk_is_present_publicly(self) -> None:
        _e, _o, generator, payload = self._rescued()

        cited = {citation.chunk_id for citation in generator.generated.citations}
        self.assertTrue(cited, "the fixture must produce a cited answer")
        self.assertLessEqual(cited, set(_public_ids(payload)))

    def test_the_rescued_projection_is_among_the_citations(self) -> None:
        _e, _o, generator, payload = self._rescued()

        cited = {citation.chunk_id for citation in generator.generated.citations}
        self.assertIn("p_gap", cited)
        self.assertIn("p_gap", _public_ids(payload))


class RawAnchorPreservationTests(unittest.TestCase):
    """C: the anchor is completed, not replaced."""

    def test_the_served_anchor_and_its_sibling_are_both_public(self) -> None:
        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        _e, orchestrator, _g, payload = _serve(plan, [*tables, wanted], served=tables)

        anchors = orchestrator.result.holding_coverage.anchors
        self.assertEqual([tier for _cid, tier, _rank in anchors], [ANCHOR_STRONG])
        self.assertEqual([rank for _cid, _tier, rank in anchors], [1])

        rows = {row["chunk_id"]: row for row in payload["retrieved_context"]}
        self.assertIn("t0", rows)                      # the raw rank-1 anchor
        self.assertEqual(rows["t0"]["chunk_type"], "table")
        self.assertIn("p_gap", rows)                   # its structured sibling
        self.assertEqual(rows["p_gap"]["chunk_type"], "table_projection")


class NoRescueIdentityTests(unittest.TestCase):
    """D and E: an untouched question is served exactly what it always was."""

    def _unchanged(self, plan, pairs, served=None):
        from app.api.pipeline import retrieved_context

        execution, orchestrator, _g, payload = _serve(plan, pairs, served=served)
        before = retrieved_context(execution, 10)
        self.assertEqual(payload["retrieved_context"], before)
        return orchestrator.result

    def test_a_covered_holding_request_is_untouched(self) -> None:
        served = [_projection("served", "d1", rank=1)]
        absent = _projection("absent", "d1", reference_date="2023년 03월 07일", rank=9)
        result = self._unchanged(_plan(HX08_Q), [*served, absent], served=served)

        self.assertFalse(result.holding_coverage.rescued)
        self.assertEqual(result.holding_coverage.status, STATUS_COVERED)

    def test_a_declined_holding_request_is_untouched(self) -> None:
        """No anchor: the lane declines, and the response must not move."""

        tables = _served_tables(3)
        orphan = _projection("orphan", "far_away_doc")
        result = self._unchanged(_plan(HX12_Q), [*tables, orphan], served=tables)

        self.assertFalse(result.holding_coverage.rescued)
        self.assertEqual(result.holding_coverage.status, STATUS_NO_ANCHOR)

    def test_a_non_holding_request_is_untouched(self) -> None:
        pairs = [_table(f"t{i}", f"d{i}", rank=i + 1, doc_group="periodic")
                 for i in range(3)]
        plan = _plan("테스트회사 2024년 매출액", task_type="financial_metric",
                     reporter=None, metric="매출액", route=("periodic",))
        result = self._unchanged(plan, pairs)

        self.assertFalse(result.holding_coverage.evaluated)

    def test_nothing_enriched_reuses_the_execution_itself(self) -> None:
        """The no-op path must not even build a view object."""

        from types import SimpleNamespace

        from app.api.pipeline import final_evidence

        execution = SimpleNamespace(results=["a", "b"])
        self.assertIs(final_evidence(execution, SimpleNamespace()), execution)
        self.assertIs(
            final_evidence(execution, SimpleNamespace(evidence_results=())),
            execution,
        )
        self.assertIs(
            final_evidence(execution, SimpleNamespace(evidence_results=("a", "b"))),
            execution,
        )


class ImmutableExecutionTests(unittest.TestCase):
    """F: the public output is repaired by carrying, never by mutating."""

    def test_retrieval_output_survives_a_rescue_unchanged(self) -> None:
        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        execution, orchestrator, _g, payload = _serve(
            plan, [*tables, wanted], served=tables)

        self.assertEqual([r.chunk_id for r in execution.results],
                         ["t0", "t1", "t2", "t3"])
        self.assertEqual([r.rank for r in execution.results], [1, 2, 3, 4])
        self.assertNotEqual(_public_ids(payload),
                            [r.chunk_id for r in execution.results])
        self.assertTrue(orchestrator.result.holding_coverage.rescued)

    def test_the_view_delegates_every_other_retrieval_attribute(self) -> None:
        from types import SimpleNamespace

        from app.api.pipeline import final_evidence

        execution = SimpleNamespace(
            results=["a"], chunks=["c"], routing={"lane": "holding"},
            correction_expansion={"correction_expanded": True},
        )
        view = final_evidence(execution, SimpleNamespace(evidence_results=("b",)))

        self.assertEqual(list(view.results), ["b"])
        self.assertEqual(view.chunks, ["c"])
        self.assertEqual(view.routing, {"lane": "holding"})
        self.assertEqual(view.correction_expansion, {"correction_expanded": True})
        self.assertEqual(execution.results, ["a"])     # original untouched


class PublicSliceTests(unittest.TestCase):
    """G and H: the response shape does not move."""

    def test_the_public_context_never_exceeds_top_k(self) -> None:
        from app.api.settings import ApiSettings

        top_k = ApiSettings().top_k
        for size in (2, 5, 12):
            with self.subTest(size=size):
                tables = _served_tables(size)
                wanted = _projection("p_gap", "d0")
                plan = _plan(HX12_Q)
                _e, _o, _g, payload = _serve(plan, [*tables, wanted], served=tables)
                self.assertLessEqual(len(payload["retrieved_context"]), top_k)

    def test_the_rescued_response_keeps_exactly_five_top_level_keys(self) -> None:
        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        _e, _o, _g, payload = _serve(plan, [*tables, wanted], served=tables)

        self.assertEqual(
            set(payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )
        self.assertNotIn("holding_coverage", payload["think_trace"])
        self.assertNotIn("rescued_context", payload)
        self.assertNotIn("enriched_context", payload)


class PublicSanitizationTests(unittest.TestCase):
    """I: the chunk is published, the internal rescue marker is not."""

    def test_the_marker_stays_internal_while_the_chunk_is_published(self) -> None:
        import json

        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        _e, orchestrator, _g, payload = _serve(plan, [*tables, wanted], served=tables)

        promoted = next(r for r in orchestrator.result.evidence_results
                        if r.chunk_id == "p_gap")
        self.assertIn(PROVENANCE_KEY, promoted.metadata_match)

        # The chunk is published in full; the marker rides on the retrieval
        # result's metadata_match, which this renderer has never serialised.
        row = next(r for r in payload["retrieved_context"] if r["chunk_id"] == "p_gap")
        self.assertTrue(row["content"])
        self.assertTrue(row["source_refs"])
        self.assertNotIn(
            PROVENANCE_KEY,
            json.dumps(payload["retrieved_context"], ensure_ascii=False),
        )
        self.assertNotIn("metadata_match", row)
        # The stage name stays: it is the existing execution summary, not
        # internal metadata attached to an evidence row.
        self.assertIn("holding_evidence_coverage", payload["think_trace"]["stages"])


class P1A2PublicRegressionTests(unittest.TestCase):
    """J: the frozen grouping intent still drives the served path."""

    def test_the_grouping_intent_is_unchanged_on_the_public_path(self) -> None:
        from app.agent.orchestrator import _EXECUTION_GROUPING_INTENT

        self.assertEqual(_EXECUTION_GROUPING_INTENT,
                         {"holding_event": "holding_change"})
        tables = _served_tables(4)
        wanted = _projection("p_gap", "d0")
        plan = _plan(HX12_Q)
        _e, orchestrator, _g, payload = _serve(plan, [*tables, wanted], served=tables)
        result = orchestrator.result

        self.assertEqual(result.task_decision.task_type, "holding_event")
        self.assertTrue([g for g in result.evidence_set.evidence_groups
                         if g.group_type == "holding_event"])
        self.assertEqual(payload["think_trace"]["route"], "holding_event_resolver")
        self.assertIn("holding_evidence_coverage", payload["think_trace"]["stages"])


# --------------------------------------------------------- acquisition proof

#: A holder no issuer-scoped universe lists, as a filer usually is.
HOLDER = "가상지주"
#: The same holder written the way a person writes it.
HOLDER_LEGAL_FORM = "(주)가상지주"
#: A different holder whose name merely *contains* the bound one.
NEIGHBOUR_HOLDER = "가상지주캐피탈"
#: A holder family and one of its members, which the frozen contract treats as
#: one identity through its own suffix rule.
SYNTH_FAMILY = "가상연금"
SYNTH_MEMBER = "가상연금공단"

ACQUISITION_Q = "가상발행사 주식 취득단가"
#: An ordinary holding request, reusing this module's own pinned query so the
#: comparison is against the behaviour those tests already fix.
ORDINARY_Q = HX12_Q


def _acquisition_row(
    chunk_id,
    doc_id="d1",
    *,
    reporter=HOLDER,
    method="장내매수(+)",
    change="1,000",
    price="-",
    change_date="2024.02.03",
    reference_date="2024년 02월 03일",
    table_id="t0031",
    row=2,
    citable=True,
    headers=None,
    rows=None,
    rank=1,
):
    """A detail row shaped like the frozen corpus' 세부변동내역 table.

    The acquisition columns live only in ``column_headers``/``table_rows``;
    the projection labels carry the position, exactly as the corpus stores it.
    Its table differs from the summary's on purpose, so anchoring has to fall
    back to the reference date the two share -- which is what production does.
    """

    default_headers = [
        "성명(명칭)", "변동일*", "취득/처분방법",
        "변동 내역 / 증감", "취득/처분단가**", "비 고",
    ]
    default_rows = [[
        {"text": value} for value in
        (reporter, change_date, method, change, price, "")
    ]]
    fields = {"보고자/보유자": reporter, "기준일/보고일": reference_date}
    ref = {"table_id": table_id, "row_start": row, "row_end": row}
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "corp_code": "00970453",
        "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서(일반)",
        "rcept_dt": "2024-02-05",
        "chunk_type": "table_projection",
        "projection_type": "holding_detail_row",
        "projection_fields": fields,
        "projection_field_refs": (
            {label: [dict(ref)] for label in fields} if citable else {}
        ),
        "source_refs": [dict(ref)] if citable else [],
        "source_table_id": table_id,
        "table_id": table_id,
        "row_start": row,
        "row_end": row,
        "column_headers": default_headers if headers is None else headers,
        "table_rows": default_rows if rows is None else rows,
        "section_path": ["제3부 직전보고일 이후 대량변동 내역", "2. 세부변동내역"],
        "content": " ".join(f"[{k}] {v}" for k, v in fields.items()),
        "retrieval_text": " ".join(f"[{k}] {v}" for k, v in fields.items()),
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()))


def _summary(chunk_id="s1", doc_id="d1", *, reporter=HOLDER,
             reference_date="2024년 02월 03일", rank=1):
    """The position projection retrieval actually ranks for these questions."""

    return _projection(
        chunk_id, doc_id, kind="holding_report", reporter=reporter,
        reference_date=reference_date, after_ratio="7.10", rank=rank,
    )


def _acquisition_plan(reporter=HOLDER, raw_query=ACQUISITION_Q):
    return _plan(raw_query, task_type="holding_change", reporter=reporter,
                 metric="acquisition_unit_price")


class AcquisitionProofActivationTests(unittest.TestCase):
    """The acquisition lane runs only for the answer field that needs it."""

    def test_acquisition_unit_price_request_activates_the_lane(self) -> None:
        summary = _summary()
        detail = _acquisition_row("a1")

        result = _assess(_acquisition_plan(), [summary, detail], [summary])

        self.assertIn(ACQUISITION_PROOF, result.requested)
        self.assertEqual(result.status, STATUS_RESCUED)
        self.assertEqual(result.rescue_mode, RESCUE_MODE_ACQUISITION_PROOF)
        self.assertEqual([r.chunk_id for r in result.selected], ["a1"])

    def test_ordinary_holding_query_is_untouched_by_the_lane(self) -> None:
        """Byte-for-byte the pre-feature selection, proved against the old body."""

        plan = _plan(ORDINARY_Q, task_type="holding_change")
        pool = [*_served_tables(2), _projection("p1", "d0", rank=3)]
        served = pool[:2]

        before = _assess_field_coverage(
            plan.raw_query, plan, [c for c, _ in pool], [r for _, r in served],
            routed_task_type="holding_event",
        )
        after = _assess(plan, pool, served)

        self.assertNotIn(ACQUISITION_PROOF, after.requested)
        self.assertEqual(after.requested, before.requested)
        self.assertEqual(after.status, before.status)
        self.assertEqual([r.chunk_id for r in after.selected],
                         [r.chunk_id for r in before.selected])
        self.assertEqual([r.chunk_id for r in after.results],
                         [r.chunk_id for r in before.results])
        self.assertEqual([r.rank for r in after.results],
                         [r.rank for r in before.results])

    def test_non_holding_execution_never_reaches_the_lane(self) -> None:
        summary = _summary()
        detail = _acquisition_row("a1")

        result = _assess(_acquisition_plan(), [summary, detail], [summary],
                         routed="general_evidence")

        self.assertEqual(result.status, STATUS_NOT_HOLDING)
        self.assertNotIn(ACQUISITION_PROOF, result.requested)


class AcquisitionAmbiguityPreservationTests(unittest.TestCase):
    """Every legitimate proof row reaches the resolver.  This lane picks none."""

    def test_minimum_subset_would_have_hidden_the_second_row(self) -> None:
        """Why the acquisition lane cannot reuse the ordinary selection."""

        first, _ = _acquisition_row("a1")
        second, _ = _acquisition_row("a2", row=3)

        picked = _minimum_subset(
            [(first, frozenset({ACQUISITION_PROOF})),
             (second, frozenset({ACQUISITION_PROOF}))],
            [ACQUISITION_PROOF],
        )

        # One row closes the requirement as well as two, so the ordinary
        # selection keeps exactly one -- which would turn a contested
        # acquisition into a confident single event.
        self.assertEqual([c.chunk_id for c in picked], ["a1"])

    def test_both_acquisition_rows_are_promoted(self) -> None:
        summary = _summary()
        first = _acquisition_row("a1", row=2, change="1,000")
        second = _acquisition_row("a2", row=3, change="2,000")

        result = _assess(_acquisition_plan(), [summary, first, second], [summary])

        # Deliberately more rows than the requirement needs: the ambiguity is
        # the resolver's to see, so selected_count exceeds the minimum.
        self.assertEqual([r.chunk_id for r in result.selected], ["a1", "a2"])
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(len(result.anchors), 2)


class AcquisitionReporterIdentityTests(unittest.TestCase):
    """An acquisition row must belong to the holder the question named."""

    def test_containment_neighbour_is_rejected_by_strict_identity(self) -> None:
        summary = _summary()
        neighbour = _acquisition_row("a1", reporter=NEIGHBOUR_HOLDER)

        # The ordinary predicate would have accepted it; the frozen identity
        # contract does not, and the acquisition lane reads the latter.
        self.assertTrue(reporter_compatible(neighbour[0].chunk, HOLDER))
        self.assertFalse(strict_reporter_identity(neighbour[0].chunk, HOLDER))

        result = _assess(_acquisition_plan(), [summary, neighbour], [summary])

        self.assertEqual(result.selected, ())
        self.assertEqual(result.status, STATUS_NO_ACQUISITION_PROOF)

    def test_frozen_equivalent_spellings_are_allowed(self) -> None:
        for label, requested, holder in (
            ("legal form", HOLDER, HOLDER_LEGAL_FORM),
            ("family suffix", SYNTH_FAMILY, SYNTH_MEMBER),
        ):
            with self.subTest(identity=label):
                summary = _summary(reporter=holder)
                detail = _acquisition_row("a1", reporter=holder)

                self.assertTrue(strict_reporter_identity(detail[0].chunk, requested))
                result = _assess(_acquisition_plan(reporter=requested),
                                 [summary, detail], [summary])

                self.assertEqual([r.chunk_id for r in result.selected], ["a1"])

    def test_unbound_reporter_recovers_nothing(self) -> None:
        for label, reporter in (("none", None), ("blank", "   ")):
            with self.subTest(reporter=label):
                summary = _summary()
                detail = _acquisition_row("a1")

                self.assertFalse(strict_reporter_identity(detail[0].chunk, reporter))
                result = _assess(_acquisition_plan(reporter=reporter),
                                 [summary, detail], [summary])

                self.assertEqual(result.selected, ())


class AcquisitionProofBoundsTests(unittest.TestCase):
    """What may not become acquisition proof."""

    def test_disposal_row_proves_no_acquisition(self) -> None:
        summary = _summary()
        disposal = _acquisition_row("a1", method="장내매도(-)", change="-1,000")

        self.assertFalse(proves_acquisition(disposal[0].chunk))
        result = _assess(_acquisition_plan(), [summary, disposal], [summary])

        self.assertEqual(result.selected, ())
        self.assertEqual(result.status, STATUS_NO_ACQUISITION_PROOF)

    def test_other_document_is_not_anchored(self) -> None:
        summary = _summary(doc_id="d1")
        elsewhere = _acquisition_row("a1", doc_id="d9")

        result = _assess(_acquisition_plan(), [summary, elsewhere], [summary])

        self.assertEqual(result.selected, ())

    def test_unanchored_same_document_row_is_rejected(self) -> None:
        summary = _summary(reference_date="2024년 02월 03일")
        # Same filing, a different event: no shared rows and no shared date.
        other_event = _acquisition_row("a1", reference_date="2024년 09월 09일")

        result = _assess(_acquisition_plan(), [summary, other_event], [summary])

        self.assertEqual(result.selected, ())

    def test_malformed_and_uncitable_rows_are_rejected(self) -> None:
        summary = _summary()
        cases = (
            ("no headers", _acquisition_row("a1", headers=[])),
            ("two rows", _acquisition_row(
                "a1", rows=[[{"text": "x"}], [{"text": "y"}]])),
            ("zero quantity", _acquisition_row("a1", change="0")),
            ("uncitable", _acquisition_row("a1", citable=False)),
        )
        for label, candidate in cases:
            with self.subTest(row=label):
                result = _assess(_acquisition_plan(), [summary, candidate], [summary])
                self.assertEqual(result.selected, ())

    def test_no_detail_row_declines_cleanly(self) -> None:
        summary = _summary()

        result = _assess(_acquisition_plan(), [summary], [summary])

        self.assertEqual(result.selected, ())
        self.assertEqual(result.status, STATUS_NO_ACQUISITION_PROOF)
        self.assertIn(ACQUISITION_PROOF, result.remaining_unresolved)


class AcquisitionRetrievalImmutabilityTests(unittest.TestCase):
    """Recovered rows are appended.  Nothing served is touched."""

    def test_baseline_rows_keep_object_order_and_rank(self) -> None:
        summary = _summary("s1", rank=1)
        second = _summary("s2", rank=2)
        detail = _acquisition_row("a1", rank=9)
        served = [summary, second]

        result = _assess(_acquisition_plan(), [summary, second, detail], served)

        baseline = [r for _, r in served]
        self.assertEqual([r.chunk_id for r in result.results[:2]],
                         [r.chunk_id for r in baseline])
        self.assertEqual([r.rank for r in result.results[:2]],
                         [r.rank for r in baseline])
        for original, kept in zip(baseline, result.results[:2]):
            self.assertIs(original, kept)
        self.assertEqual(result.displaced, ())

    def test_recovered_row_is_appended_with_recovery_provenance(self) -> None:
        summary = _summary()
        detail = _acquisition_row("a1")

        result = _assess(_acquisition_plan(), [summary, detail], [summary])
        promoted = result.results[-1]

        self.assertEqual(promoted.chunk_id, "a1")
        self.assertEqual(promoted.rank, len(result.results))
        self.assertGreater(promoted.rank, 1)
        self.assertEqual(promoted.bm25_score, 0.0)
        self.assertEqual(promoted.metadata_match[PROVENANCE_KEY],
                         {"selected_for": "holding_field_coverage"})
        self.assertEqual(result.anchors[0][1], ANCHOR_MEDIUM)
