from __future__ import annotations

import csv
import json
import unittest
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

from app.reasoning.holding_company_role_resolution import (
    ROLE_PATH_FILER,
    ROLE_PROVENANCE_KEY,
    HoldingCompanyRoleResolver,
)
from app.reasoning.holding_report_index import HoldingReportIndex, HoldingReportRecord
from app.reasoning.holding_reporter import canonical_reporter_key
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.query_plan import QueryPlan
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_understanding import (
    ACTOR_SOURCE_DIRECTED_HOLDER,
    HOLDING_ACTOR_CANDIDATE_KEY,
)
from app.reasoning.query_validation import (
    CorpusScope,
    QuerySlotSource,
    QueryState,
    QueryValidator,
    _holding_company_role_resolution_allowed,
    _holding_filer_resolution_allowed,
)


ROOT = Path(__file__).resolve().parents[1]


def _company_rows():
    with (ROOT / "data/corpus/universe.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def _compact(value: str) -> str:
    return "".join(value.split()).casefold()


def _resolver(question: str):
    compact = _compact(question)
    matches = []
    for row in _company_rows():
        for name in (row["corp_name"], row["listed_name"]):
            if name and _compact(name) in compact:
                matches.append((len(_compact(name)), row))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _components(*, bounded_events: bool = False):
    scope = CorpusScope.repository_default()
    assert scope is not None
    if bounded_events:
        scope = replace(scope, event_from="2022-01-01", event_to="2026-06-30")
    understanding = QueryUnderstanding(company_resolver=_resolver)
    validator = QueryValidator(
        corpus_scope=scope,
        multi_document_planner=MultiDocumentPlanner(),
    )
    return understanding, validator


class ClearQueryRegressionTests(unittest.TestCase):
    def test_clear_correction_query_stays_deterministically_resolved(self) -> None:
        understanding, validator = _components()
        question = (
            "현대건설의 2023년 6월 26일 단일판매·공급계약체결 공시는 "
            "최종 정정 기준으로 계약 내용이 어떻게 되어 있어?"
        )

        result = validator.validate(understanding.understand(question))

        self.assertIs(result.state, QueryState.RESOLVED)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.plan.period.period_type, "receipt_date")
        self.assertEqual(result.plan.evidence["correction_intent"], "latest")

    def test_gold60_all_resolve_without_semantic_fallback(self) -> None:
        understanding, validator = _components()
        questions = [
            json.loads(line)["question"]
            for line in (
                ROOT
                / "reports/evaluation/gold60/2026-08-21-agent-90pct/gold60_agent_questions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]

        results = [validator.validate(understanding.understand(q)) for q in questions]

        self.assertEqual(len(results), 60)
        self.assertTrue(all(result.state is QueryState.RESOLVED for result in results))
        self.assertTrue(all(not result.fallback_used for result in results))

    def test_all_26_p0c_official_style_queries_resolve(self) -> None:
        understanding, validator = _components()
        cases = json.loads(
            (ROOT / "data/evaluation/p0c_official_style_v1.json").read_text(
                encoding="utf-8"
            )
        )["cases"]

        results = [
            validator.validate(understanding.understand(case["question"]))
            for case in cases
        ]

        self.assertEqual(len(results), 26)
        self.assertTrue(all(result.state is QueryState.RESOLVED for result in results))

    def test_runtime_event_bounds_preserve_p0c_contract_date_semantics(self) -> None:
        scope = CorpusScope.repository_default()
        assert scope is not None
        calls = 0

        def event_scope_provider():
            nonlocal calls
            calls += 1
            return "2021-01-01", "2026-06-30"

        understanding = QueryUnderstanding(
            scope.company_aliases(), company_resolver=_resolver
        )
        validator = QueryValidator(
            corpus_scope=scope,
            multi_document_planner=MultiDocumentPlanner(),
            event_scope_provider=event_scope_provider,
        )

        result = validator.validate(
            understanding.understand(
                "두산퓨얼셀이 2022년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"
            )
        )

        self.assertEqual(result.plan.period.period_type, "reference_year")
        self.assertEqual(result.plan.date_basis, "contract_date")
        self.assertGreaterEqual(calls, 1)
        self.assertIs(result.state, QueryState.RESOLVED)

    def test_deterministic_slots_are_locked(self) -> None:
        understanding, validator = _components()
        result = validator.validate(
            understanding.understand("삼성전자 2025년 연결 매출액은 얼마인가?")
        )

        self.assertIs(result.state, QueryState.RESOLVED)
        for name in ("company", "task_type", "period", "metric", "operation"):
            self.assertTrue(result.slots[name].locked, name)
            self.assertIs(result.slots[name].source, QuerySlotSource.DETERMINISTIC)

    def test_company_name_and_corp_code_must_match(self) -> None:
        _, validator = _components()
        plan = QueryUnderstanding(company_resolver=_resolver).understand(
            "삼성전자 2025년 연결 매출액은 얼마인가?"
        )
        conflicting = replace(plan, corp_codes=("00126478",))

        result = validator.validate(conflicting)

        self.assertIs(result.state, QueryState.AMBIGUOUS)
        self.assertEqual(result.slots["company"].status.value, "invalid")

    def test_multi_company_comparison_resolves_every_corp_code(self) -> None:
        scope = CorpusScope.repository_default()
        assert scope is not None
        understanding = QueryUnderstanding(
            scope.company_aliases(), company_resolver=_resolver
        )
        validator = QueryValidator(corpus_scope=scope)

        result = validator.validate(
            understanding.understand(
                "삼성전자와 삼성중공업의 2025년 매출액을 비교해줘"
            )
        )

        self.assertIs(result.state, QueryState.RESOLVED)
        self.assertEqual(len(result.plan.companies), 2)
        self.assertEqual(len(result.plan.corp_codes), 2)


class AmbiguousAndIncompleteQueryTests(unittest.TestCase):
    def test_twenty_ambiguous_queries_never_resolve_by_guessing(self) -> None:
        understanding, validator = _components()
        questions = [
            "삼성중공업이 작년에 취소한 건 있어?",
            "삼성중공업 취소된 공시 찾아줘",
            "삼성중공업의 취소 건 알려줘",
            "삼성중공업 취소가 있었나?",
            "삼성중공업에서 취소한 내용은?",
            "삼성전자 전에 바뀐 거 알려줘",
            "삼성전자 변경된 거 정리해줘",
            "삼성전자 그건 어떻게 됐어?",
            "삼성전자 무슨 일이 있었어?",
            "삼성전자 피해 본 건 정리해줘",
            "현대건설이 취소한 건 있어?",
            "현대건설 취소 공시는 무엇이야?",
            "한화오션이 취소한 내용 알려줘",
            "한화오션 취소 여부를 확인해줘",
            "대우건설 취소 건 목록",
            "셀트리온 전에 바뀐 거 알려줘",
            "카카오 변경된 거 찾아줘",
            "NAVER 그건 어떻게 됐어?",
            "기아 무슨 일이 있었어?",
            "효성중공업 피해 본 건 정리해줘",
        ]

        results = [validator.validate(understanding.understand(q)) for q in questions]

        self.assertEqual(len(results), 20)
        self.assertTrue(all(result.state is QueryState.AMBIGUOUS for result in results))
        self.assertTrue(all(result.clarification is not None for result in results))

    def test_missing_company_is_incomplete_and_has_no_retrieval_route(self) -> None:
        understanding, validator = _components()
        questions = [
            "그 회사 공급계약 알려줘",
            "이 회사 2025년 매출액은?",
            "해당 회사 시설투자 내용은?",
            "어느 회사의 유상증자인지 알려줘",
            "그 기업 보유주식 변동은?",
        ]

        results = [validator.validate(understanding.understand(q)) for q in questions]

        self.assertTrue(all(result.state is QueryState.INCOMPLETE for result in results))
        self.assertTrue(all("company" in result.missing_slots for result in results))


class UnsupportedAndOutOfScopeTests(unittest.TestCase):
    def test_unsupported_requests_are_blocked(self) -> None:
        understanding, validator = _components(bounded_events=True)
        questions = [
            "삼성전자 주가가 내년에 오를까?",
            "삼성전자 주식을 지금 매수해도 될까?",
            "삼성전자 임원의 개인 신상을 알려줘",
            "삼성전자 관련 루머를 정리해줘",
            "삼성전자 공시를 이메일로 보내줘",
            "삼성전자 공시 문서를 수정해줘",
            "삼성전자 오늘 뉴스를 알려줘",
        ]

        results = [validator.validate(understanding.understand(q)) for q in questions]

        self.assertTrue(all(result.state is QueryState.UNSUPPORTED for result in results))

    def test_company_and_period_outside_frozen_scope_are_blocked(self) -> None:
        understanding, validator = _components(bounded_events=True)
        questions = [
            "애플 2025년 매출액 알려줘",
            "마이크로소프트 2025년 영업이익 알려줘",
            "삼성전자 2021년 매출액 알려줘",
            "삼성전자 2027년 매출액 알려줘",
            "삼성중공업 2021년 공급계약 알려줘",
        ]

        results = [validator.validate(understanding.understand(q)) for q in questions]

        self.assertTrue(all(result.state is QueryState.OUT_OF_SCOPE for result in results))

    def test_explicit_reference_year_is_in_scope_before_semantic_completeness(self) -> None:
        understanding, validator = _components()

        before = validator.validate(
            understanding.understand("삼성중공업의 2021년 계약을 알려줘")
        )
        inside = validator.validate(
            understanding.understand("삼성중공업의 2025년 계약을 알려줘")
        )

        self.assertEqual(before.plan.period.period_type, "reference_year")
        self.assertIs(before.state, QueryState.OUT_OF_SCOPE)
        self.assertFalse(before.retrieval_allowed)
        self.assertEqual(before.issues, ("period_out_of_corpus",))
        self.assertEqual(inside.plan.period.period_type, "reference_year")
        self.assertIsNot(inside.state, QueryState.OUT_OF_SCOPE)
        self.assertIs(inside.state, QueryState.AMBIGUOUS)
        self.assertFalse(inside.retrieval_allowed)
        self.assertIn("event_family", inside.ambiguous_slots)
        self.assertIn("date_basis", inside.ambiguous_slots)
        self.assertTrue(inside.fallback_recommended)

    def test_supported_domain_cues_keep_existing_deterministic_semantics(self) -> None:
        understanding, validator = _components()
        cases = (
            (
                "삼성중공업의 2025년 공급계약을 알려줘",
                "corporate_event",
                "supply_contract",
                None,
            ),
            (
                "삼성중공업의 2025년 시설투자를 알려줘",
                "corporate_event",
                "facility_investment",
                None,
            ),
            ("삼성중공업의 2025년 매출을 알려줘", "financial_metric", None, "매출액"),
            (
                "삼성중공업의 2025년 영업이익을 알려줘",
                "financial_metric",
                None,
                "영업이익",
            ),
        )

        for question, task, event, metric in cases:
            with self.subTest(question=question):
                plan = understanding.understand(question)
                result = validator.validate(plan)

                self.assertEqual(plan.task_type, task)
                self.assertEqual(plan.event_type, event)
                self.assertEqual(plan.metric, metric)
                self.assertIs(result.state, QueryState.RESOLVED)

    def test_fiscal_year_before_corpus_is_blocked(self) -> None:
        understanding, validator = _components()

        result = validator.validate(
            understanding.understand("삼성전자 2021년 매출액 알려줘")
        )

        self.assertEqual(result.plan.period.period_type, "fiscal_year")
        self.assertIs(result.state, QueryState.OUT_OF_SCOPE)
        self.assertEqual(result.issues, ("period_out_of_corpus",))

    def test_receipt_date_before_corpus_is_blocked(self) -> None:
        understanding, validator = _components()
        question = (
            "현대건설의 2021년에 공시한 단일판매·공급계약체결 공시는 "
            "최종 정정 기준으로 어떻게 되어 있어?"
        )

        result = validator.validate(understanding.understand(question))

        self.assertEqual(result.plan.period.period_type, "receipt_date")
        self.assertIs(result.state, QueryState.OUT_OF_SCOPE)
        self.assertEqual(result.issues, ("period_out_of_corpus",))

def _and_particle(name: str) -> str:
    """Attach 와/과 to a company name, picking the form Korean requires."""

    last = name[-1]
    if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28:
        return name + "과"
    return name + "와"


class ComparisonFirewallPrecedenceTests(unittest.TestCase):
    """A comparison frame decides the company slot before any role rule can.

    No issuer/reporter reinterpretation exists yet, so the visible outcome of
    these questions is the ambiguity they already had.  What is asserted here
    is the precedence such a rule would have to obey, and the recorded decision
    it would have to read.
    """

    #: Corpus companies used as data.  Neither the classifier nor the guard may
    #: special-case any of them, nor any relation between them.
    A = "한화오션"
    B = "한화에어로스페이스"
    C = "에스엠"
    D = "하이브"

    def validated(self, query: str):
        scope = CorpusScope.repository_default()
        assert scope is not None
        understanding = QueryUnderstanding(
            scope.company_aliases(), company_resolver=_resolver
        )
        validator = QueryValidator(
            corpus_scope=scope, multi_document_planner=MultiDocumentPlanner()
        )
        return validator.validate(understanding.understand(query))

    def test_a_cross_company_frame_blocks_role_reinterpretation(self) -> None:
        for query in (
            f"{_and_particle(self.A)} {self.B} 중 어디가 보유 주식수가 더 많아?",
            f"{_and_particle(self.A)} {self.B} 중 누가 보유 비율이 더 높아?",
            f"{_and_particle(self.A)} {self.B} 각각 보유 주식수 알려줘",
            f"{self.A}보다 {self.B}의 보유 비율이 높아?",
            f"{_and_particle(self.A)} {self.B}의 지분율 차이는 얼마야?",
            f"{_and_particle(self.B)} {self.A} 중 어디가 보유 비율이 더 높아?",
        ):
            with self.subTest(query=query):
                result = self.validated(query)
                self.assertIs(result.state, QueryState.AMBIGUOUS)
                self.assertTrue(result.plan.evidence["role_reinterpretation_blocked"])

    def test_an_unresolved_comparative_frame_fails_closed(self) -> None:
        result = self.validated(
            f"{self.D}가 {self.C} 주식을 더 많이 취득한 시점은 언제야?"
        )

        self.assertEqual(result.plan.evidence["comparison_frame"], "uncertain")
        self.assertIs(result.state, QueryState.AMBIGUOUS)
        self.assertTrue(result.plan.evidence["role_reinterpretation_blocked"])

    def test_issuer_reporter_questions_are_not_blocked(self) -> None:
        """The firewall must cost the target questions nothing.

        These name two companies in a disclosure-about-a-holder shape.  They
        stay ambiguous today because nothing resolves the roles yet -- but the
        firewall must not be the reason.
        """

        for query in (
            f"{self.C}에서 {self.D}가 보유한 주식수 알려줘",
            f"{self.C} 공시에서 {self.D}의 보유 비율 알려줘",
            f"{self.D}가 {self.C} 주식을 얼마나 보유하고 있어?",
            f"{self.C}에 대한 {self.D}의 지분 변동 알려줘",
            f"{self.A}에서 {self.B}가 보유한 주식수 알려줘",
            f"{self.B}가 {self.A} 주식을 얼마나 보유하고 있어?",
            f"{self.C} {self.D} 이번 보고 보유 주식수와 비율",
            f"{self.C}에서 {self.D}가 직전 보고 대비 늘린 주식수",
        ):
            with self.subTest(query=query):
                result = self.validated(query)
                self.assertIs(result.state, QueryState.AMBIGUOUS)
                self.assertFalse(result.plan.evidence["role_reinterpretation_blocked"])

    def test_bounded_ownership_intent_still_waits_for_role_resolution(self) -> None:
        cases = (
            (
                f"{self.D}가 보유한 {self.C} 주식은 "
                "2024년 2월 3일 기준 몇 주야?",
                "holding_shares",
            ),
            (
                f"{self.D}가 {self.C} 주식을 "
                "2024년 2월 3일 기준 얼마나 들고 있어?",
                None,
            ),
            (
                f"{self.D}의 {self.C} 지분은 "
                "2024년 2월 3일 보고 기준 얼마나 되나?",
                None,
            ),
        )
        for query, metric in cases:
            with self.subTest(query=query):
                result = self.validated(query)
                self.assertEqual(result.plan.task_type, "holding_change")
                self.assertEqual(result.plan.metric, metric)
                self.assertEqual(result.plan.disclosure_route, ("holding",))
                self.assertEqual(
                    result.plan.period.period_type, "holding_reference_date"
                )
                self.assertIs(result.state, QueryState.AMBIGUOUS)
                self.assertFalse(result.retrieval_allowed)
                self.assertEqual(len(result.plan.companies), 2)
                self.assertIsNone(result.plan.company)
                self.assertIsNone(result.plan.reporter)
                self.assertFalse(
                    result.plan.evidence["role_reinterpretation_blocked"]
                )

    def test_an_explicit_comparison_still_resolves(self) -> None:
        result = self.validated("삼성전자와 삼성중공업의 2025년 매출액을 비교해줘")

        self.assertIs(result.state, QueryState.RESOLVED)
        self.assertEqual(len(result.plan.companies), 2)
        self.assertEqual(len(result.plan.corp_codes), 2)

    def test_a_frame_does_not_make_a_comparison_answerable(self) -> None:
        """The firewall recognizes these; it does not start executing them."""

        for query in (
            f"{_and_particle(self.A)} {self.B} 중 어디가 보유 주식수가 더 많아?",
            f"{_and_particle(self.A)} {self.B} 각각 보유 주식수 알려줘",
        ):
            with self.subTest(query=query):
                result = self.validated(query)
                self.assertIsNone(result.plan.comparison)
                self.assertIs(result.state, QueryState.AMBIGUOUS)
                self.assertFalse(result.retrieval_allowed)

    def test_three_companies_stay_ambiguous_without_pair_selection(self) -> None:
        result = self.validated(
            f"{self.A}, {self.B}, {self.C} 중 어디가 보유 주식수가 더 많아?"
        )

        self.assertIs(result.state, QueryState.AMBIGUOUS)
        self.assertEqual(len(result.plan.companies), 3)
        self.assertTrue(result.plan.evidence["role_reinterpretation_blocked"])

    def test_the_guard_reads_the_frame_and_not_the_route(self) -> None:
        for query in (
            f"{_and_particle(self.A)} {self.B} 중 어디가 보유 주식수가 더 많아?",
            f"{_and_particle(self.A)} {self.B} 중 어디가 2023년 매출액이 더 많아?",
            f"{_and_particle(self.A)} {self.B} 중 어디가 유상증자 규모가 더 커?",
        ):
            with self.subTest(query=query):
                result = self.validated(query)
                self.assertEqual(
                    result.plan.evidence["comparison_frame"], "cross_company"
                )
                self.assertTrue(result.plan.evidence["role_reinterpretation_blocked"])


class HoldingCompanyRoleValidationTests(unittest.TestCase):
    A = "가상발행사"
    B = "가상투자사"
    C = "가상제삼사"
    A_CODE = "00000001"
    B_CODE = "00000002"
    C_CODE = "00000003"

    def setUp(self) -> None:
        self.scope = CorpusScope(
            companies={
                self.A: (self.A, self.A_CODE),
                self.B: (self.B, self.B_CODE),
                self.C: (self.C, self.C_CODE),
            },
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())

    def record(
        self,
        issuer: str,
        reporter: str,
        suffix: str,
    ) -> HoldingReportRecord:
        return HoldingReportRecord(
            issuer_corp_code=issuer,
            reporter_key=canonical_reporter_key(reporter),
            raw_reporter=reporter,
            doc_id=f"holding_{suffix}",
            projection_chunk_id=f"holding_{suffix}:projection",
            reference_date="20240203",
            receipt_date="20240205",
            after_shares="100",
            after_ratio="1.00",
        )

    def role_resolver(self, *records: HoldingReportRecord):
        return HoldingCompanyRoleResolver(
            HoldingReportIndex(
                records,
                complete=True,
                correction_finality_available=True,
            )
        )

    def validated(self, query: str, *records: HoldingReportRecord):
        return QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=self.role_resolver(*records),
        ).validate(self.understanding.understand(query))

    def _holding_plan(self, **overrides) -> QueryPlan:
        base = dict(
            query="보유주식수",
            raw_query=f"{self.B} {self.A} 보유주식수",
            companies=(self.A, self.B),
            corp_codes=(self.A_CODE, self.B_CODE),
            task_type="holding_change",
            disclosure_route=("holding",),
        )
        base.update(overrides)
        return QueryPlan(**base)

    def test_comparison_frames_block_role_resolution_directly(self) -> None:
        for frame in ("cross_company", "uncertain"):
            with self.subTest(frame=frame):
                plan = self._holding_plan(evidence={"comparison_frame": frame})
                self.assertFalse(_holding_company_role_resolution_allowed(plan))

    def test_neutral_frame_leaves_role_resolution_available(self) -> None:
        for frame in (None, "same_company"):
            with self.subTest(frame=frame):
                plan = self._holding_plan(evidence={"comparison_frame": frame})
                self.assertTrue(_holding_company_role_resolution_allowed(plan))

    def test_unique_direction_sets_issuer_filter_and_reporter_constraint(self) -> None:
        result = self.validated(
            f"{self.B}가 보유한 {self.A} 주식은 "
            "2024년 2월 3일 기준 몇 주야?",
            self.record(self.A_CODE, self.B, "a_b"),
        )

        self.assertIs(result.state, QueryState.RESOLVED)
        self.assertTrue(result.retrieval_allowed)
        self.assertEqual(result.plan.companies, (self.A,))
        self.assertEqual(result.plan.company, self.A)
        self.assertEqual(result.plan.corp_codes, (self.A_CODE,))
        self.assertEqual(result.plan.corp_code, self.A_CODE)
        self.assertEqual(result.plan.reporter, self.B)
        self.assertEqual(result.plan.backend_filters()["company"], [self.A])
        self.assertEqual(result.plan.backend_filters()["corp_code"], [self.A_CODE])

    def test_reverse_mention_order_resolves_the_same_roles(self) -> None:
        relation = self.record(self.A_CODE, self.B, "a_b")
        first = self.validated(
            f"{self.B}가 보유한 {self.A} 주식은 몇 주야?", relation
        )
        second = self.validated(
            f"{self.A}에서 {self.B}가 보유한 보유주식수 알려줘", relation
        )

        self.assertEqual(first.plan.company, second.plan.company)
        self.assertEqual(first.plan.corp_code, second.plan.corp_code)
        self.assertEqual(first.plan.reporter, second.plan.reporter)

    def test_reference_and_receipt_axes_survive_role_resolution(self) -> None:
        relation = self.record(self.A_CODE, self.B, "a_b")
        reference = self.validated(
            f"{self.B}가 보유한 {self.A} 주식은 "
            "2024년 2월 3일 기준 몇 주야?",
            relation,
        )
        receipt = self.validated(
            f"{self.B}가 보유한 {self.A} 주식을 "
            "2024년 2월 5일 접수된 보고서 기준 몇 주야?",
            relation,
        )

        self.assertEqual(reference.plan.period.period_type, "holding_reference_date")
        self.assertEqual(reference.plan.period.from_date, "2024-02-03")
        self.assertEqual(
            reference.plan.evidence["date_semantics"]["role"],
            "holding_reference",
        )
        self.assertEqual(receipt.plan.period.period_type, "receipt_date")
        self.assertEqual(receipt.plan.period.from_date, "2024-02-05")
        self.assertEqual(receipt.plan.evidence["date_semantics"]["role"], "receipt")

    def test_generic_amount_resolves_without_naming_a_metric(self) -> None:
        """Roles resolve here, and the unnamed unit no longer blocks.

        This was the metric=None blocker recorded during role resolution.
        B.3 answered it by requesting the pair the filing publishes together
        rather than guessing a unit, so the plan still carries no metric while
        the slot is satisfied.  Everything the earlier test pinned about role
        resolution is pinned here unchanged.
        """

        result = self.validated(
            f"{self.B}가 {self.A} 주식을 "
            "2024년 2월 3일 기준 얼마나 들고 있어?",
            self.record(self.A_CODE, self.B, "a_b"),
        )

        self.assertEqual(result.plan.company, self.A)
        self.assertEqual(result.plan.reporter, self.B)
        self.assertIsNone(result.plan.metric, "the plan must not be relabelled")
        self.assertIs(result.state, QueryState.RESOLVED)
        self.assertTrue(result.retrieval_allowed)
        self.assertEqual(result.slots["metric"].status.value, "resolved")

    def test_metric_stays_missing_when_no_unit_and_no_stative_request(self) -> None:
        """The slot is only satisfied by the current-state pair, never by
        mere non-emptiness of the requested-field tuple."""

        result = self.validated(
            f"{self.B}의 {self.A} 지분은 얼마나 매각됐어?",
            self.record(self.A_CODE, self.B, "a_b"),
        )

        self.assertIsNone(result.plan.metric)
        self.assertIs(result.state, QueryState.INCOMPLETE)
        self.assertFalse(result.retrieval_allowed)
        self.assertEqual(result.slots["metric"].status.value, "missing")

    def test_zero_and_bidirectional_relations_remain_ambiguous(self) -> None:
        query = f"{self.B}가 보유한 {self.A} 주식은 몇 주야?"
        zero = self.validated(query)
        bidirectional = self.validated(
            query,
            self.record(self.A_CODE, self.B, "a_b"),
            self.record(self.B_CODE, self.A, "b_a"),
        )

        for result in (zero, bidirectional):
            self.assertIs(result.state, QueryState.AMBIGUOUS)
            self.assertFalse(result.retrieval_allowed)
            self.assertEqual(len(result.plan.companies), 2)
            self.assertIsNone(result.plan.company)
            self.assertIsNone(result.plan.reporter)

    def test_unknown_company_fails_closed_before_relation_lookup(self) -> None:
        plan = QueryPlan(
            query="보유주식수",
            raw_query="가상투자사가 미등록회사 주식을 몇 주 보유했어?",
            companies=(self.B, "미등록회사"),
            task_type="holding_change",
            metric="holding_shares",
            disclosure_route=("holding",),
            evidence={"operation": "lookup_holding", "comparison_frame": None},
        )
        result = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=self.role_resolver(
                self.record(self.A_CODE, self.B, "a_b")
            ),
        ).validate(plan)

        self.assertIs(result.state, QueryState.OUT_OF_SCOPE)
        self.assertFalse(result.retrieval_allowed)
        self.assertEqual(result.issues, ("company_out_of_corpus",))

    def test_safety_predicates_do_not_invoke_the_role_resolver(self) -> None:
        class RecordingResolver:
            def __init__(self) -> None:
                self.calls = 0
                self.query_grounded_calls = 0

            def resolve(self, *args):
                self.calls += 1
                raise AssertionError(f"role resolver must not run: {args!r}")

            def resolve_query_grounded(self, *args):
                # A single-company holding question may ask the corpus which of
                # this issuer's holders it names; declining is the answer here,
                # and no reporter may be produced from it.
                self.query_grounded_calls += 1
                return SimpleNamespace(resolved=False)

        cases = (
            f"{self.A}와 {self.B} 보유비율 비교",
            f"{self.A}와 {self.B} 중 누가 보유 주식수가 더 많아?",
            f"{self.B}가 {self.A} 주식을 더 많이 취득한 시점은 언제야?",
            f"{self.A}, {self.B}, {self.C} 중 보유비율이 가장 높아?",
            f"{self.A} 보유주식수는?",
            f"{self.A} {self.B} 사업보고서 내용 알려줘",
            f"{self.A} {self.B} 풋옵션 행사 주식 취득일과 취득 수량",
        )
        for query in cases:
            with self.subTest(query=query):
                recording = RecordingResolver()
                result = QueryValidator(
                    corpus_scope=self.scope,
                    holding_company_role_resolver=recording,
                ).validate(self.understanding.understand(query))
                self.assertEqual(recording.calls, 0)
                self.assertIsNone(result.plan.reporter)
                self.assertIsNotNone(result.state)


if __name__ == "__main__":
    unittest.main()


class DirectedFilerBindingValidationTests(unittest.TestCase):
    """Binding a named acquirer to a holder the corpus proves it is.

    Query understanding can show that a surface stood in the actor slot; only
    the corpus can say whether that surface is one of this issuer's holders.
    So this stage asks exactly that one question, and every answer other than
    a unique holder leaves the plan exactly as it arrived.
    """

    ISSUER = "가상발행사"
    ISSUER_CODE = "00000001"
    HOLDER = "가상지주"
    PENSION = "가상연금"
    PENSION_AGENCY = "가상연금공단"
    PENSION_FUND = "가상연금기금"
    OTHER_HOLDER = "가상제삼사"

    def setUp(self) -> None:
        # An issuer-scoped universe: the acquirer is not in it, which is the
        # whole reason a corpus-backed filer lookup has to exist.
        self.scope = CorpusScope(
            companies={self.ISSUER: (self.ISSUER, self.ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())

    def record(self, reporter: str, suffix: str) -> HoldingReportRecord:
        return HoldingReportRecord(
            issuer_corp_code=self.ISSUER_CODE,
            reporter_key=canonical_reporter_key(reporter),
            raw_reporter=reporter,
            doc_id=f"holding_{suffix}",
            projection_chunk_id=f"holding_{suffix}:projection",
            reference_date="20240203",
            receipt_date="20240205",
            after_shares="100",
            after_ratio="1.00",
        )

    def validator(self, *records) -> QueryValidator:
        return QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=HoldingCompanyRoleResolver(
                HoldingReportIndex(
                    records, complete=True, correction_finality_available=True
                )
            ),
        )

    def directed(self) -> str:
        return (
            f"{self.HOLDER}가 {self.ISSUER} 주식을 "
            "취득할 때의 취득단가는 얼마야?"
        )

    def bound_plan(self, query: str, *records) -> QueryPlan:
        return self.validator(*records).validate(
            self.understanding.understand(query)
        ).plan

    def holding_plan(self, **overrides) -> QueryPlan:
        base = dict(
            query="취득단가",
            raw_query=self.directed(),
            companies=(self.ISSUER,),
            corp_codes=(self.ISSUER_CODE,),
            task_type="holding_change",
            disclosure_route=("holding",),
            metric="acquisition_unit_price",
            evidence={
                HOLDING_ACTOR_CANDIDATE_KEY: {
                    "surface": self.HOLDER,
                    "reporter_key": canonical_reporter_key(self.HOLDER),
                    "source": ACTOR_SOURCE_DIRECTED_HOLDER,
                    "resolved": False,
                }
            },
        )
        base.update(overrides)
        return QueryPlan(**base)

    def test_named_filer_binds_with_filer_path_provenance(self) -> None:
        plan = self.bound_plan(self.directed(), self.record(self.HOLDER, "a1"))

        self.assertEqual(plan.reporter, self.HOLDER)
        marker = plan.evidence[ROLE_PROVENANCE_KEY]
        self.assertTrue(marker["resolved"])
        # The path says which corpus question proved it, without either party
        # having to be restated inside the provenance.
        self.assertEqual(marker["path"], ROLE_PATH_FILER)
        self.assertNotIn(self.HOLDER, str(marker))
        self.assertNotIn(self.ISSUER, str(marker))

    def test_unknown_and_ambiguous_filers_leave_plan_unbound(self) -> None:
        cases = (
            ("unknown", (self.record(self.OTHER_HOLDER, "a1"),), self.HOLDER),
            (
                "ambiguous",
                (
                    self.record(self.PENSION_AGENCY, "a1"),
                    self.record(self.PENSION_FUND, "a2"),
                ),
                self.PENSION,
            ),
        )
        for label, records, surface in cases:
            with self.subTest(filer=label):
                query = (
                    f"{surface}가 {self.ISSUER} 주식을 "
                    "취득할 때의 취득단가는 얼마야?"
                )
                plan = self.bound_plan(query, *records)

                # Inventing a holder would answer with someone else's position.
                self.assertIsNone(plan.reporter)
                self.assertNotIn(ROLE_PROVENANCE_KEY, plan.evidence)

    def test_asker_supplied_reporter_is_never_overwritten(self) -> None:
        plan = self.holding_plan(reporter=self.OTHER_HOLDER)

        bound = self.validator(self.record(self.HOLDER, "a1"))._bind_named_filer(plan)

        self.assertEqual(bound.reporter, self.OTHER_HOLDER)
        self.assertNotIn(ROLE_PROVENANCE_KEY, bound.evidence)

    def test_disclosure_lookup_plan_cannot_be_promoted_by_filer_binding(self) -> None:
        plan = self.holding_plan(
            task_type="disclosure_lookup", disclosure_route=(), metric=None
        )

        bound = self.validator(self.record(self.HOLDER, "a1"))._bind_named_filer(plan)

        # Binding is binding.  It may not decide what kind of question this is.
        self.assertEqual(bound.task_type, "disclosure_lookup")
        self.assertEqual(bound.disclosure_route, ())
        self.assertIsNone(bound.reporter)
        self.assertNotIn(ROLE_PROVENANCE_KEY, bound.evidence)

    def test_filer_binding_requires_holding_route_and_single_company(self) -> None:
        self.assertTrue(_holding_filer_resolution_allowed(self.holding_plan()))
        for label, overrides in (
            ("wrong task type", dict(task_type="disclosure_lookup")),
            ("no holding route", dict(disclosure_route=())),
            ("event type present", dict(event_type="supply_contract")),
            ("reporter already known", dict(reporter=self.HOLDER)),
            (
                "two companies",
                dict(
                    companies=(self.ISSUER, self.OTHER_HOLDER),
                    corp_codes=(self.ISSUER_CODE, "00000003"),
                ),
            ),
            ("no actor candidate", dict(evidence={})),
        ):
            with self.subTest(gate=label):
                self.assertFalse(
                    _holding_filer_resolution_allowed(self.holding_plan(**overrides))
                )

    def test_comparison_input_cannot_receive_filer_binding(self) -> None:
        query = (
            f"{self.HOLDER}와 다른 회사의 {self.ISSUER} 주식 "
            "취득단가를 비교해줘"
        )
        understood = self.understanding.understand(query)
        plan = self.bound_plan(query, self.record(self.HOLDER, "a1"))

        # The comparison never reaches the holding lane, so there is no single
        # holder for this binding to be about.
        self.assertNotEqual(understood.task_type, "holding_change")
        self.assertIsNone(understood.evidence[HOLDING_ACTOR_CANDIDATE_KEY])
        self.assertFalse(_holding_filer_resolution_allowed(understood))
        self.assertIsNone(plan.reporter)
        self.assertNotIn(ROLE_PROVENANCE_KEY, plan.evidence)
