from __future__ import annotations

import csv
import json
import unittest
from dataclasses import replace
from pathlib import Path

from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import (
    CorpusScope,
    QuerySlotSource,
    QueryState,
    QueryValidator,
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

if __name__ == "__main__":
    unittest.main()
