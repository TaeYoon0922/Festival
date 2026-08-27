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


if __name__ == "__main__":
    unittest.main()
