"""P0-C Step 3: the Evidence Slot planner.

The planner is a compiler: question -> MultiDocumentPlan.  It executes nothing,
so these tests assert the *shape* of the plan, never a document count.

Two properties matter more than the rest:

* 체결 vs 공시.  43.9% of this corpus's supply contracts were disclosed in a
  different year than they were signed, so routing a "체결한" question onto the
  receipt axis silently changes the answer.  Dedicated tests pin both.
* Non-engagement.  All 60 real Gold60 questions must decline, including the
  three that a naive keyword gate flagged (0% precision).
"""

from __future__ import annotations

import csv
import json
import unicodedata
import unittest
from pathlib import Path

from app.reasoning.multi_document_plan import (
    PLAN_NOT_APPLICABLE,
    REASON_MIXED_DATE_BASIS,
    REASON_NO_CORP_CODE,
    REASON_NO_DATE_RANGE,
    REASON_NO_FAMILY,
    REASON_NO_SET_INTENT,
    REASON_UNRESOLVED_DATE_BASIS,
    REASON_UNSUPPORTED_CALCULATION,
    REASON_UNSUPPORTED_TRUST_CONTRACT_BASIS,
    MultiDocumentIntent,
    SlotType,
)
from app.reasoning.multi_document_planner import (
    FAMILY_BARE_CONTRACT_FALLBACK,
    FAMILY_EXPLICIT_EVENT,
    MultiDocumentPlanner,
)
from app.reasoning.query_plan import DateBasis
from app.reasoning.query_understanding import QueryUnderstanding, _date_basis_from_query


REPO = Path(__file__).resolve().parents[1]
GOLD60 = (
    REPO
    / "reports/evaluation/gold60/2026-08-21-agent-90pct/gold60_agent_questions.jsonl"
)
UNIVERSE = REPO / "data/corpus/universe.csv"


def _company_resolver():
    """Resolve against the frozen universe table -- no database."""

    names: dict[str, str] = {}
    if UNIVERSE.exists():
        with UNIVERSE.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for key in ("corp_name", "listed_name"):
                    value = (row.get(key) or "").replace(" ", "")
                    if value:
                        names[unicodedata.normalize("NFC", value)] = row["corp_code"]
    else:  # pragma: no cover - corpus not checked out
        names = {"삼성중공업": "00126478", "OCI홀딩스": "00148896"}

    def resolve(query: str):
        text = unicodedata.normalize("NFC", (query or "").replace(" ", ""))
        best = None
        for name, code in names.items():
            if name in text and (best is None or len(name) > len(best[0])):
                best = (name, code)
        return {"corp_code": best[1], "corp_name": best[0]} if best else None

    return resolve


class _PlannerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.understanding = QueryUnderstanding(company_resolver=_company_resolver())
        cls.planner = MultiDocumentPlanner()

    def plan_for(self, question: str):
        return self.planner.plan(question, self.understanding.understand(question))


# --------------------------------------------------------------- DateBasis


class DateBasisDetectionTests(unittest.TestCase):
    def test_contract_date_markers(self) -> None:
        for question in (
            "삼성중공업이 2025년에 체결한 공급계약",
            "2025년에 체결된 계약",
            "2024년에 수주한 계약",
            "2025년에 계약을 맺은 건",
        ):
            self.assertEqual(_date_basis_from_query(question), "contract_date", question)

    def test_receipt_date_markers(self) -> None:
        for question in (
            "2025년에 공시한 공급계약",
            "2025년에 공시된 계약",
            "2025년에 접수된 공시",
            "2025년에 제출한 보고서",
        ):
            self.assertEqual(_date_basis_from_query(question), "receipt_date", question)

    def test_period_start_markers(self) -> None:
        for question in (
            "2024년에 시작한 자기주식취득신탁계약",
            "2024년에 개시된 신탁계약",
        ):
            self.assertEqual(_date_basis_from_query(question), "period_start", question)

    def test_bare_year_stays_unspecified(self) -> None:
        """Never promoted: the two bases select different documents."""

        for question in ("삼성중공업 2025년 공급계약", "2025년 주요 계약"):
            self.assertEqual(_date_basis_from_query(question), "unspecified", question)

    def test_two_bases_on_two_periods_is_mixed(self) -> None:
        self.assertEqual(
            _date_basis_from_query("2025년에 체결하고 2026년에 공시된 계약"), "mixed"
        )

    def test_unrelated_marker_does_not_pollute_the_basis(self) -> None:
        """"공시 이후 해지" is about the termination, not about the year."""

        self.assertEqual(
            _date_basis_from_query(
                "삼성중공업이 2025년에 체결한 계약 중 공시 이후 해지된 것이 있는가?"
            ),
            "contract_date",
        )

    def test_no_period_expression_is_unspecified(self) -> None:
        self.assertEqual(_date_basis_from_query("공급계약 해지금액은?"), "unspecified")

    def test_query_plan_carries_the_basis(self) -> None:
        understanding = QueryUnderstanding(company_resolver=_company_resolver())
        plan = understanding.understand("삼성중공업이 2025년에 체결한 공급계약")
        self.assertIs(plan.date_basis, DateBasis.CONTRACT_DATE)
        self.assertEqual(plan.to_dict()["date_basis"], "contract_date")

    def test_basis_does_not_reach_backend_filters(self) -> None:
        """Additive metadata only: retrieval must be byte-identical."""

        understanding = QueryUnderstanding(company_resolver=_company_resolver())
        signed = understanding.understand("삼성중공업이 2025년에 체결한 공급계약")
        bare = understanding.understand("삼성중공업 2025년 공급계약")
        self.assertNotEqual(signed.date_basis, bare.date_basis)
        self.assertEqual(signed.backend_filters(), bare.backend_filters())


# ------------------------------------------------------- official-style plans


class OfficialStylePlanTests(_PlannerCase):
    """§21 P1-P6, the shapes the official task asks for."""

    def test_p1_enumeration_supply_tier1(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 주요 공급계약은 모두 몇 건인가?")
        self.assertTrue(plan.applied)
        self.assertEqual(plan.plan_type, MultiDocumentIntent.ENUMERATION.value)
        self.assertEqual(len(plan.slots), 1)
        slot = plan.slots[0]
        self.assertIs(slot.slot_type, SlotType.ENUMERATE_EVENTS)
        self.assertEqual(slot.event_family, "supply_contract")
        self.assertEqual(slot.member_role, "contract")
        self.assertEqual(slot.date_field, "opened_at")
        self.assertEqual((slot.date_from, slot.date_to), ("2025-01-01", "2026-01-01"))
        self.assertEqual(slot.corp_code, "00126478")

    def test_p2_enumeration_plus_event(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?")
        self.assertEqual(plan.plan_type, MultiDocumentIntent.ENUMERATION_PLUS_EVENT.value)
        self.assertEqual(len(plan.slots), 2)
        enumerate_slot, lifecycle = plan.slots
        self.assertIs(enumerate_slot.slot_type, SlotType.ENUMERATE_EVENTS)
        self.assertEqual(enumerate_slot.date_field, "opened_at")
        self.assertIs(lifecycle.slot_type, SlotType.EVENT_STATE)
        self.assertEqual(lifecycle.depends_on, (enumerate_slot.slot_id,))

    def test_p3_receipt_axis_uses_tier2(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 공시한 공급계약은 몇 건인가?")
        self.assertEqual(plan.plan_type, MultiDocumentIntent.ENUMERATION.value)
        slot = plan.slots[0]
        self.assertIs(slot.slot_type, SlotType.ENUMERATE_DOCUMENTS)
        self.assertEqual(slot.date_field, "rcept_dt")
        self.assertEqual(slot.doc_group, "exchange")
        self.assertEqual(slot.doc_subtype, "단일판매공급계약체결")
        self.assertEqual((slot.date_from, slot.date_to), ("2025-01-01", "2026-01-01"))

    def test_p4_trust_start_enumeration_plus_event(self) -> None:
        plan = self.plan_for("OCI홀딩스가 2024년에 시작한 자기주식취득신탁계약 중 해지된 것이 있는가?")
        self.assertEqual(plan.plan_type, MultiDocumentIntent.ENUMERATION_PLUS_EVENT.value)
        enumerate_slot = plan.slots[0]
        self.assertEqual(enumerate_slot.event_family, "treasury_trust_contract")
        self.assertIs(enumerate_slot.slot_type, SlotType.ENUMERATE_EVENTS)
        self.assertEqual((enumerate_slot.date_from, enumerate_slot.date_to),
                         ("2024-01-01", "2025-01-01"))

    def test_p5_bare_year_declines(self) -> None:
        plan = self.plan_for("삼성중공업 2025년 공급계약")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.plan_type, PLAN_NOT_APPLICABLE)

    def test_p6_single_field_lookup_declines(self) -> None:
        plan = self.plan_for("삼성중공업 공급계약 해지금액은?")
        self.assertFalse(plan.applied)


class DateAxisInvariantTests(_PlannerCase):
    """The 43.9% drift makes this the most consequential invariant in P0-C."""

    def test_signed_question_never_routes_to_the_receipt_axis(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 공급계약은 모두 몇 건인가?")
        for slot in plan.slots:
            self.assertNotEqual(slot.date_field, "rcept_dt")
            if slot.slot_type is SlotType.ENUMERATE_EVENTS:
                self.assertEqual(slot.date_field, "opened_at")

    def test_disclosed_question_never_routes_to_the_event_axis(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 공시한 공급계약은 모두 몇 건인가?")
        for slot in plan.slots:
            self.assertNotEqual(slot.date_field, "opened_at")
            self.assertIsNot(slot.slot_type, SlotType.ENUMERATE_EVENTS)

    def test_ranges_are_half_open(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 공급계약은 모두 몇 건인가?")
        slot = plan.slots[0]
        self.assertEqual(slot.date_to, "2026-01-01")
        self.assertNotEqual(slot.date_to, "2025-12-31")


class TrustBasisTests(_PlannerCase):
    def test_trust_signed_basis_is_declined(self) -> None:
        """P0-B derives trust opened_at from 계약기간 시작일, not from
        계약체결 예정일자.  The two coincide in all 43 measured filings, but the
        stored provenance is period_start, so answering "체결한" from it would
        name the wrong field."""

        plan = self.plan_for("OCI홀딩스가 2024년에 체결한 자기주식취득신탁계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_UNSUPPORTED_TRUST_CONTRACT_BASIS)

    def test_trust_start_basis_is_supported(self) -> None:
        plan = self.plan_for("OCI홀딩스가 2024년에 시작한 자기주식취득신탁계약은 모두 몇 건인가?")
        self.assertTrue(plan.applied)
        self.assertEqual(plan.slots[0].event_family, "treasury_trust_contract")

    def test_trust_receipt_basis_uses_tier2(self) -> None:
        plan = self.plan_for("OCI홀딩스가 2024년에 공시한 자기주식취득신탁계약은 모두 몇 건인가?")
        self.assertTrue(plan.applied)
        slot = plan.slots[0]
        self.assertIs(slot.slot_type, SlotType.ENUMERATE_DOCUMENTS)
        self.assertEqual(slot.doc_group, "major")
        self.assertEqual(slot.date_field, "rcept_dt")


class EngagementGateTests(_PlannerCase):
    def test_unresolved_company_declines(self) -> None:
        plan = self.plan_for("2025년에 체결한 공급계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_CORP_CODE)

    def test_unsupported_family_declines(self) -> None:
        plan = self.plan_for("삼성전자가 2025년에 공시한 반기보고서는 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_FAMILY)

    def test_unbounded_range_declines(self) -> None:
        plan = self.plan_for("삼성중공업이 최근 체결한 공급계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertIn(plan.stop_reason, {REASON_NO_DATE_RANGE, REASON_UNRESOLVED_DATE_BASIS})

    def test_unspecified_basis_declines(self) -> None:
        plan = self.plan_for("삼성중공업의 2025년 공급계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_UNRESOLVED_DATE_BASIS)

    def test_mixed_basis_declines(self) -> None:
        plan = self.plan_for(
            "삼성중공업이 2025년에 체결하고 2026년에 공시된 공급계약은 모두 몇 건인가?"
        )
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_MIXED_DATE_BASIS)

    def test_no_set_intent_declines(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 공급계약의 계약상대방은?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_SET_INTENT)


class UnsupportedCalculationTests(_PlannerCase):
    """§22 -- recognized so the planner declines explicitly, never enumerates
    and calls an arithmetic question complete."""

    def test_sum_engages_with_bare_contract_fallback(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 계약의 총액은 얼마인가?")
        self.assertTrue(plan.applied)
        self.assertIn("sum", plan.aggregate_ops)

    def test_average_engages_with_supply_contract(self) -> None:
        plan = self.plan_for(
            "삼성중공업 2025년에 공시한 공급계약 건당 평균 계약금액은?"
        )
        self.assertTrue(plan.applied)
        self.assertIn("average", plan.aggregate_ops)

    def test_facility_recent_pair_engages_with_corpus_bounds(self) -> None:
        from app.reasoning.query_plan import QueryPeriod, QueryPlan

        question = (
            "가상Corp 최근 신규시설투자 공시의 투자금액은 "
            "직전 시설투자 공시 대비 커졌어?"
        )
        query_plan = QueryPlan(
            query=question,
            raw_query=question,
            companies=("가상Corp",),
            corp_codes=("00126478",),
            period=QueryPeriod(period_type="latest_event"),
            task_type="corporate_event",
            event_type="facility_investment",
            date_basis=DateBasis.RECEIPT_DATE,
            evidence={
                "exchange_recent_pair": {"limit": 2, "field": "investment_amount"},
                "corpus_receipt_from": "2020-01-01",
                "corpus_receipt_to": "2025-12-31",
            },
        )
        plan = MultiDocumentPlanner().plan(question, query_plan)
        self.assertTrue(plan.applied, plan.stop_reason)
        self.assertEqual(plan.recent_pair_limit, 2)
        self.assertEqual(plan.slots[0].event_family, "facility_investment")

    def test_cross_company_comparison_declines(self) -> None:
        plan = self.plan_for(
            "삼성중공업과 한화오션 중 2025년 설비투자 규모가 더 큰 기업은?"
        )
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_UNSUPPORTED_CALCULATION)

    def test_growth_rate_declines(self) -> None:
        plan = self.plan_for("삼성중공업의 2024년 대비 2025년 계약금액 증가율은?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_UNSUPPORTED_CALCULATION)


class Gold60NonEngagementTests(_PlannerCase):
    """§20 -- every real local question must decline."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.rows = (
            [json.loads(line) for line in GOLD60.read_text(encoding="utf-8").splitlines() if line.strip()]
            if GOLD60.exists()
            else []
        )

    def test_all_sixty_decline(self) -> None:
        if not self.rows:
            self.skipTest("gold60 artifact not present")
        self.assertEqual(len(self.rows), 60)
        engaged = [
            (row["question_id"], row["question"])
            for row in self.rows
            if self.plan_for(row["question"]).applied
        ]
        self.assertEqual(engaged, [], f"P0-C engaged on Gold60: {engaged}")

    def test_historical_keyword_false_positives_decline(self) -> None:
        """평균가동률 / 권면총액 / 증감 수량 -- field labels, not set intent."""

        if not self.rows:
            self.skipTest("gold60 artifact not present")
        by_id = {row["question_id"]: row["question"] for row in self.rows}
        for question_id in ("P08", "M05", "H02"):
            plan = self.plan_for(by_id[question_id])
            self.assertFalse(plan.applied, f"{question_id}: {by_id[question_id]}")



class BareContractFallbackTests(_PlannerCase):
    """Step 3.5 -- the official reference question says "주요 계약", not "공급계약".

    The frozen ``_EVENTS`` vocabulary deliberately does not read a bare 계약 as a
    supply contract (that word also appears in periodic and correction
    questions), and widening it would move Gold60 routing.  So the fallback is
    plan-local: it lives in the planner, never writes back to
    ``query_plan.event_type``, and fires only when every surrounding constraint
    already holds.
    """

    def test_b1_official_reference_question_engages(self) -> None:
        question = "삼성중공업이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"
        plan = self.plan_for(question)
        self.assertTrue(plan.applied)
        self.assertEqual(plan.plan_type, MultiDocumentIntent.ENUMERATION_PLUS_EVENT.value)
        self.assertEqual(len(plan.slots), 2)
        enumerate_slot, lifecycle = plan.slots
        self.assertEqual(enumerate_slot.event_family, "supply_contract")
        self.assertEqual(enumerate_slot.date_field, "opened_at")
        self.assertEqual(
            (enumerate_slot.date_from, enumerate_slot.date_to),
            ("2025-01-01", "2026-01-01"),
        )
        self.assertIs(lifecycle.slot_type, SlotType.EVENT_STATE)
        # Resolved through the frozen vocabulary, which already carries "계약이후"
        # as a supply alias -- so the fallback is not even reached here.
        self.assertEqual(plan.family_resolution, FAMILY_EXPLICIT_EVENT)

    def test_b1_variant_without_an_explicit_alias_uses_the_fallback(self) -> None:
        """Same official shape, phrased so no frozen alias matches."""

        plan = self.plan_for("삼성중공업이 2025년에 체결한 주요 계약은 모두 몇 건인가?")
        self.assertTrue(plan.applied)
        self.assertEqual(plan.family_resolution, FAMILY_BARE_CONTRACT_FALLBACK)
        self.assertEqual(plan.slots[0].event_family, "supply_contract")
        self.assertEqual(plan.slots[0].date_field, "opened_at")

    def test_b2_bare_contract_enumeration(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 계약은 모두 몇 건인가?")
        self.assertTrue(plan.applied)
        self.assertEqual(plan.plan_type, MultiDocumentIntent.ENUMERATION.value)
        self.assertEqual(plan.family_resolution, FAMILY_BARE_CONTRACT_FALLBACK)
        self.assertEqual(plan.slots[0].event_family, "supply_contract")

    def test_b3_explicit_supply_family_takes_priority(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 공급계약은 모두 몇 건인가?")
        self.assertTrue(plan.applied)
        self.assertEqual(plan.family_resolution, FAMILY_EXPLICIT_EVENT)

    def test_fallback_never_rewrites_the_query_plan(self) -> None:
        """The frozen event vocabulary must see no change from P0-C."""

        question = "삼성중공업이 2025년에 체결한 계약은 모두 몇 건인가?"
        query_plan = self.understanding.understand(question)
        before = query_plan.event_type
        plan = self.planner.plan(question, query_plan)
        self.assertEqual(plan.family_resolution, FAMILY_BARE_CONTRACT_FALLBACK)
        self.assertIsNone(before)
        self.assertIsNone(query_plan.event_type)

    # ------------------------------------------------------------- negatives

    def test_n1_bare_contract_without_set_intent_declines(self) -> None:
        plan = self.plan_for("삼성중공업 2025년 주요 계약")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_SET_INTENT)

    def test_n2_receipt_basis_never_falls_back(self) -> None:
        """opened_at and rcept_dt disagree for 43.9% of supply contracts."""

        plan = self.plan_for("삼성중공업이 2025년에 공시한 주요 계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_FAMILY)

    def test_unspecified_basis_never_falls_back(self) -> None:
        plan = self.plan_for("삼성중공업의 2025년 주요 계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_UNRESOLVED_DATE_BASIS)

    def test_n3_trust_never_falls_back_to_supply(self) -> None:
        for question in (
            "삼성중공업이 2024년에 시작한 신탁계약은 모두 몇 건인가?",
            "삼성중공업이 2025년에 체결한 신탁계약은 모두 몇 건인가?",
        ):
            plan = self.plan_for(question)
            families = {slot.event_family for slot in plan.slots}
            self.assertNotIn("supply_contract", families, question)

    def test_n4_financing_contract_never_falls_back(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 자금조달 계약을 모두 알려줘")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_FAMILY)

    def test_n5_facility_investment_never_falls_back(self) -> None:
        plan = self.plan_for("삼성중공업이 2025년에 체결한 설비투자 계약을 모두 알려줘")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_FAMILY)

    def test_n6_single_field_lookup_declines(self) -> None:
        plan = self.plan_for("삼성중공업 계약금액은?")
        self.assertFalse(plan.applied)

    def test_other_incompatible_families_never_fall_back(self) -> None:
        """Grounded in the audited corpus taxonomy: the 계약 filings here are
        supply, trust, and a tail of 라이선스 / 공동연구 / 주주간 agreements."""

        for marker in ("라이선스", "공동연구", "주주간", "전환사채", "유상증자", "합병"):
            question = f"삼성중공업이 2025년에 체결한 {marker} 계약은 모두 몇 건인가?"
            plan = self.plan_for(question)
            self.assertFalse(plan.applied, question)

    def test_company_is_still_required(self) -> None:
        plan = self.plan_for("2025년에 체결한 주요 계약은 모두 몇 건인가?")
        self.assertFalse(plan.applied)
        self.assertEqual(plan.stop_reason, REASON_NO_CORP_CODE)


class PlannerPurityTests(unittest.TestCase):
    """§23 -- Step 3 is a plan compiler; it must not reach a database."""

    def test_planner_module_imports_no_database(self) -> None:
        source = (REPO / "app/reasoning/multi_document_planner.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "psycopg",
            "PostgresBackend",
            "PostgresCorporateEventRepository",
            "PostgresCorrectionRepository",
            "SELECT",
            "app.retrieval",
        ):
            self.assertNotIn(forbidden, source, forbidden)

    def test_planner_fills_no_execution_results(self) -> None:
        """expected_ids/found_ids are Step 4 output, not Step 3."""

        understanding = QueryUnderstanding(company_resolver=_company_resolver())
        planner = MultiDocumentPlanner()
        question = "삼성중공업이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"
        plan = planner.plan(question, understanding.understand(question))
        for slot in plan.slots:
            self.assertEqual(slot.expected_ids, ())
            self.assertEqual(slot.found_ids, ())
            self.assertEqual(slot.unresolved_ids, ())

    def test_event_vocabulary_stays_frozen(self) -> None:
        """The bare-contract fallback must live in the planner, not in _EVENTS.

        Widening the shared vocabulary would change doc_group routing for every
        question that mentions a contract, including the frozen Gold60 path.
        """

        source = (REPO / "app/reasoning/query_understanding.py").read_text(
            encoding="utf-8"
        )
        events = source[source.index("_EVENTS:") : source.index("_EVENT_DOC_SUBTYPES")]
        self.assertNotIn("주요 계약", events)
        self.assertNotIn('"계약"', events)
        # The supply aliases are exactly the ones P0-B was built against.
        for alias in ("단일판매", "공급계약", "수주계약", "계약체결"):
            self.assertIn(alias, events)

    def test_fallback_lives_only_in_the_planner(self) -> None:
        planner = (REPO / "app/reasoning/multi_document_planner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_BARE_CONTRACT_MARKERS", planner)
        understanding = (REPO / "app/reasoning/query_understanding.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_BARE_CONTRACT_MARKERS", understanding)
        self.assertNotIn("bare_contract_fallback", understanding)

    def test_planning_is_deterministic(self) -> None:
        understanding = QueryUnderstanding(company_resolver=_company_resolver())
        planner = MultiDocumentPlanner()
        question = "삼성중공업이 2025년에 체결한 주요 공급계약은 모두 몇 건인가?"
        first = planner.plan(question, understanding.understand(question)).to_dict()
        second = planner.plan(question, understanding.understand(question)).to_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
