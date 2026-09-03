"""Two metric names this corpus overloads, decided by the phrase around them.

Independent Eval v2 failed C059 and C086 the same way: a generic single token
inside a longer domain phrase decided the route, and the longer phrase never got
to speak.

C059 asks a supply contract for its own ``매출액대비(%)`` cell.  The ``매출액``
inside ``매출액 대비`` read as the periodic revenue line, so a contract question
was sent to the periodic fact resolver, which answered
``explicit_period_evidence_unmatched`` about a report nobody asked for.

C086 asks how many affiliates a 기업집단 has, listed and unlisted together.  The
``상장`` inside ``상장과 비상장`` read as listing history, so the question came
back with ``confirmed_fields=["listing_history"]`` -- a listing date, for a
question about a count.

Neither is a retrieval failure: both gold documents rank first.  What follows
fixes the intent, and the ordering it fixes it with is deterministic -- a strong
multi-token domain phrase outranks the generic single-token metric inside it.
"""

from __future__ import annotations

import unittest

from app.agent.task_router import route_task
from app.reasoning.affiliate_group import affiliate_counts
from app.reasoning.corporate_event_field_evidence import (
    CONTRACT_AMOUNT,
    SALES_RATIO,
    requested_corporate_fields,
)
from app.reasoning.field_evidence import FieldReason, FieldStatus
from app.reasoning.metric_disambiguation import (
    AFFILIATE_COUNT,
    CONTRACT_SALES_RATIO,
    affiliate_count_intent,
    contract_sales_ratio_intent,
)
from app.reasoning.query_understanding import QueryUnderstanding
from tests.test_explicit_unavailability import (
    CORP,
    RECEIPT,
    _corporate_evidence,
    _row,
    _state,
)


#: IEV2-C059, verbatim.  gold doc ``exchange_20230117800134`` ranks first.
C059 = (
    "HD현대중공업이 2023년 1월 17일 공시한 LNGC 3척 계약의 "
    "매출액 대비 비율은 몇 퍼센트인가?"
)
#: IEV2-C086, verbatim.  gold doc ``periodic_20240312000629``, table ``t0645``.
C086 = "세아베스틸지주가 속한 기업집단의 계열회사는 상장과 비상장을 합쳐 몇 개인가?"

_ALIASES = {
    "HD현대중공업": {"HD현대중공업"},
    "고려아연": {"고려아연"},
    "세아베스틸지주": {"세아베스틸지주"},
    "시프트업": {"시프트업"},
}


def _plan(question: str):
    return QueryUnderstanding(_ALIASES).understand(question)


# --------------------------------------------------------- C059: 매출액 대비


class ContractSalesRatioPhraseTests(unittest.TestCase):
    """``매출액 대비`` is a contract field only where a contract is named."""

    def test_contract_context_licenses_the_ratio_phrase(self) -> None:
        for question in (
            "단일판매ㆍ공급계약의 매출액대비(%)는 얼마인가",
            "수주 계약의 매출액 대비 비율은?",
            "체결계약의 매출액대비 비율",
            C059,
        ):
            with self.subTest(question=question):
                intent = contract_sales_ratio_intent(question)
                self.assertIsNotNone(intent, question)
                self.assertEqual(intent.metric, CONTRACT_SALES_RATIO)
                self.assertEqual(intent.phrase, "매출액대비")

    def test_the_phrase_alone_is_not_a_contract_question(self) -> None:
        for question in (
            "고려아연 2024년 매출액 대비 영업이익 비율은?",
            "매출액 대비 연구개발비 비중",
        ):
            with self.subTest(question=question):
                self.assertIsNone(contract_sales_ratio_intent(question))

    def test_a_contract_alone_is_not_a_ratio_question(self) -> None:
        for question in (
            "고려아연 단일판매 공급계약 금액",
            "계약 상대방은 누구인가요?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(contract_sales_ratio_intent(question))


class ContractSalesRatioRoutingTests(unittest.TestCase):
    def test_c059_is_not_read_as_a_periodic_revenue_metric(self) -> None:
        plan = _plan(C059)

        self.assertNotEqual(plan.task_type, "financial_metric")
        self.assertIsNone(plan.metric)
        self.assertNotIn("periodic", plan.disclosure_route)
        self.assertIsNone(plan.evidence["periodic_intent"])

    def test_c059_routes_to_the_exchange_contract_lane(self) -> None:
        plan = _plan(C059)
        decision = route_task(C059, plan)

        self.assertEqual(plan.task_type, "corporate_event")
        self.assertEqual(plan.event_type, "supply_contract")
        self.assertEqual(plan.disclosure_route, ("exchange",))
        self.assertEqual(plan.evidence["event_type"], "매출액대비")
        self.assertEqual(decision.task_type, "corporate_event")
        self.assertIsNone(decision.resolver_type)

    def test_periodic_revenue_questions_keep_their_route(self) -> None:
        for question in (
            "고려아연 2024년 매출액",
            "고려아연 2024년 매출액 대비 영업이익 비율은?",
        ):
            with self.subTest(question=question):
                plan = _plan(question)
                self.assertEqual(plan.task_type, "financial_metric")
                self.assertEqual(plan.metric, "매출액")
                self.assertEqual(plan.disclosure_route, ("periodic",))
                self.assertEqual(
                    route_task(question, plan).resolver_type,
                    "periodic_fact_resolver",
                )


# ------------------------------------------------- C059: the ratio cell itself


def _lngc_chunk(
    chunk_id: str = "c-lngc",
    doc_id: str = "exchange_20230117800134",
    *,
    ratio: str = "11.69",
    table_id: str = "t0001",
    row_start: int = 0,
) -> dict[str, object]:
    """The LNGC contract table as the chunker persists it.

    Every value is the gold record's own: the formal 계약금액, the 최근매출액 it
    is measured against, and the ratio the question asks for.
    """

    rows = [
        _row("1. 체결계약명", "LNGC 3척"),
        _row("2. 계약상대", "아시아 지역 선사"),
        _row("3. 계약금액(원)", "971,400,000,000"),
        _row("4. 최근매출액(원)", "8,311,300,000,000"),
        _row("5. 매출액대비(%)", ratio),
        _row("6. 시작일", "2023-01-16"),
        _row("7. 종료일", "2026-11-30"),
    ]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": CORP,
        "corp_name": "HD현대중공업",
        "doc_group": "exchange",
        "chunk_type": "table",
        "rcept_dt": RECEIPT.replace("-", ""),
        "section_path": ["단일판매ㆍ공급계약체결"],
        "content": "단일판매ㆍ공급계약 체결",
        "retrieval_text": "단일판매ㆍ공급계약 체결",
        "table_id": table_id,
        "source_table_id": table_id,
        "row_start": row_start,
        "row_end": row_start + len(rows) - 1,
        "table_rows": rows,
        "source_refs": [
            {
                "table_id": table_id,
                "row_start": row_start,
                "row_end": row_start + len(rows) - 1,
            }
        ],
    }


class ContractSalesRatioFieldTests(unittest.TestCase):
    def test_the_ratio_question_requests_the_ratio_field(self) -> None:
        self.assertEqual(requested_corporate_fields(C059), (SALES_RATIO,))

    def test_the_contract_amount_field_is_unchanged(self) -> None:
        self.assertEqual(
            requested_corporate_fields("계약금액은 얼마인가요?"), (CONTRACT_AMOUNT,)
        )
        self.assertEqual(
            requested_corporate_fields("공급계약의 계약금액과 매출액 대비 비율은?"),
            (CONTRACT_AMOUNT, SALES_RATIO),
        )

    def test_c059_reads_eleven_point_six_nine_from_its_own_row(self) -> None:
        records = _corporate_evidence(
            C059, (_lngc_chunk(),), states=(_state("exchange_20230117800134"),)
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.field, SALES_RATIO)
        self.assertEqual(record.status, FieldStatus.AVAILABLE)
        self.assertEqual(record.value, "11.69")
        self.assertEqual(record.doc_id, "exchange_20230117800134")
        self.assertEqual(record.chunk_id, "c-lngc")
        self.assertEqual(record.table_id, "t0001")
        self.assertEqual((record.row_start, record.row_end), (4, 4))

    def test_a_blank_ratio_cell_is_never_softened_into_a_value(self) -> None:
        records = _corporate_evidence(
            C059,
            (_lngc_chunk(ratio="-"),),
            states=(_state("exchange_20230117800134"),),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, FieldStatus.UNAVAILABLE)
        self.assertEqual(records[0].reason, FieldReason.NOT_STATED)
        self.assertIsNone(records[0].value)


# ------------------------------------------------------- C086: 계열회사 count


class AffiliateCountPhraseTests(unittest.TestCase):
    """``상장`` inside an affiliate question is a column, not a listing date."""

    def test_the_group_affiliate_combination_is_a_count(self) -> None:
        for question in (
            C086,
            "세아베스틸지주 기업집단 계열사는 상장 및 비상장을 합쳐 몇 개인가",
            "세아베스틸지주의 상장/비상장 계열회사 수는?",
            "세아베스틸지주 계열회사 현황",
        ):
            with self.subTest(question=question):
                intent = affiliate_count_intent(question)
                self.assertIsNotNone(intent, question)
                self.assertEqual(intent.metric, AFFILIATE_COUNT)

    def test_an_affiliate_mention_without_a_count_request_stays_out(self) -> None:
        for question in (
            "계열회사와의 거래 내역은?",
            "기업집단 소속 여부를 알려줘",
        ):
            with self.subTest(question=question):
                self.assertIsNone(affiliate_count_intent(question))

    def test_an_explicit_listing_date_still_belongs_to_listing_history(self) -> None:
        self.assertIsNone(
            affiliate_count_intent("계열회사 중 상장일이 가장 이른 곳은?")
        )


class AffiliateCountRoutingTests(unittest.TestCase):
    def test_c086_is_an_affiliate_count_not_listing_history(self) -> None:
        plan = _plan(C086)

        self.assertEqual(plan.evidence["periodic_intent"], AFFILIATE_COUNT)
        self.assertEqual(plan.evidence["periodic_intent_evidence"], "계열회사")
        self.assertEqual(plan.section_boosts["계열회사"], 1.0)

    def test_c086_stays_on_the_periodic_lane_it_is_answered_from(self) -> None:
        plan = _plan(C086)
        decision = route_task(C086, plan)

        self.assertEqual(decision.task_type, "periodic_fact")
        self.assertEqual(decision.resolver_type, "periodic_fact_resolver")
        self.assertIn(f"plan:periodic={AFFILIATE_COUNT}", decision.matched_signals)

    def test_listing_history_questions_are_untouched(self) -> None:
        for question in (
            "시프트업 유가증권시장 상장일",
            "시프트업 상장일은?",
            "시프트업 코스닥시장 상장 이력",
        ):
            with self.subTest(question=question):
                plan = _plan(question)
                self.assertEqual(plan.evidence["periodic_intent"], "listing_history")
                self.assertEqual(plan.section_boosts, {"회사의 개요": 1.0})


# ------------------------------------------- C086: the counts and where they sit


_HEADERS = [
    "기업집단의 명칭",
    "계열회사의 수 / 상장",
    "계열회사의 수 / 비상장",
    "계열회사의 수 / 계",
]


def _affiliate_chunk(
    *,
    headers: list[str] | None = None,
    rows: list[list[dict[str, object]]] | None = None,
    table_id: str = "t0645",
    row_start: int = 2,
) -> dict[str, object]:
    """The 계열회사 현황 row as the chunker persists it.

    ``t0645`` row 2 is the gold source ref for C086; the header rows above it
    are the two the chunker folded into ``column_headers``.
    """

    rows = rows if rows is not None else [_row("세아", "5", "22", "27")]
    return {
        "chunk_id": "periodic_20240312000629:ch_878421d46c72c812591b",
        "doc_id": "periodic_20240312000629",
        "doc_group": "periodic",
        "chunk_type": "table",
        "section_path": ["IX. 계열회사 등에 관한 사항", "1. 계열회사 현황(요약)"],
        "column_headers": headers if headers is not None else list(_HEADERS),
        "table_id": table_id,
        "source_table_id": table_id,
        "row_start": row_start,
        "row_end": row_start + len(rows) - 1,
        "table_rows": rows,
    }


class AffiliateCountFactTests(unittest.TestCase):
    def test_listed_and_unlisted_compose_the_total(self) -> None:
        counts = affiliate_counts(_affiliate_chunk())

        self.assertIsNotNone(counts)
        self.assertEqual((counts.listed, counts.unlisted), (5, 22))
        self.assertEqual(counts.total, 27)
        self.assertEqual(counts.stated_total, 27)

    def test_the_total_is_composed_even_when_the_table_states_none(self) -> None:
        counts = affiliate_counts(
            _affiliate_chunk(headers=_HEADERS[:3], rows=[_row("세아", "5", "22")])
        )

        self.assertIsNotNone(counts)
        self.assertEqual(counts.total, 27)
        self.assertIsNone(counts.stated_total)

    def test_the_row_it_was_read_from_is_citable(self) -> None:
        counts = affiliate_counts(_affiliate_chunk())

        self.assertEqual(
            dict(counts.source_ref),
            {"table_id": "t0645", "row_start": 2, "row_end": 2},
        )

    def test_a_row_below_headers_keeps_its_own_index(self) -> None:
        counts = affiliate_counts(
            _affiliate_chunk(
                rows=[
                    _row("기업집단의 명칭", "상장", "비상장", "계"),
                    _row("세아", "5", "22", "27"),
                ],
                row_start=1,
            )
        )

        self.assertEqual(counts.total, 27)
        self.assertEqual(dict(counts.source_ref)["row_start"], 2)

    def test_a_stated_total_that_disagrees_is_not_composed_away(self) -> None:
        self.assertIsNone(
            affiliate_counts(_affiliate_chunk(rows=[_row("세아", "5", "22", "30")]))
        )

    def test_two_countable_rows_are_ambiguous_rather_than_guessed(self) -> None:
        self.assertIsNone(
            affiliate_counts(
                _affiliate_chunk(
                    rows=[_row("세아", "5", "22", "27"), _row("기타", "1", "2", "3")]
                )
            )
        )

    def test_a_table_without_both_columns_reads_nothing(self) -> None:
        self.assertIsNone(
            affiliate_counts(
                _affiliate_chunk(
                    headers=["기업집단의 명칭", "계열회사의 수 / 상장"],
                    rows=[_row("세아", "5")],
                )
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
