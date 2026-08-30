"""Acquisition facts may only come from a row that proves its own acquisition.

HX04 asks for 취득일 and 취득 수량 about a put-option exercise.  The filing holds
both in one detail row -- 2024.03.07 and 868,948 beside
풋옵션권리행사배정에따른주식취득(+) -- while the same document also carries the
2024-03-14 report date, the 2,967,759 resulting position and a 120,000 unit
price.  Aliasing 취득일 onto ``reference_date`` would answer with the report
date, so the transaction method has to carry the proof, and it has to come from
the same row as the date and the quantity.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.reasoning.evidence_builder import build_evidence_set
from app.reasoning.holding_acquisition import (
    acquisition_facts,
    classify_transaction_method,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult

#: The HX04 table shape, headers and all, as the projection stores it.
HEADERS = [
    "성명(명칭)", "생년월일 또는사업자등록번호 등", "변동일*", "취득/처분방법",
    "주식등의종류", "변동 내역 / 변동전", "변동 내역 / 증감", "변동 내역 / 변동후",
    "취득/처분단가**", "취득/처분단가**", "비 고",
]


def _row(*values):
    return [{"text": value, "colspan": 1, "rowspan": 1, "is_header": False}
            for value in values]


HX04_ROW = _row(
    "(주)하이브", "120-86-78223", "2024.03.07", "풋옵션권리행사배정에따른주식취득(+)",
    "의결권있는 주식", "2,098,811", "868,948", "2,967,759", "120,000", "-", "-",
)
#: The same reporter and date on one table, one acquisition and one disposal.
DISPOSAL_ROW = _row(
    "코봇홀딩스 유한회사", "110-11-11111", "2023.10.13", "장내매도(-)",
    "의결권있는 주식", "5,635,483", "-2,320,493", "3,314,990", "9,000", "-", "-",
)
LISTING_ROW = _row(
    "코봇홀딩스 유한회사", "110-11-11111", "2023.10.13", "신규상장(유상취득)(+)",
    "의결권있는 주식", "0", "3,314,990", "3,314,990", "9,000", "-", "-",
)
ORDINARY_ROW = _row(
    "국민연금공단", "111-82-00000", "2023.06.13", "기타(+)",
    "의결권있는 주식", "2,485,201", "283,151", "2,768,352", "-", "-", "-",
)


#: The option was granted long before it was exercised; that earlier date sits
#: in the same filing and must never be reported as the acquisition date.
GRANT_ROW = _row(
    "(주)하이브", "120-86-78223", "2023.02.09", "주식매수선택권부여(+)",
    "의결권있는 주식", "2,098,811", "0", "2,098,811", "-", "-", "-",
)


def _report_chunk(chunk_id, doc_id, *, base_date="2024.03.14"):
    """A report projection: a position as of a base date, with no method."""

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": "00123456",
        "corp_name": "에스엠",
        "doc_group": "holding",
        "chunk_type": "table_projection",
        "rcept_dt": "20240314",
        "section_path": ["보유주식등의 수 및 보유비율"],
        "content": "보유 요약",
        "retrieval_text": "보유 요약",
        "projection_type": "holding_report",
        "table_id": "t0013",
        "source_table_id": "t0013",
        "row_start": 3,
        "row_end": 3,
        "source_refs": [{"table_id": "t0013", "row_start": 3, "row_end": 3}],
        "projection_fields": {
            "보고자/보유자": "(주)하이브",
            "기준일/보고일": base_date,
            "직전 보유주식수": "2,098,811",
            "증감주식수": "868,948",
            "보유주식수": "2,967,759",
        },
        "projection_field_refs": {
            label: [{"table_id": "t0013", "row_start": 3, "row_end": 3}]
            for label in ("보고자/보유자", "기준일/보고일", "직전 보유주식수",
                          "증감주식수", "보유주식수")
        },
    }


def _detail_chunk(chunk_id, doc_id, row, *, table_id="t0019", row_index=2,
                  headers=None, projection_type="holding_detail_row"):
    cells = [cell["text"] for cell in row]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": "00123456",
        "corp_name": "에스엠",
        "doc_group": "holding",
        "chunk_type": "table_projection",
        "rcept_dt": "20240314",
        "section_path": ["세부변동내역"],
        "content": "세부 변동 내역",
        "retrieval_text": "세부 변동 내역",
        "projection_type": projection_type,
        "table_id": table_id,
        "source_table_id": table_id,
        "row_start": row_index,
        "row_end": row_index,
        "column_headers": headers if headers is not None else HEADERS,
        "table_rows": [row],
        "source_refs": [{"table_id": table_id, "row_start": row_index,
                         "row_end": row_index}],
        "projection_fields": {
            "보고자/보유자": cells[0],
            "기준일/보고일": cells[2],
            "직전 보유주식수": cells[5],
            "증감주식수": cells[6],
            "보유주식수": cells[7],
        },
        "projection_field_refs": {
            label: [{"table_id": table_id, "row_start": row_index,
                     "row_end": row_index}]
            for label in ("보고자/보유자", "기준일/보고일", "직전 보유주식수",
                          "증감주식수", "보유주식수")
        },
    }


def _pair(chunk, rank):
    candidate = CandidateChunk(chunk["chunk_id"], chunk["doc_id"], chunk,
                               MetadataMatch())
    result = RetrievalResult(chunk["chunk_id"], chunk["doc_id"], 1.0 / rank, rank,
                             {"hybrid": {"final_score": 1.0 - rank / 100.0}})
    return candidate, result


def _resolve(question, chunks, *, reporter=None):
    pairs = [_pair(chunk, index) for index, chunk in enumerate(chunks, start=1)]
    plan = QueryPlan(
        query=question,
        raw_query=question,
        task_type="holding_change",
        reporter=reporter,
    )
    evidence = build_evidence_set(
        question=question,
        query_plan=plan,
        candidates=[candidate for candidate, _ in pairs],
        results=[result for _, result in pairs],
    )
    return resolve_holding_events(evidence, query_plan=plan)


class TransactionMethodTests(unittest.TestCase):
    """The method must name the acquisition; a rising number is not a reason."""

    def test_methods_that_name_an_acquisition(self) -> None:
        for method in (
            "풋옵션권리행사배정에따른주식취득(+)",
            "무상신주취득(+)",
            "유상신주취득(+)",
            "신규상장(무상취득)(+)",
            "신규상장(유상취득)(+)",
            "신규선임(유상취득)(+)",
            "장내매수(+)",
            "장외매수(+)",
        ):
            with self.subTest(method=method):
                self.assertEqual(
                    classify_transaction_method(method), "acquisition")

    def test_increases_that_explain_nothing_are_not_acquisitions(self) -> None:
        for method in ("기타(+)", "신규보고(+)", "주식배당(+)", "합병(+)",
                       "상속(+)", "수증(+)", "제3자배정유상증자(+)"):
            with self.subTest(method=method):
                self.assertNotEqual(
                    classify_transaction_method(method), "acquisition")

    def test_a_purchase_right_is_not_a_purchase(self) -> None:
        """Granting or exercising an option moves no shares by itself."""

        for method in ("주식매수선택권부여(+)", "주식매수선택권부여취소(-)",
                       "주식매수선택권행사(+)", "주식매수청구권 행사(+)"):
            with self.subTest(method=method):
                self.assertNotEqual(
                    classify_transaction_method(method), "acquisition")

    def test_disposals_are_never_acquisitions(self) -> None:
        for method in ("장내매도(-)", "장외매도(-)", "특별관계해소(-)",
                       "임원퇴임(-)", "시간외매매(-)", "공개매수청약(-)"):
            with self.subTest(method=method):
                self.assertNotEqual(
                    classify_transaction_method(method), "acquisition")

    def test_a_missing_method_proves_nothing(self) -> None:
        for method in (None, "", "-", "회사분할"):
            with self.subTest(method=method):
                self.assertNotEqual(
                    classify_transaction_method(method), "acquisition")


class SameRowExtractionTests(unittest.TestCase):
    def test_hx04_row_yields_its_own_date_and_quantity(self) -> None:
        facts = acquisition_facts(_detail_chunk("c1", "d1", HX04_ROW))
        self.assertEqual(facts["acquisition_date"], "2024-03-07")
        self.assertEqual(facts["acquired_shares"], "868,948")
        self.assertEqual(facts["acquired_shares_value"], 868948)
        self.assertEqual(facts["transaction_method"],
                         "풋옵션권리행사배정에따른주식취득(+)")
        self.assertEqual(facts["source_ref"],
                         {"table_id": "t0019", "row_start": 2, "row_end": 2})

    def test_unit_price_can_never_become_the_quantity(self) -> None:
        facts = acquisition_facts(_detail_chunk("c1", "d1", HX04_ROW))
        self.assertNotEqual(facts["acquired_shares_value"], 120000)
        self.assertNotIn("120,000", facts["acquired_shares"])

    def test_resulting_and_prior_positions_are_not_the_quantity(self) -> None:
        facts = acquisition_facts(_detail_chunk("c1", "d1", HX04_ROW))
        for value in (2967759, 2098811):
            self.assertNotEqual(facts["acquired_shares_value"], value)

    def test_a_disposal_row_yields_nothing(self) -> None:
        self.assertIsNone(
            acquisition_facts(_detail_chunk("c1", "d1", DISPOSAL_ROW)))

    def test_a_disposal_magnitude_is_never_rescued(self) -> None:
        """abs(-2,320,493) must not become an acquired quantity."""

        row = list(DISPOSAL_ROW)
        row[3] = {"text": "장내매수(+)", "colspan": 1, "rowspan": 1,
                  "is_header": False}
        # Method now claims a purchase but the signed change still falls.
        self.assertIsNone(acquisition_facts(_detail_chunk("c1", "d1", row)))

    def test_an_unexplained_increase_yields_nothing(self) -> None:
        self.assertIsNone(
            acquisition_facts(_detail_chunk("c1", "d1", ORDINARY_ROW)))

    def test_a_report_projection_can_never_prove_an_acquisition(self) -> None:
        chunk = _detail_chunk("c1", "d1", HX04_ROW,
                              projection_type="holding_report")
        self.assertIsNone(acquisition_facts(chunk))

    def test_a_row_without_a_method_column_yields_nothing(self) -> None:
        headers = list(HEADERS)
        headers[3] = "비고"
        self.assertIsNone(
            acquisition_facts(_detail_chunk("c1", "d1", HX04_ROW,
                                            headers=headers)))

    def test_a_multi_row_chunk_can_prove_nothing(self) -> None:
        """One projection is one row; a pool of rows is not same-row evidence."""

        method_only = _row("(주)하이브", "120-86-78223", "2024.03.07",
                           "풋옵션권리행사배정에따른주식취득(+)", "의결권있는 주식",
                           "", "", "", "", "-", "-")
        quantity_only = _row("(주)하이브", "120-86-78223", "2023.02.09", "",
                             "의결권있는 주식", "2,098,811", "868,948",
                             "2,967,759", "120,000", "-", "-")
        chunk = _detail_chunk("c1", "d1", method_only)
        chunk["table_rows"] = [method_only, quantity_only]
        self.assertIsNone(acquisition_facts(chunk))

    def test_provenance_is_required(self) -> None:
        chunk = _detail_chunk("c1", "d1", HX04_ROW)
        chunk["row_start"] = None
        chunk["source_table_id"] = None
        chunk["table_id"] = None
        self.assertIsNone(acquisition_facts(chunk))


class RequestedFieldTests(unittest.TestCase):
    def _requested(self, question):
        chunk = _detail_chunk("c1", "d1", HX04_ROW)
        return list(_resolve(question, [chunk]).requested_fields)

    def test_acquisition_wording_is_recognised(self) -> None:
        for question, expected in (
            ("에스엠 하이브 풋옵션 행사 주식 취득일과 취득 수량",
             ["acquisition_date", "acquired_shares"]),
            ("취득일", ["acquisition_date"]),
            ("취득 일자", ["acquisition_date"]),
            ("취득 수량", ["acquired_shares"]),
            ("취득수량", ["acquired_shares"]),
            ("취득 주식수", ["acquired_shares"]),
            ("취득주식수", ["acquired_shares"]),
        ):
            with self.subTest(question=question):
                self.assertEqual(self._requested(question), expected)

    def test_existing_wording_keeps_its_meaning(self) -> None:
        for question, expected in (
            ("변동일", ["reference_date"]),
            ("기준일", ["reference_date"]),
            ("보유주식수", ["after_shares"]),
            ("증감 주식수", ["change_shares"]),
            ("보유비율", ["after_ratio"]),
        ):
            with self.subTest(question=question):
                self.assertEqual(self._requested(question), expected)

    def test_generic_date_and_quantity_language_stays_generic(self) -> None:
        """Neither field may be reached without the acquisition noun."""

        for question in ("변동일과 수량", "보유 수량", "증감 수량", "날짜"):
            with self.subTest(question=question):
                requested = self._requested(question)
                self.assertNotIn("acquisition_date", requested)
                self.assertNotIn("acquired_shares", requested)


class AcquisitionResolutionTests(unittest.TestCase):
    QUESTION = "에스엠 하이브 풋옵션 행사 주식 취득일과 취득 수량"

    def _filing(self):
        """The acquisition row, the option grant, and the report summary."""

        return [
            _detail_chunk("d1:ch_a", "d1", HX04_ROW),
            _detail_chunk("d1:ch_g", "d1", GRANT_ROW, row_index=3),
            _report_chunk("d1:ch_r", "d1"),
        ]

    def test_only_the_proving_row_carries_the_acquisition(self) -> None:
        """Three rows of one filing; exactly one of them proves an acquisition."""

        resolution = _resolve(self.QUESTION, self._filing(), reporter="하이브")
        acquisitions = [
            event for event in resolution.events
            if event.transaction_method is not None
        ]
        self.assertEqual(len(acquisitions), 1)
        event = acquisitions[0]
        self.assertEqual(event.acquisition_date, "2024-03-07")
        self.assertEqual(event.acquired_shares.normalized, 868948)
        self.assertEqual(event.transaction_method,
                         "풋옵션권리행사배정에따른주식취득(+)")
        self.assertEqual(resolution.unresolved_fields, ())

    def test_the_proving_row_alone_resolves_exactly_one_match(self) -> None:
        resolution = _resolve(
            self.QUESTION, [_detail_chunk("d1:ch_a", "d1", HX04_ROW)],
            reporter="하이브")
        self.assertEqual(resolution.matching_event_count, 1)
        self.assertEqual(resolution.events[0].acquisition_date, "2024-03-07")

    def test_no_other_date_in_the_filing_can_answer_it(self) -> None:
        """The report date, the receipt date and the grant date all lose."""

        resolution = _resolve(self.QUESTION, self._filing(), reporter="하이브")
        dates = {
            event.acquisition_date
            for event in resolution.events
            if event.acquisition_date is not None
        }
        self.assertEqual(dates, {"2024-03-07"})
        for wrong in ("2024-03-14", "2023-02-09"):
            self.assertNotIn(wrong, dates)

    def test_no_other_quantity_in_the_filing_can_answer_it(self) -> None:
        resolution = _resolve(self.QUESTION, self._filing(), reporter="하이브")
        quantities = {
            event.acquired_shares.normalized
            for event in resolution.events
            if event.acquired_shares is not None
        }
        self.assertEqual(quantities, {868948})
        for wrong in (2967759, 2098811, 120000, 0):
            self.assertNotIn(wrong, quantities)

    def test_acquisition_facts_cite_the_row_that_proved_them(self) -> None:
        resolution = _resolve(self.QUESTION, self._filing(), reporter="하이브")
        event = next(e for e in resolution.events
                     if e.acquisition_date is not None)
        for field in ("acquisition_date", "acquired_shares"):
            refs = [
                dict(ref)
                for source in event.field_provenance[field].sources
                for ref in source.source_refs
            ]
            self.assertEqual(
                refs, [{"table_id": "t0019", "row_start": 2, "row_end": 2}])

    def test_a_disposal_beside_an_acquisition_supplies_neither_fact(self) -> None:
        """One table, one reporter, one date, opposite transactions."""

        resolution = _resolve(
            self.QUESTION,
            [
                _detail_chunk("d2:ch_sell", "d2", DISPOSAL_ROW,
                              table_id="t0026", row_index=3),
                _detail_chunk("d2:ch_buy", "d2", LISTING_ROW,
                              table_id="t0026", row_index=2),
            ],
            reporter="코봇홀딩스")
        acquired = [
            event.acquired_shares.normalized
            for event in resolution.events
            if event.acquired_shares is not None
        ]
        self.assertEqual(acquired, [3314990])
        self.assertNotIn(-2320493, acquired)
        self.assertNotIn(2320493, acquired)

    def test_another_reporter_acquisition_does_not_match(self) -> None:
        """Reporter matching still decides; acquisition only adds facts."""

        other = _row("국민연금공단", "111-82-00000", "2024.03.07", "장내매수(+)",
                     "의결권있는 주식", "1,000", "500", "1,500", "9,000", "-", "-")
        resolution = _resolve(
            self.QUESTION,
            [
                _detail_chunk("d1:ch_a", "d1", HX04_ROW),
                _detail_chunk("d1:ch_x", "d1", other, row_index=4),
            ],
            reporter="하이브")
        matched = [e for e in resolution.events if e.matches_query is True]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].acquired_shares.normalized, 868948)
        self.assertNotEqual(matched[0].acquired_shares.normalized, 500)

    def test_a_report_only_projection_resolves_no_acquisition(self) -> None:
        resolution = _resolve(
            self.QUESTION,
            [_detail_chunk("d1:ch_r", "d1", HX04_ROW,
                           projection_type="holding_report")],
            reporter="하이브")
        for event in resolution.events:
            self.assertIsNone(event.acquisition_date)
            self.assertIsNone(event.acquired_shares)
        self.assertEqual(
            sorted(resolution.unresolved_fields),
            ["acquired_shares", "acquisition_date"])

    def test_an_unexplained_increase_resolves_no_acquisition(self) -> None:
        resolution = _resolve(
            self.QUESTION, [_detail_chunk("d3:ch_o", "d3", ORDINARY_ROW)],
            reporter="국민연금")
        for event in resolution.events:
            self.assertIsNone(event.acquisition_date)
            self.assertIsNone(event.acquired_shares)

    def test_ordinary_holding_events_keep_their_shape(self) -> None:
        """Nothing acquisition-shaped appears on an unrelated question."""

        resolution = _resolve(
            "국민연금 보유주식수", [_detail_chunk("d3:ch_o", "d3", ORDINARY_ROW)])
        self.assertEqual(list(resolution.requested_fields), ["after_shares"])
        for event in resolution.events:
            self.assertIsNone(event.transaction_method)
            self.assertNotIn("acquisition_date", event.to_dict())
            self.assertNotIn("acquired_shares", event.to_dict())


class NoGoldLiteralsTests(unittest.TestCase):
    """Gold is for scoring. The lane must work from disclosure structure alone."""

    #: Everything specific to the diagnosed example: its question, its
    #: companies, its filing and its table.
    FORBIDDEN = (
        "HX04", "에스엠", "하이브", "holding_20240314001102", "t0019",
        "868,948", "868948", "2024-03-07", "2024.03.07",
        "holding_20231013000452", "t0026",
    )
    SOURCES = (
        "app/reasoning/holding_acquisition.py",
        "app/reasoning/holding_event_resolver.py",
        "app/reasoning/holding_evidence_coverage.py",
        "app/reasoning/evidence_builder.py",
        "app/reasoning/answer_composer.py",
        "app/generation/answer_generator.py",
    )

    def test_production_code_names_no_gold_specific_value(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in self.SOURCES:
            text = (root / relative).read_text(encoding="utf-8")
            for literal in self.FORBIDDEN:
                with self.subTest(source=relative, literal=literal):
                    self.assertNotIn(literal, text)


if __name__ == "__main__":
    unittest.main()
