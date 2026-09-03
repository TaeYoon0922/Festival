# -*- coding: utf-8 -*-
"""Two filings, one contract description: the asker is asked which one.

IEV2-C093 names 삼성바이오로직스, Pfizer Ireland Pharmaceuticals and
위탁생산계약, and two 단일판매ㆍ공급계약체결 filings answer to exactly that
description with different 계약금액.  Neither was ever terminated, so each is
its own single-member lifecycle, and expansion records both as
``lifecycle_not_resolved`` because there was no counterpart filing to add.

These tests run that whole seam for real -- ``extract_contract_document``, the
canonical event graph, ``seed_expansion_targets``, the trace writer the
retrieval seam uses, then the clarification provider, resolver and public
renderer -- so a change that stops carrying the lifecycle through breaks here
rather than quietly returning "no candidates" again.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.reasoning.clarification_candidates import execution_clarification_request
from app.reasoning.clarification_request import (
    ClarificationCandidate,
    ClarificationRequest,
    ClarificationState,
    clarification_text,
)
from app.reasoning.clarification_resolver import ClarificationResolver
from app.reasoning.corporate_event_graph import (
    build_corporate_event_graph,
    extract_contract_document,
)
from app.reasoning.corporate_event_resolver import seed_expansion_targets
from app.reasoning.correction_graph import DisclosureRecord
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.event_expansion import _trace as event_expansion_trace
from app.retrieval.interfaces import CandidateDocument, MetadataMatch


CORP = "00877059"
QUESTION = (
    "삼성바이오로직스가 Pfizer Ireland Pharmaceuticals와 체결한 "
    "위탁생산계약의 계약금액은?"
)
COUNTERPARTY = "Pfizer Ireland Pharmaceuticals"
CONCLUSION = "단일판매공급계약체결"
TERMINATION = "단일판매공급계약해지"
FIRST_AMOUNT = "240,993,039,040"
SECOND_AMOUNT = "922,746,708,000"


def _row(*values: str) -> list[dict[str, object]]:
    return [
        {"text": value, "colspan": 1, "rowspan": 1, "is_header": False}
        for value in values
    ]


def _record(doc_id: str, rcept_dt: str, *, subtype: str = CONCLUSION) -> DisclosureRecord:
    report = (
        "단일판매ㆍ공급계약체결"
        if subtype == CONCLUSION
        else "단일판매ㆍ공급계약해지"
    )
    return DisclosureRecord(
        doc_id=doc_id,
        corp_code=CORP,
        doc_group="exchange",
        report_nm=report,
        rcept_no=doc_id.split("_")[-1],
        rcept_dt=rcept_dt,
        doc_subtype=subtype,
        is_correction=False,
    )


def _conclusion_rows(
    amount: str, start: str, *, counterparty: str = COUNTERPARTY
) -> list[list[dict[str, object]]]:
    """The labelled cells a real 단일판매ㆍ공급계약체결 filing states."""

    return [
        _row("1. 판매ㆍ공급계약 구분", "기타 판매ㆍ공급계약"),
        _row("- 체결계약명", "의약품 위탁생산계약"),
        _row("2. 계약내역", "계약금액(원)", amount),
        _row("3. 계약상대", counterparty),
        _row("5. 계약기간", "시작일", start),
        _row("종료일", "2029-12-31"),
    ]


def _termination_rows(reference_date: str) -> list[list[dict[str, object]]]:
    return [
        _row("1. 판매ㆍ공급계약 구분", "기타 판매ㆍ공급계약"),
        _row("- 체결계약명", "의약품 위탁생산계약"),
        _row("3. 계약상대", COUNTERPARTY),
        _row("- 해지일자", "2024-01-31"),
        _row("※ 관련공시", f"{reference_date} 단일판매ㆍ공급계약체결"),
    ]


def _filing(doc_id: str, rcept_dt: str, rows, *, subtype: str = CONCLUSION):
    record = _record(doc_id, rcept_dt, subtype=subtype)
    document = extract_contract_document(
        record, [{"table_id": "t0001", "table_rows": rows}]
    )
    return record, document


def _chain(filings, *, question: str = QUESTION):
    """Graph -> expansion trace -> clarification request, all real."""

    records = [record for record, _document in filings]
    documents = {
        record.doc_id: document
        for record, document in filings
        if document is not None
    }
    graph = build_corporate_event_graph(records, documents)
    seeds = [record.doc_id for record in records]
    _wanted, events, info = seed_expansion_targets(graph, seeds)
    trace = event_expansion_trace(
        "supply_contract", "skipped", seeds=seeds, events=events, diagnostics=info
    )
    execution = SimpleNamespace(
        event_expansion=trace,
        documents=[
            CandidateDocument(
                doc_id=record.doc_id,
                metadata={
                    "report_nm": record.report_nm,
                    "rcept_dt": record.rcept_dt,
                    "corp_code": record.corp_code,
                },
                metadata_match=MetadataMatch(),
            )
            for record in records
        ],
    )
    plan = QueryUnderstanding().understand(question)
    return execution_clarification_request(
        question,
        plan,
        SimpleNamespace(resolution=None),
        execution,
        multi_document=None,
    )


def _two_pfizer_contracts():
    return [
        _filing(
            "exchange_20230302800001",
            "2023-03-02",
            _conclusion_rows(FIRST_AMOUNT, "2023-02-20"),
        ),
        _filing(
            "exchange_20230704800004",
            "2023-07-04",
            _conclusion_rows(SECOND_AMOUNT, "2023-06-30"),
        ),
    ]


class TwoOpenContractsAreOfferedAsAChoice(unittest.TestCase):
    def test_two_never_terminated_filings_produce_two_candidates(self) -> None:
        request = _chain(_two_pfizer_contracts())

        self.assertIsNotNone(request)
        self.assertEqual(request.reason, "multiple_event_instances")
        self.assertEqual(request.target_slot, "event_instance")
        self.assertEqual(len(request.candidates), 2)

    def test_each_candidate_names_the_filing_it_came_from(self) -> None:
        request = _chain(_two_pfizer_contracts())

        labels = [candidate.label for candidate in request.candidates]
        self.assertIn("2023-03-02", labels[0])
        self.assertIn("2023-07-04", labels[1])
        for candidate in request.candidates:
            self.assertEqual(candidate.semantic_type, "event_instance")
            self.assertEqual(candidate.provenance, "corporate_event_graph")
            self.assertTrue(candidate.value.startswith("evt_"))
        self.assertEqual(
            len({candidate.value for candidate in request.candidates}), 2
        )

    def test_no_amount_is_asserted_by_the_candidates(self) -> None:
        request = _chain(_two_pfizer_contracts())

        rendered = " ".join(candidate.label for candidate in request.candidates)
        self.assertNotIn(FIRST_AMOUNT, rendered)
        self.assertNotIn(SECOND_AMOUNT, rendered)

    def test_the_decision_is_clarify_without_a_classifier(self) -> None:
        request = _chain(_two_pfizer_contracts())
        decision = ClarificationResolver().resolve(request)

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        self.assertEqual(len(decision.candidates), 2)

    def test_a_classifier_pick_still_clarifies(self) -> None:
        """The provider never declared event choice safe to resolve for you."""

        request = _chain(_two_pfizer_contracts())
        classifier = SimpleNamespace(
            classify=lambda question, candidates: SimpleNamespace(
                succeeded=True,
                status="ok",
                result=SimpleNamespace(decision="resolved", candidate_ids=("E1",)),
            )
        )
        decision = ClarificationResolver(classifier).resolve(request)

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        self.assertEqual(decision.reason, "classifier_resolution_not_declared_safe")
        self.assertIsNone(decision.selected_candidate_id)


class ThePublicQuestionNamesWhatToAnswerWith(unittest.TestCase):
    def test_the_text_asks_which_filing_by_receipt_date(self) -> None:
        decision = ClarificationResolver().resolve(_chain(_two_pfizer_contracts()))
        text = clarification_text(decision)

        self.assertIn("어느", text)
        self.assertIn("공시일", text)
        self.assertIn("2023-03-02", text)
        self.assertIn("2023-07-04", text)

    def test_the_text_asserts_neither_amount(self) -> None:
        decision = ClarificationResolver().resolve(_chain(_two_pfizer_contracts()))
        text = clarification_text(decision)

        self.assertNotIn(FIRST_AMOUNT, text)
        self.assertNotIn(SECOND_AMOUNT, text)

    def test_a_metric_choice_keeps_its_own_wording(self) -> None:
        """Only event instances are asked about by 공시일."""

        candidates = (
            ClarificationCandidate("M1", "금융비용", "metric", "synthetic"),
            ClarificationCandidate("M2", "기타비용", "metric", "synthetic"),
        )
        decision = ClarificationResolver().resolve(
            ClarificationRequest("영업외비용은?", candidates)
        )

        self.assertEqual(
            clarification_text(decision),
            "금융비용을 말씀하시는 건가요, 아니면 기타비용을 말씀하시는 건가요?",
        )


class OnlySettledLifecyclesAreOffered(unittest.TestCase):
    def test_one_contract_alone_is_not_a_choice(self) -> None:
        request = _chain(_two_pfizer_contracts()[:1])

        self.assertIsNone(request)

    def test_a_terminated_contract_is_offered_as_its_own_filing(self) -> None:
        """A 해지 joins its contract's lifecycle; the choice stays two contracts.

        The first contract and its termination resolve into one two-member
        event, so that event is expanded rather than skipped.  It is still one
        of the two contracts the question describes, and it is named by the
        filing that concluded it -- never by the 해지 that ended it.
        """

        filings = _two_pfizer_contracts()
        filings.append(
            _filing(
                "exchange_20240201800009",
                "2024-02-01",
                _termination_rows("2023-03-02"),
                subtype=TERMINATION,
            )
        )
        request = _chain(filings)

        self.assertIsNotNone(request)
        labels = [candidate.label for candidate in request.candidates]
        self.assertEqual(len(labels), 2)
        self.assertIn("2023-03-02", labels[0])
        self.assertIn("2023-07-04", labels[1])
        for label in labels:
            self.assertNotIn("해지", label)
            self.assertNotIn("2024-02-01", label)

    def test_a_reference_outside_the_corpus_is_a_gap_not_a_choice(self) -> None:
        """``*_not_in_corpus`` is the real gap the singleton rule must exclude."""

        filings = _two_pfizer_contracts()[:1]
        filings.append(
            _filing(
                "exchange_20240201800009",
                "2024-02-01",
                _termination_rows("2019-05-05"),
                subtype=TERMINATION,
            )
        )
        request = _chain(filings)

        self.assertIsNone(request)


class UnrelatedContractsStillFailClosed(unittest.TestCase):
    def test_two_different_counterparties_the_question_never_named(self) -> None:
        filings = [
            _filing(
                "exchange_20230302800001",
                "2023-03-02",
                _conclusion_rows(FIRST_AMOUNT, "2023-02-20", counterparty="Acme Bio"),
            ),
            _filing(
                "exchange_20230704800004",
                "2023-07-04",
                _conclusion_rows(
                    SECOND_AMOUNT, "2023-06-30", counterparty="Beta Pharma"
                ),
            ),
        ]
        request = _chain(filings, question="삼성바이오로직스의 위탁생산계약 계약금액은?")

        self.assertIsNone(request)

    def test_the_named_counterparty_selects_one_contract_and_ends_the_choice(self) -> None:
        filings = [
            _filing(
                "exchange_20230302800001",
                "2023-03-02",
                _conclusion_rows(FIRST_AMOUNT, "2023-02-20"),
            ),
            _filing(
                "exchange_20230704800004",
                "2023-07-04",
                _conclusion_rows(SECOND_AMOUNT, "2023-06-30", counterparty="Acme Bio"),
            ),
        ]
        request = _chain(filings)

        self.assertIsNone(request)


if __name__ == "__main__":
    unittest.main()
