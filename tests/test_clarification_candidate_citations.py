# -*- coding: utf-8 -*-
"""Citing the filings that prove a contract choice exists.

IEV2-C093 asks for one 계약금액 and the corpus holds two filings that answer
its description equally well, so the served answer is a question back.  That
question is itself a claim about the corpus -- two distinguishable contracts
were disclosed -- and a claim needs evidence.  These tests pin what that
evidence may be: the candidates' own filings, each cited, and nothing else.

What is deliberately *not* asserted anywhere below is a 계약금액.  Citing a
candidate states that its filing exists, never that its number is the answer.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.api.pipeline import AnswerPipeline, _clarification_candidate_evidence
from app.api.schemas import AnswerResponse
from app.api.settings import ApiSettings
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.clarification_candidates import execution_clarification_request
from app.reasoning.clarification_request import (
    ClarificationCandidate,
    ClarificationDecision,
    ClarificationState,
)
from app.reasoning.clarification_resolver import ClarificationResolver
from app.reasoning.corporate_event_graph import (
    build_corporate_event_graph,
    extract_contract_document,
)
from app.reasoning.corporate_event_resolver import seed_expansion_targets
from app.reasoning.correction_graph import (
    CorrectionNotice,
    DisclosureRecord,
    build_correction_graph,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.reasoning.query_validation import QueryValidator
from app.retrieval.event_expansion import _trace as event_expansion_trace
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


CORP = "00000001"
COUNTERPARTY = "Pfizer Ireland Pharmaceuticals"
QUESTION = (
    "테스트 회사가 Pfizer Ireland Pharmaceuticals와 체결한 "
    "위탁생산계약의 계약금액은?"
)
FIRST_DOC = "exchange_20230302800001"
SECOND_DOC = "exchange_20230704800004"
FIRST_AMOUNT = "240,993,039,040"
SECOND_AMOUNT = "922,746,708,000"
REPORT_NM = "단일판매ㆍ공급계약체결"


class _Understanding:
    def __init__(self, plan: QueryPlan) -> None:
        self.plan = plan

    def understand(self, question, *, top_k):
        del question, top_k
        return self.plan


class _Executor:
    def __init__(self, execution) -> None:
        self.execution = execution

    def execute(self, plan):
        self.execution.plan = plan
        return self.execution


def _row(*values: str) -> list[dict[str, object]]:
    return [
        {"text": value, "colspan": 1, "rowspan": 1, "is_header": False}
        for value in values
    ]


def _conclusion_rows(amount: str, start: str) -> list[list[dict[str, object]]]:
    return [
        _row("1. 판매ㆍ공급계약 구분", "기타 판매ㆍ공급계약"),
        _row("- 체결계약명", "의약품 위탁생산계약"),
        _row("2. 계약내역", "계약금액(원)", amount),
        _row("3. 계약상대", COUNTERPARTY),
        _row("5. 계약기간", "시작일", start),
        _row("종료일", "2029-12-31"),
    ]


def _filing(
    doc_id: str,
    rcept_dt: str,
    amount: str,
    start: str,
    *,
    report_nm: str = REPORT_NM,
    is_correction: bool = False,
):
    record = DisclosureRecord(
        doc_id=doc_id,
        corp_code=CORP,
        doc_group="exchange",
        report_nm=report_nm,
        rcept_no=doc_id.split("_")[-1],
        rcept_dt=rcept_dt,
        doc_subtype="단일판매공급계약체결",
        is_correction=is_correction,
    )
    document = extract_contract_document(
        record, [{"table_id": "t0001", "table_rows": _conclusion_rows(amount, start)}]
    )
    return record, document


def _pair(doc_id: str, amount: str, rcept_dt: str, rank: int, *, report_nm: str):
    chunk_id = f"{doc_id}:ch01"
    payload = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "exchange",
        "chunk_type": "table",
        "retrieval_text": f"| 계약금액 | {amount} |",
        "content": f"| 계약금액 | {amount} |",
        "corp_code": CORP,
        "corp_name": "테스트 회사",
        "rcept_dt": rcept_dt,
        "report_nm": report_nm,
        "section_path": ["계약내역"],
        "period": {"fiscal_year": 2023, "period_type": "fiscal_year"},
        "source_refs": [{"table_id": "t0001", "row_start": 1, "row_end": 2}],
    }
    return (
        CandidateChunk(chunk_id, doc_id, payload, MetadataMatch()),
        RetrievalResult(chunk_id, doc_id, 1.0, rank, {}),
    )


def _plan(*, period: QueryPeriod | None = None, raw_query: str = QUESTION) -> QueryPlan:
    return QueryPlan(
        query="Pfizer Ireland Pharmaceuticals 위탁생산계약 계약금액",
        raw_query=raw_query,
        company="테스트 회사",
        corp_code=CORP,
        task_type="corporate_event",
        event_type="supply_contract",
        period=period,
        disclosure_route=("exchange",),
        evidence={"operation": "inspect_event"},
    )


def _expansion_trace(filings, *, notices=None):
    """The real graph, resolver and trace writer -- no hand-written events."""

    records = [record for record, _document in filings]
    documents = {record.doc_id: document for record, document in filings}
    graph = build_corporate_event_graph(
        records,
        documents,
        correction_graph=(
            build_correction_graph(records, notices) if notices else None
        ),
    )
    seeds = [record.doc_id for record in records]
    _wanted, events, info = seed_expansion_targets(graph, seeds)
    return event_expansion_trace(
        "supply_contract", "skipped", seeds=seeds, events=events, diagnostics=info
    )


def _two_contracts():
    return [
        _filing(FIRST_DOC, "2023-03-02", FIRST_AMOUNT, "2023-02-20"),
        _filing(SECOND_DOC, "2023-07-04", SECOND_AMOUNT, "2023-06-30"),
    ]


def _execution(plan, pairs, *, trace, documents):
    return SimpleNamespace(
        plan=plan,
        documents=tuple(documents),
        chunks=tuple(pair[0] for pair in pairs),
        results=tuple(pair[1] for pair in pairs),
        routing={},
        correction_expansion={},
        event_expansion=dict(trace),
    )


def _documents(*specs):
    return tuple(
        CandidateDocument(
            doc_id,
            {"report_nm": report_nm, "rcept_dt": rcept_dt, "corp_code": CORP},
            MetadataMatch(),
        )
        for doc_id, report_nm, rcept_dt in specs
    )


CONTRACT_DOCUMENTS = _documents(
    (FIRST_DOC, REPORT_NM, "20230302"),
    (SECOND_DOC, REPORT_NM, "20230704"),
)


CORRECTION_DOC = "exchange_20230704800001"
CORRECTED_AMOUNT = "250,000,000,000"
CORRECTION_REPORT_NM = f"[기재정정]{REPORT_NM}"

CORRECTED_DOCUMENTS = _documents(
    (FIRST_DOC, REPORT_NM, "20230302"),
    (CORRECTION_DOC, CORRECTION_REPORT_NM, "20230704"),
    (SECOND_DOC, REPORT_NM, "20230704"),
)


def _corrected_contracts():
    """FIRST_DOC, the 기재정정 that supersedes it, and an unrelated contract.

    The correcting filing is received the same day as the second contract, so
    naming a candidate by the member a chain collapsed to would print one date
    twice and destroy the very dimension the question asks about.
    """

    return [
        _filing(FIRST_DOC, "2023-03-02", FIRST_AMOUNT, "2023-02-20"),
        _filing(
            CORRECTION_DOC,
            "2023-07-04",
            CORRECTED_AMOUNT,
            "2023-02-20",
            report_nm=CORRECTION_REPORT_NM,
            is_correction=True,
        ),
        _filing(SECOND_DOC, "2023-07-04", SECOND_AMOUNT, "2023-06-30"),
    ]


def _correction_notices():
    """What the 기재정정 filing itself states about the filing it corrects."""

    return {
        CORRECTION_DOC: CorrectionNotice(
            doc_id=CORRECTION_DOC,
            target_submitted_on="2023-03-02",
            target_report_nm="단일판매ㆍ공급계약 체결",
            corrected_on="2023-07-04",
            source_table_id="t0002",
            source_label="정정관련공시서류제출일",
        )
    }


def _corrected_trace():
    return _expansion_trace(_corrected_contracts(), notices=_correction_notices())


def _corrected_pairs(*, serve_root: bool = True):
    pairs = [
        _pair(
            CORRECTION_DOC,
            CORRECTED_AMOUNT,
            "20230704",
            2,
            report_nm=CORRECTION_REPORT_NM,
        ),
        _pair(SECOND_DOC, SECOND_AMOUNT, "20230704", 3, report_nm=REPORT_NM),
    ]
    if serve_root:
        pairs.insert(
            0, _pair(FIRST_DOC, FIRST_AMOUNT, "20230302", 1, report_nm=REPORT_NM)
        )
    return pairs


def _contract_pairs(*, first_rank: int = 1, second_rank: int = 2):
    return [
        _pair(FIRST_DOC, FIRST_AMOUNT, "20230302", first_rank, report_nm=REPORT_NM),
        _pair(SECOND_DOC, SECOND_AMOUNT, "20230704", second_rank, report_nm=REPORT_NM),
    ]


def _answer(plan, execution):
    pipeline = AnswerPipeline(
        understanding=_Understanding(plan),
        executor=_Executor(execution),
        settings=ApiSettings(top_k=10),
        verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
        query_validator=QueryValidator(),
        answerability_guard=AnswerabilityGuard(),
        clarification_resolver=ClarificationResolver(),
    )
    return pipeline.answer("IEV2-C093", plan.raw_query)


def _served(payload):
    return [
        (row["rank"], row["doc_id"], row["chunk_id"])
        for row in payload["retrieved_context"]
    ]


def _request(execution, plan):
    return execution_clarification_request(
        QUESTION, plan, SimpleNamespace(resolution=None), execution
    )


class EachCandidateCarriesItsOwnFiling(unittest.TestCase):
    def _execution_for(self, pairs):
        plan = _plan()
        return plan, _execution(
            plan,
            pairs,
            trace=_expansion_trace(_two_contracts()),
            documents=CONTRACT_DOCUMENTS,
        )

    def test_two_candidates_are_bound_to_two_distinct_filings(self) -> None:
        plan, execution = self._execution_for(_contract_pairs())
        request = _request(execution, plan)

        self.assertIsNotNone(request)
        self.assertEqual(request.target_slot, "event_instance")
        self.assertEqual(
            [candidate.source for candidate in request.candidates],
            [(f"{FIRST_DOC}:ch01", FIRST_DOC), (f"{SECOND_DOC}:ch01", SECOND_DOC)],
        )

    def test_the_decision_is_still_clarify(self) -> None:
        plan, execution = self._execution_for(_contract_pairs())
        request = _request(execution, plan)
        decision = ClarificationResolver().resolve(request)

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        # Binding a filing to each candidate says nothing about the choice
        # itself, so both halves of the state machine keep their own account
        # of it: why the candidates were built, and why it asked back.
        self.assertEqual(request.reason, "multiple_event_instances")
        self.assertEqual(decision.reason, "multiple_bounded_candidates")

    def test_a_filing_retrieval_never_served_leaves_the_candidate_unbound(self) -> None:
        plan, execution = self._execution_for(_contract_pairs()[:1])
        request = _request(execution, plan)

        self.assertIsNotNone(request)
        self.assertEqual(
            [candidate.source for candidate in request.candidates],
            [(f"{FIRST_DOC}:ch01", FIRST_DOC), None],
        )


class ThePublicQuestionCitesTheFilingsThatProveIt(unittest.TestCase):
    def setUp(self) -> None:
        plan = _plan()
        self.payload = _answer(
            plan,
            _execution(
                plan,
                _contract_pairs(),
                trace=_expansion_trace(_two_contracts()),
                documents=CONTRACT_DOCUMENTS,
            ),
        )

    def test_the_route_is_still_clarification(self) -> None:
        self.assertEqual(self.payload["think_trace"]["route"], "clarification")
        self.assertFalse(self.payload["think_trace"]["answerable"])
        AnswerResponse.model_validate(self.payload)

    def test_the_text_still_names_the_dimension_to_answer_with(self) -> None:
        self.assertIn("공시일", self.payload["answer"])
        self.assertIn("어느", self.payload["answer"])
        self.assertIn("2023-03-02", self.payload["answer"])
        self.assertIn("2023-07-04", self.payload["answer"])

    def test_every_candidate_is_cited(self) -> None:
        self.assertIn("[1]", self.payload["answer"])
        self.assertIn("[2]", self.payload["answer"])

    def test_each_marker_names_its_own_candidate_filing(self) -> None:
        lines = self.payload["answer"].splitlines()
        first = next(line for line in lines if "2023-03-02" in line)
        second = next(line for line in lines if "2023-07-04" in line)
        rows = {row["rank"]: row for row in self.payload["retrieved_context"]}

        self.assertIn("[1]", first)
        self.assertIn("[2]", second)
        self.assertEqual(rows[1]["doc_id"], FIRST_DOC)
        self.assertEqual(rows[1]["chunk_id"], f"{FIRST_DOC}:ch01")
        self.assertEqual(rows[2]["doc_id"], SECOND_DOC)
        self.assertEqual(rows[2]["chunk_id"], f"{SECOND_DOC}:ch01")

    def test_no_contract_amount_is_stated(self) -> None:
        self.assertNotIn(FIRST_AMOUNT, self.payload["answer"])
        self.assertNotIn(SECOND_AMOUNT, self.payload["answer"])
        self.assertNotIn("계약금액", self.payload["answer"])

    def test_no_machine_identifier_reaches_the_public_text(self) -> None:
        for candidate in self.payload["think_trace"]["clarification"]["candidates"]:
            self.assertNotIn("evt_", candidate["label"])
            self.assertNotIn("exchange_", candidate["label"])
        self.assertNotIn("evt_", self.payload["answer"])
        self.assertNotIn("exchange_", self.payload["answer"])


class OnlyTheProvenCandidatesAreServed(unittest.TestCase):
    def test_arbitrary_top_k_evidence_is_not_exposed(self) -> None:
        """A higher-ranked unrelated filing proves nothing about this choice."""

        plan = _plan()
        noise = _pair("periodic_2024", "999", "20240401", 1, report_nm="사업보고서")
        payload = _answer(
            plan,
            _execution(
                plan,
                [noise, *_contract_pairs(first_rank=2, second_rank=3)],
                trace=_expansion_trace(_two_contracts()),
                documents=CONTRACT_DOCUMENTS,
            ),
        )

        self.assertEqual(
            _served(payload),
            [
                (1, FIRST_DOC, f"{FIRST_DOC}:ch01"),
                (2, SECOND_DOC, f"{SECOND_DOC}:ch01"),
            ],
        )

    def test_one_row_per_candidate(self) -> None:
        plan = _plan()
        payload = _answer(
            plan,
            _execution(
                plan,
                _contract_pairs(),
                trace=_expansion_trace(_two_contracts()),
                documents=CONTRACT_DOCUMENTS,
            ),
        )

        self.assertEqual(len(payload["retrieved_context"]), 2)


class AnUnboundCandidateIsNeverCited(unittest.TestCase):
    def test_no_marker_and_no_evidence_when_one_filing_was_not_served(self) -> None:
        plan = _plan()
        payload = _answer(
            plan,
            _execution(
                plan,
                _contract_pairs()[:1],
                trace=_expansion_trace(_two_contracts()),
                documents=CONTRACT_DOCUMENTS,
            ),
        )

        self.assertEqual(payload["think_trace"]["route"], "clarification")
        self.assertEqual(payload["retrieved_context"], [])
        self.assertNotIn("[1]", payload["answer"])
        self.assertNotIn("[2]", payload["answer"])
        self.assertIn("공시일", payload["answer"])

    def test_a_half_identity_is_rejected_by_the_contract(self) -> None:
        with self.assertRaises(ValueError):
            ClarificationCandidate(
                "E1",
                f"{REPORT_NM} (2023-03-02)",
                "event_instance",
                "corporate_event_graph",
                source_doc_id=FIRST_DOC,
            )


class OneLogicalRootIsOneChoice(unittest.TestCase):
    def test_correction_members_of_one_contract_are_not_two_candidates(self) -> None:
        """C2 collapsed them into one event id; nothing here re-splits them."""

        plan = _plan()
        trace = {
            "corporate_event_expansion": {
                "events": [
                    {"event_id": "evt_same", "seed_member_doc_id": FIRST_DOC},
                    {"event_id": "evt_same", "seed_member_doc_id": SECOND_DOC},
                ]
            }
        }
        payload = _answer(
            plan,
            _execution(
                plan,
                _contract_pairs(),
                trace=trace,
                documents=CONTRACT_DOCUMENTS,
            ),
        )

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])

    def test_two_candidates_sharing_one_filing_share_one_row(self) -> None:
        decision = ClarificationDecision(
            state=ClarificationState.CLARIFY,
            reason="multiple_event_instances",
            candidates=(
                ClarificationCandidate(
                    "E1",
                    f"{REPORT_NM} (2023-03-02)",
                    "event_instance",
                    "corporate_event_graph",
                    source_doc_id=FIRST_DOC,
                    source_chunk_id=f"{FIRST_DOC}:ch01",
                ),
                ClarificationCandidate(
                    "E2",
                    f"정정 {REPORT_NM} (2023-03-02)",
                    "event_instance",
                    "corporate_event_graph",
                    source_doc_id=FIRST_DOC,
                    source_chunk_id=f"{FIRST_DOC}:ch01",
                ),
            ),
        )
        plan = _plan()
        execution = _execution(
            plan,
            _contract_pairs(),
            trace=_expansion_trace(_two_contracts()),
            documents=CONTRACT_DOCUMENTS,
        )

        rows, citations, markers = _clarification_candidate_evidence(execution, decision)

        self.assertEqual([row["doc_id"] for row in rows], [FIRST_DOC])
        self.assertEqual(markers, {"E1": "[1]", "E2": "[1]"})
        self.assertEqual([citation.citation_id for citation in citations], ["[1]"])


class OtherClarificationLanesAreUntouched(unittest.TestCase):
    def test_a_metric_choice_gets_no_candidate_evidence(self) -> None:
        decision = ClarificationDecision(
            state=ClarificationState.CLARIFY,
            reason="holding_metric_ambiguity",
            candidates=(
                ClarificationCandidate(
                    "M1", "보유주식수", "holding_metric", "query_understanding"
                ),
                ClarificationCandidate(
                    "M2", "보유비율", "holding_metric", "query_understanding"
                ),
            ),
        )
        plan = _plan()
        execution = _execution(
            plan,
            _contract_pairs(),
            trace=_expansion_trace(_two_contracts()),
            documents=CONTRACT_DOCUMENTS,
        )

        self.assertIsNone(_clarification_candidate_evidence(execution, decision))

    def test_an_explicit_filing_date_still_answers_normally(self) -> None:
        plan = _plan(
            period=QueryPeriod(
                from_date="2023-03-02",
                to_date="2023-03-02",
                period_type="event_date",
            ),
            raw_query="테스트 회사가 2023년 3월 2일 공시한 위탁생산계약의 계약금액은?",
        )
        payload = _answer(
            plan,
            _execution(
                plan,
                _contract_pairs(),
                trace=_expansion_trace(_two_contracts()),
                documents=CONTRACT_DOCUMENTS,
            ),
        )

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])

    def test_a_single_contract_still_answers_normally(self) -> None:
        plan = _plan()
        payload = _answer(
            plan,
            _execution(
                plan,
                _contract_pairs()[:1],
                trace=_expansion_trace(_two_contracts()[:1]),
                documents=CONTRACT_DOCUMENTS[:1],
            ),
        )

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])


class ACorrectedContractIsNamedByTheFilingItStartedFrom(unittest.TestCase):
    """IEV2-C093's live shape: one contract was corrected, the other was not.

    P0-A collapses a correction chain to its latest member, so the member id
    the expansion trace carries for this contract is the 기재정정 filing of
    2023-07-04.  The other contract was disclosed on 2023-07-04 too.  Naming
    candidates by that collapsed member therefore offers "2023-07-04" against
    "2023-07-04" -- two contracts rendered as one date, which is the dimension
    the clarification is asking the reader to choose along.

    The graph already proves the way out: every member carries the root filing
    of its own correction chain, and for an uncorrected filing that root is
    itself.  That root is what names a contract as against another contract.
    The latest member still states what the contract now says, but it can never
    be what tells two contracts apart.
    """

    def _request(self, *, serve_root: bool = True, documents=None):
        plan = _plan()
        execution = _execution(
            plan,
            _corrected_pairs(serve_root=serve_root),
            trace=_corrected_trace(),
            documents=CORRECTED_DOCUMENTS if documents is None else documents,
        )
        return _request(execution, plan)

    def test_the_trace_carries_the_root_of_each_contract(self) -> None:
        block = _corrected_trace()["corporate_event_expansion"]
        events = [*block["events"], *block["skipped"]]

        self.assertEqual(
            sorted(event["seed_root_doc_id"] for event in events),
            [FIRST_DOC, SECOND_DOC],
        )

    def test_each_candidate_is_named_by_its_root_filing(self) -> None:
        request = self._request()

        self.assertIsNotNone(request)
        self.assertEqual(
            [candidate.label for candidate in request.candidates],
            [f"{REPORT_NM} (2023-03-02)", f"{REPORT_NM} (2023-07-04)"],
        )

    def test_each_citation_is_the_same_root_filing_as_its_label(self) -> None:
        request = self._request()

        self.assertEqual(
            [candidate.source for candidate in request.candidates],
            [(f"{FIRST_DOC}:ch01", FIRST_DOC), (f"{SECOND_DOC}:ch01", SECOND_DOC)],
        )

    def test_the_correction_is_neither_a_candidate_nor_a_candidate_filing(self) -> None:
        request = self._request()

        self.assertEqual(len(request.candidates), 2)
        for candidate in request.candidates:
            self.assertNotEqual(candidate.source_doc_id, CORRECTION_DOC)
            self.assertNotIn("기재정정", candidate.label)

    def test_the_decision_is_still_clarify(self) -> None:
        decision = ClarificationResolver().resolve(self._request())

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        self.assertEqual(decision.reason, "multiple_bounded_candidates")

    def test_an_unserved_root_is_not_replaced_by_the_member_that_corrects_it(
        self,
    ) -> None:
        """The root filing exists but retrieval never served a chunk of it.

        The candidate keeps the identity the graph proved and simply goes
        uncited, exactly as any unserved candidate filing does.  Borrowing the
        correction's chunk would cite 2023-07-04 under a 2023-03-02 label.
        """

        request = self._request(serve_root=False)

        self.assertIsNotNone(request)
        self.assertEqual(
            [candidate.label for candidate in request.candidates],
            [f"{REPORT_NM} (2023-03-02)", f"{REPORT_NM} (2023-07-04)"],
        )
        self.assertEqual(
            [candidate.source for candidate in request.candidates],
            [None, (f"{SECOND_DOC}:ch01", SECOND_DOC)],
        )

    def test_a_root_retrieval_never_reached_at_all_fails_closed(self) -> None:
        """With no root filing to name, the correction does not stand in for it."""

        request = self._request(
            serve_root=False,
            documents=_documents(
                (CORRECTION_DOC, CORRECTION_REPORT_NM, "20230704"),
                (SECOND_DOC, REPORT_NM, "20230704"),
            ),
        )

        self.assertIsNone(request)


class TheServedEvidenceIsTheRootFilings(unittest.TestCase):
    def setUp(self) -> None:
        plan = _plan()
        self.payload = _answer(
            plan,
            _execution(
                plan,
                _corrected_pairs(),
                trace=_corrected_trace(),
                documents=CORRECTED_DOCUMENTS,
            ),
        )

    def test_the_route_is_still_clarification(self) -> None:
        self.assertEqual(self.payload["think_trace"]["route"], "clarification")
        self.assertFalse(self.payload["think_trace"]["answerable"])
        AnswerResponse.model_validate(self.payload)

    def test_the_two_dates_offered_are_the_two_contracts(self) -> None:
        self.assertIn("2023-03-02", self.payload["answer"])
        self.assertIn("2023-07-04", self.payload["answer"])
        self.assertIn("공시일", self.payload["answer"])

    def test_only_the_two_root_filings_are_served(self) -> None:
        self.assertEqual(
            _served(self.payload),
            [
                (1, FIRST_DOC, f"{FIRST_DOC}:ch01"),
                (2, SECOND_DOC, f"{SECOND_DOC}:ch01"),
            ],
        )

    def test_each_marker_names_its_own_root_filing(self) -> None:
        lines = self.payload["answer"].splitlines()
        first = next(line for line in lines if "2023-03-02" in line)
        second = next(line for line in lines if "2023-07-04" in line)

        self.assertIn("[1]", first)
        self.assertIn("[2]", second)

    def test_no_contract_amount_is_stated(self) -> None:
        for amount in (FIRST_AMOUNT, SECOND_AMOUNT, CORRECTED_AMOUNT):
            self.assertNotIn(amount, self.payload["answer"])


class AnUncorrectedContractKeepsItsExistingIdentity(unittest.TestCase):
    def test_a_root_is_its_own_root_when_nothing_corrected_it(self) -> None:
        block = _expansion_trace(_two_contracts())["corporate_event_expansion"]
        events = [*block["events"], *block["skipped"]]

        for event in events:
            self.assertEqual(event["seed_root_doc_id"], event["seed_member_doc_id"])

    def test_the_candidates_are_unchanged(self) -> None:
        plan = _plan()
        request = _request(
            _execution(
                plan,
                _contract_pairs(),
                trace=_expansion_trace(_two_contracts()),
                documents=CONTRACT_DOCUMENTS,
            ),
            plan,
        )

        self.assertEqual(
            [candidate.source for candidate in request.candidates],
            [(f"{FIRST_DOC}:ch01", FIRST_DOC), (f"{SECOND_DOC}:ch01", SECOND_DOC)],
        )


if __name__ == "__main__":
    unittest.main()
