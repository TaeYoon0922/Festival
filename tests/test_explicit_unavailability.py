"""Citable evidence is not a supported field value.

STEP 11-C.  Independent Eval v2 proved false answerability in two shapes.  A
supply contract whose formal 계약금액 cell read ``-`` answered with
``confirmed_fields=["contract_amount"]`` because the alias appeared in the
generated prose beside real but unrelated numbers.  A holding report that
recorded its 취득단가 as omitted answered with no confirmed field at all,
because citations existed.

STEP 11-C.2 then closed six pre-commit findings.  The corporate producer no
longer decides which filing it may read: it consumes the P0-B state that
retrieval already computed, so these tests build that state with the real
:func:`seed_expansion_targets` and the real trace writer rather than a
producer-only convention.  A test that hand-rolled the metadata could only
prove the producer agrees with itself.

Nothing below asserts an implementation detail of the retired STEP 11-A
design: there is no markdown parser to exercise, no nearest-field rule, no
``temporal_match`` authority and no document-global precedence.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import generate_answer
from app.reasoning.answerability import (
    AnswerabilityGuard,
    AnswerabilityStatus,
    guarded_answer_text,
)
from app.reasoning.corporate_event import (
    AMBIGUOUS,
    FAMILY_SUPPLY_CONTRACT,
    LIFECYCLE_OPEN,
    RESOLVED,
    ROLE_CONTRACT,
    ROLE_TERMINATION,
    UNRESOLVED,
    CorporateEventState,
)
from app.reasoning.corporate_event_authority import (
    _correction_scoped_targets,
    _event_scoped_targets,
    selected_corporate_member,
)
from app.reasoning.corporate_event_field_evidence import (
    CONTRACT_AMOUNT,
    _selection_intent,
    corporate_event_field_evidence,
    requested_corporate_fields,
)
from app.reasoning.corporate_event_resolver import seed_expansion_targets
from app.retrieval.correction_expansion import _trace as correction_trace
from app.retrieval.event_expansion import _seed_doc_ids
from app.reasoning.evidence_builder import build_evidence_set
from app.reasoning.field_evidence import (
    DOMAIN_CORPORATE_EVENT,
    FieldEvidence,
    FieldReason,
    FieldStatus,
    accepted_field_evidence,
    resolve_field_states,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.holding_field_evidence import (
    ACQUISITION_UNIT_PRICE,
    _unit_price_column,
    holding_field_evidence,
    requested_holding_fields,
)
from app.reasoning.multi_document_evidence import (
    MultiDocumentEvidence,
    MultiDocumentFacts,
)
from app.reasoning.query_plan import DateBasis, QueryPeriod, QueryPlan
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.event_expansion import _trace as event_expansion_trace
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


CORP = "00126380"
EVENT = "evt-supply-1"
RECEIPT = "2024-05-20"
LATER_RECEIPT = "2024-11-04"


# ------------------------------------------------------- P0-B state fixtures


def _state(
    doc_id: str,
    *,
    canonical: str | None = None,
    role: str = ROLE_CONTRACT,
    event_id: str = EVENT,
    corp_code: str = CORP,
    group: str | None = None,
    correction_status: str | None = None,
    member_count: int = 1,
) -> CorporateEventState:
    """One filing as P0-B records it.

    ``canonical_doc_id`` is P0-A's answer to "which version stands": for a
    resolved chain every member reports the chain's last filing, which is what
    makes a superseded member recognisable without walking the chain.
    """

    return CorporateEventState(
        doc_id=doc_id,
        event_id=event_id,
        corp_code=corp_code,
        event_family=FAMILY_SUPPLY_CONTRACT,
        member_role=role,
        lifecycle_status=LIFECYCLE_OPEN,
        resolution_status=RESOLVED,
        canonical_doc_id=canonical or doc_id,
        member_count=member_count,
        correction_group_id=group,
        correction_resolution_status=correction_status,
    )


class _GraphView:
    """The slice of the P0-B view the expansion seam actually calls."""

    def __init__(self, states):
        self._states = {state.doc_id: state for state in states}

    def event_states(self, doc_ids):
        return {
            doc_id: self._states[doc_id]
            for doc_id in doc_ids
            if doc_id in self._states
        }

    def get_event_timeline(self, doc_id):
        return ()

    def get_event(self, doc_id):
        return None

    def get_related_documents(self, doc_id):
        return ()


def _execution(
    *,
    states=(),
    seeds=(),
    status: str = "expanded",
    correction: dict | None = None,
    expanded_events=(),
):
    """An execution carrying the trace the real expansion seam writes.

    Built by running the real ``seed_expansion_targets`` and the real trace
    writer, so a change that stopped carrying P0-B state through would break
    these tests rather than quietly leave them passing.

    ``expanded_events`` supplies the per-event entries expansion would have
    produced -- ``{"event_id": ..., "target_doc_ids": [...]}`` -- which is the
    shape the trace records and the shape the authority layer scopes by.
    """

    entries = [dict(entry) for entry in expanded_events]
    if states:
        _wanted, events, info = seed_expansion_targets(_GraphView(states), list(seeds))
        trace = event_expansion_trace(
            FAMILY_SUPPLY_CONTRACT, status, seeds=list(seeds),
            events=entries or events, diagnostics=info,
            added_documents=[
                SimpleNamespace(doc_id=doc_id)
                for entry in entries
                for doc_id in entry.get("target_doc_ids") or ()
            ],
        )
    else:
        trace = event_expansion_trace(FAMILY_SUPPLY_CONTRACT, status)
    return SimpleNamespace(
        results=tuple(SimpleNamespace(chunk_id=f"served-{i}") for i in range(1)),
        event_expansion=trace,
        correction_expansion=correction or {},
    )


# --------------------------------------------------------- corporate chunks


def _cell(text: str) -> dict[str, object]:
    return {"text": text, "colspan": 1, "rowspan": 1, "is_header": False}


def _row(*values: str) -> list[dict[str, object]]:
    return [_cell(value) for value in values]


def _contract_chunk(
    chunk_id: str,
    doc_id: str,
    *,
    amount: str,
    rcept_dt: str = RECEIPT,
    corp_code: str = CORP,
    table_id: str = "t0001",
    row_start: int = 0,
    extra_rows: tuple[list[dict[str, object]], ...] = (),
    section: str = "단일판매ㆍ공급계약체결",
) -> dict[str, object]:
    """A supply-contract table as the chunker persists it.

    The real numbers beside the contract amount are what made a blank amount
    look answerable: 최근매출액 and 매출액대비 are genuine, cited, and say
    nothing at all about the contract amount.
    """

    rows = [
        _row("1. 체결계약명", "반도체 부품 공급계약"),
        _row("2. 계약상대", "주식회사 상대회사"),
        _row("3. 계약금액(원)", amount),
        _row("4. 최근매출액(원)", "1,482,000,000,000"),
        _row("5. 매출액대비(%)", "12.4"),
        _row("6. 시작일", "2024-06-01"),
        _row("7. 종료일", "2026-05-31"),
        *extra_rows,
    ]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": corp_code,
        "corp_name": "예시전자",
        "doc_group": "major",
        "chunk_type": "table",
        "rcept_dt": rcept_dt.replace("-", ""),
        "section_path": [section],
        "content": "단일판매ㆍ공급계약 체결",
        "retrieval_text": "단일판매ㆍ공급계약 체결",
        "table_id": table_id,
        "source_table_id": table_id,
        "row_start": row_start,
        "row_end": row_start + len(rows) - 1,
        "table_rows": rows,
        "source_refs": [
            {"table_id": table_id, "row_start": row_start,
             "row_end": row_start + len(rows) - 1}
        ],
    }


def _unrelated_chunk(chunk_id: str, doc_id: str, *, rcept_dt: str = RECEIPT):
    """A different disclosure the same company filed the same day."""

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": CORP,
        "corp_name": "예시전자",
        "doc_group": "major",
        "chunk_type": "paragraph",
        "rcept_dt": rcept_dt.replace("-", ""),
        "section_path": ["임원ㆍ주요주주 특정증권등 소유상황보고"],
        "content": "임원 변동 안내",
        "retrieval_text": "임원 변동 안내",
        "source_refs": [],
    }


def _corporate_plan(
    question: str,
    *,
    corp_code: str | None = CORP,
    exact_receipt: bool = True,
    correction_intent: str | None = None,
) -> QueryPlan:
    correction_policy = "any"
    correction_route_evidence = {}
    plan_correction_intent = correction_intent
    if correction_intent == "latest":
        correction_policy = "latest_preferred"
        correction_route_evidence = {"is_correction": "explicit latest"}
    elif correction_intent == "corrected":
        correction_policy = "corrected_only"
        correction_route_evidence = {"is_correction": "explicit corrected"}
        plan_correction_intent = None
    return QueryPlan(
        query=question,
        raw_query=question,
        task_type="disclosure_lookup",
        corp_code=corp_code,
        event_type="supply_contract",
        date_basis=(
            DateBasis.RECEIPT_DATE if exact_receipt else DateBasis.UNSPECIFIED
        ),
        period=(
            QueryPeriod(from_date=RECEIPT, to_date=RECEIPT,
                        period_type="receipt_date")
            if exact_receipt
            else None
        ),
        correction_policy=correction_policy,
        route_evidence=correction_route_evidence,
        evidence={"correction_intent": plan_correction_intent},
    )


def _repository_shaped_execution(states_by_asked_doc):
    """Carry the repository's asked-key/canonical-value state shape verbatim."""

    return SimpleNamespace(
        event_expansion={
            "event_member_states": {
                asked_doc_id: state.to_dict()
                for asked_doc_id, state in states_by_asked_doc.items()
            }
        },
        correction_expansion={},
    )


def _items(*chunks: dict[str, object]):
    pairs = [
        (
            CandidateChunk(str(chunk["chunk_id"]), str(chunk["doc_id"]), chunk,
                           MetadataMatch()),
            RetrievalResult(str(chunk["chunk_id"]), str(chunk["doc_id"]),
                            1.0 / index, index, {}),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    evidence = build_evidence_set(
        question="q",
        query_plan=QueryPlan(query="q", task_type="disclosure_lookup"),
        candidates=[candidate for candidate, _ in pairs],
        results=[result for _, result in pairs],
    )
    return evidence.served_items


def _corporate_evidence(
    question: str,
    chunks: tuple[dict[str, object], ...],
    *,
    states=(),
    plan: QueryPlan | None = None,
    status: str = "expanded",
    multi_document=None,
    seeds=None,
    expanded_events=(),
    correction=None,
):
    if seeds is None:
        seeds = [str(chunk["doc_id"]) for chunk in chunks]
    return corporate_event_field_evidence(
        question=question,
        plan=plan if plan is not None else _corporate_plan(question),
        execution=_execution(states=states, seeds=seeds, status=status,
                             expanded_events=expanded_events, correction=correction),
        evidence_items=_items(*chunks),
        multi_document=multi_document,
    )


# ----------------------------------------------------- corporate activation


class CorporateRequestedFieldContractTests(unittest.TestCase):
    """Intent comes from the question, never from evidence or from prose."""

    def test_generic_contract_amount_wording_activates(self) -> None:
        for question in ("계약금액은 얼마인가요?", "계약 금액이 궁금합니다"):
            self.assertEqual(
                requested_corporate_fields(question), (CONTRACT_AMOUNT,), question
            )

    def test_other_generic_event_questions_do_not_activate(self) -> None:
        for question in (
            "계약 시작일은 언제인가요?",
            "계약 종료일은 언제인가요?",
            "해지 주요사유는 무엇인가요?",
            "계약 상대방은 누구인가요?",
            "해지금액은 얼마인가요?",
        ):
            self.assertEqual(requested_corporate_fields(question), (), question)


# ------------------------------------------------------- corporate findings


class CorporateFieldEvidenceTests(unittest.TestCase):
    QUESTION = "계약금액은 얼마인가요?"

    def test_c1_blank_amount_is_unavailable_and_ignores_nearby_numbers(self) -> None:
        records = _corporate_evidence(
            self.QUESTION, (_contract_chunk("c1", "d1", amount="-"),),
            states=(_state("d1"),),
        )

        self.assertEqual(len(records), 1)
        found = records[0]
        self.assertIs(found.status, FieldStatus.UNAVAILABLE)
        self.assertIs(found.reason, FieldReason.NOT_STATED)
        self.assertIsNone(found.value)
        # The provenance is the contract-amount row, not the revenue line below.
        self.assertEqual(found.chunk_id, "c1")
        self.assertEqual(found.table_id, "t0001")
        self.assertEqual((found.row_start, found.row_end), (2, 2))

    def test_c2_field_bound_deferral_is_withheld_or_deferred(self) -> None:
        deferred = _contract_chunk(
            "c1", "d1", amount="-",
            extra_rows=(_row("8. 비고", "계약금액은 유보기한 종료 후 공시 예정"),),
        )

        records = _corporate_evidence(self.QUESTION, (deferred,), states=(_state("d1"),))

        self.assertIs(records[0].status, FieldStatus.UNAVAILABLE)
        self.assertIs(records[0].reason, FieldReason.WITHHELD_OR_DEFERRED)

    def test_document_level_reservation_note_does_not_bind_the_field(self) -> None:
        """A remark that names no field proves nothing about this one."""

        generic = _contract_chunk(
            "c1", "d1", amount="-", extra_rows=(_row("8. 비고", "공시유보 신청"),)
        )

        records = _corporate_evidence(self.QUESTION, (generic,), states=(_state("d1"),))

        self.assertIs(records[0].reason, FieldReason.NOT_STATED)

    def test_c3_stated_amount_is_available(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("c1", "d1", amount="184,000,000,000"),),
            states=(_state("d1"),),
        )

        self.assertIs(records[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(records[0].value, "184,000,000,000")
        self.assertEqual(records[0].chunk_id, "c1")

    def test_the_field_is_absent_from_the_filing(self) -> None:
        chunk = _contract_chunk("c1", "d1", amount="-")
        chunk["table_rows"] = [
            row for row in list(chunk["table_rows"])
            if "계약금액" not in str(row[0]["text"])
        ]

        records = _corporate_evidence(
            self.QUESTION,
            (chunk,),
            states=(_state("d1"),),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        self.assertIs(records[0].status, FieldStatus.MISSING)
        self.assertIsNone(records[0].chunk_id)


class CorporateHistoricalReceiptAuthorityTests(unittest.TestCase):
    """An exact receipt date reads the filing selected by that date."""

    QUESTION = "2024년 5월 20일 공시된 공급계약의 계약금액은 얼마인가요?"

    def _run(self, served, states_by_asked_doc, *, plan=None, unserved=()):
        all_chunks = [*served, *unserved]
        candidates = [
            CandidateChunk(str(chunk["chunk_id"]), str(chunk["doc_id"]), chunk,
                           MetadataMatch())
            for chunk in all_chunks
        ]
        results = [
            RetrievalResult(str(chunk["chunk_id"]), str(chunk["doc_id"]),
                            1.0 / index, index, {})
            for index, chunk in enumerate(served, start=1)
        ]
        selected_plan = plan or _corporate_plan(self.QUESTION)
        carried = _repository_shaped_execution(states_by_asked_doc)
        execution = SimpleNamespace(
            plan=selected_plan,
            chunks=candidates,
            results=results,
            event_expansion=carried.event_expansion,
            correction_expansion=carried.correction_expansion,
        )
        result = AgentOrchestrator().run(self.QUESTION, selected_plan, execution)
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated,
            plan=selected_plan,
            agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        return result, generated, verdict, guarded_answer_text(
            verdict, generated.answer_text
        )

    def test_d1_historical_mapping_key_is_citable_field_authority(self) -> None:
        historical = _contract_chunk(
            "historical-amount", "historical-contract", amount="-",
            extra_rows=(_row("8. 비고", "계약금액은 유보기한 종료 후 공시 예정"),),
        )
        corrected = _contract_chunk(
            "corrected-amount", "corrected-contract", amount="205,000,000,000",
            rcept_dt=LATER_RECEIPT, table_id="t0002",
        )
        termination = _contract_chunk(
            "termination-amount", "contract-termination", amount="99,000,000,000",
            rcept_dt=LATER_RECEIPT, table_id="t0003",
            section="단일판매ㆍ공급계약해지",
        )
        states = {
            # This is the real repository shape: the asked key is historical,
            # while both identity fields in its value name the chain latest.
            "historical-contract": _state(
                "corrected-contract", canonical="corrected-contract", group="chain-1",
                correction_status=RESOLVED, member_count=2,
            ),
            "contract-termination": _state(
                "contract-termination", role=ROLE_TERMINATION
            ),
        }

        result, generated, verdict, answer = self._run(
            (historical, termination), states, unserved=(corrected,)
        )

        self.assertEqual(len(result.field_evidence), 1)
        found = result.field_evidence[0]
        self.assertEqual(found.doc_id, "historical-contract")
        self.assertEqual(found.chunk_id, "historical-amount")
        self.assertIs(found.status, FieldStatus.UNAVAILABLE)
        self.assertIs(found.reason, FieldReason.WITHHELD_OR_DEFERRED)
        self.assertFalse(verdict.answerable)
        self.assertEqual(verdict.unavailable_fields, (CONTRACT_AMOUNT,))
        self.assertIsNotNone(verdict.refusal_citation)
        self.assertIn(verdict.refusal_citation, answer)
        citations = {citation.citation_id: citation.chunk_id
                     for citation in generated.citations}
        self.assertEqual(citations[verdict.refusal_citation], "historical-amount")
        self.assertNotIn("205,000,000,000", answer)
        self.assertNotIn("99,000,000,000", answer)

    def test_d1_explicit_latest_or_corrected_keeps_canonical_authority(self) -> None:
        historical = _contract_chunk(
            "historical-amount", "historical-contract", amount="100,000,000"
        )
        corrected = _contract_chunk(
            "corrected-amount", "corrected-contract", amount="205,000,000,000",
            rcept_dt=LATER_RECEIPT, table_id="t0002",
        )
        final = _state(
            "corrected-contract", canonical="corrected-contract", group="chain-1",
            correction_status=RESOLVED, member_count=2,
        )

        for intent in ("latest", "corrected"):
            with self.subTest(intent=intent):
                result, _generated_answer, verdict, _answer = self._run(
                    (historical, corrected),
                    {"historical-contract": final, "corrected-contract": final},
                    plan=_corporate_plan(self.QUESTION, correction_intent=intent),
                )

                authoritative = [record for record in result.field_evidence
                                 if record.authoritative]
                self.assertEqual(len(authoritative), 1)
                self.assertEqual(authoritative[0].doc_id, "corrected-contract")
                self.assertEqual(authoritative[0].value, "205,000,000,000")
                self.assertTrue(verdict.answerable)

    def test_d1_unresolved_finality_still_fails_closed(self) -> None:
        historical = _contract_chunk(
            "historical-amount", "historical-contract", amount="184,000,000,000"
        )
        state = _state(
            "corrected-contract", canonical="corrected-contract", group="chain-1",
            correction_status=UNRESOLVED, member_count=2,
        )

        result, _generated_answer, verdict, _answer = self._run(
            (historical,), {"historical-contract": state}
        )

        self.assertEqual([record.status for record in result.field_evidence],
                         [FieldStatus.CONFLICT])
        self.assertFalse(verdict.answerable)

    def test_d1_multiple_events_still_fail_closed(self) -> None:
        first = _contract_chunk("amount-a", "contract-a", amount="100,000,000")
        second = _contract_chunk(
            "amount-b", "contract-b", amount="200,000,000", table_id="t0002"
        )

        result, _generated_answer, verdict, _answer = self._run(
            (first, second),
            {
                "contract-a": _state("contract-a", event_id="event-a"),
                "contract-b": _state("contract-b", event_id="event-b"),
            },
        )

        self.assertEqual([record.status for record in result.field_evidence],
                         [FieldStatus.CONFLICT])
        self.assertFalse(verdict.answerable)

    def test_d2_numeric_historical_amount_uses_the_formal_row(self) -> None:
        historical = _contract_chunk(
            "historical-amount", "historical-contract", amount="184,000,000,000"
        )
        state = _state(
            "corrected-contract", canonical="corrected-contract", group="chain-1",
            correction_status=RESOLVED, member_count=2,
        )

        result, _generated_answer, verdict, _answer = self._run(
            (historical,), {"historical-contract": state}
        )

        self.assertEqual(len(result.field_evidence), 1)
        found = result.field_evidence[0]
        self.assertIs(found.status, FieldStatus.AVAILABLE)
        self.assertEqual(found.value, "184,000,000,000")
        self.assertEqual(found.chunk_id, "historical-amount")
        self.assertEqual((found.row_start, found.row_end), (2, 2))
        self.assertNotEqual(found.value, "1,482,000,000,000")
        self.assertNotEqual(found.value, "12.4")
        self.assertTrue(verdict.answerable)


class CorporateProducerDeclineTests(unittest.TestCase):
    """No authority, no finding: the request keeps the lane it always had."""

    QUESTION = "계약금액은 얼마인가요?"

    def test_a_question_that_requests_no_canonical_field_declines(self) -> None:
        self.assertEqual(
            _corporate_evidence(
                "계약 상대방은 누구인가요?",
                (_contract_chunk("c1", "d1", amount="-"),),
                states=(_state("d1"),),
                plan=_corporate_plan("계약 상대방은 누구인가요?"),
            ),
            (),
        )

    def test_no_carried_event_state_declines(self) -> None:
        """P0-B said nothing, so this lane does not claim the request."""

        self.assertEqual(
            _corporate_evidence(
                self.QUESTION, (_contract_chunk("c1", "d1", amount="-"),)
            ),
            (),
        )

    def test_a_served_filing_p0b_does_not_know_declines(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("c1", "d1", amount="-"),),
            states=(_state("other-doc"),),
        )

        self.assertEqual(records, ())

    def test_a_different_company_declines(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("c1", "d1", amount="-"),),
            states=(_state("d1", corp_code="00999999"),),
        )

        self.assertEqual(records, ())

    def test_a_termination_member_cannot_answer_for_the_contract(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("c1", "d1", amount="205,000,000,000"),),
            states=(_state("d1", role=ROLE_TERMINATION),),
        )

        self.assertEqual(records, ())


# ================================================================= P1 CLOSURE


class P1_1_NegativeCitationTests(unittest.TestCase):
    """A grounded refusal must be able to point at what grounds it.

    Ordinary composition cites the highest-ranked handful of served evidence.
    A filing whose formal field is blank can sit outside that slice and still
    be the only thing that proves the refusal, which used to leave the answer
    correct and uncited -- the original T18 defect on the citation axis.
    """

    QUESTION = "계약금액은 얼마인가요?"

    def _run(self, chunks, states):
        seeds = [str(chunk["doc_id"]) for chunk in chunks]
        pairs = [
            (
                CandidateChunk(str(c["chunk_id"]), str(c["doc_id"]), c, MetadataMatch()),
                RetrievalResult(str(c["chunk_id"]), str(c["doc_id"]), 1.0 / i, i, {}),
            )
            for i, c in enumerate(chunks, start=1)
        ]
        plan = _corporate_plan(self.QUESTION)
        expansion = _execution(states=states, seeds=seeds).event_expansion
        execution = SimpleNamespace(
            plan=plan,
            chunks=[candidate for candidate, _ in pairs],
            results=[result for _, result in pairs],
            event_expansion=expansion,
            correction_expansion={},
        )
        result = AgentOrchestrator().run(self.QUESTION, plan, execution)
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated, plan=plan, agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        return result, generated, verdict

    def _blank_beyond_the_cap(self):
        fillers = [
            _unrelated_chunk(f"f{i}", f"df{i}", rcept_dt=LATER_RECEIPT)
            for i in range(1, 7)
        ]
        blank = _contract_chunk("cAMT", "dC", amount="-")
        return [*fillers, blank]

    def test_negative_evidence_beyond_the_citation_cap_is_still_cited(self) -> None:
        chunks = self._blank_beyond_the_cap()
        result, generated, verdict = self._run(chunks, (_state("dC"),))
        answer = guarded_answer_text(verdict, generated.answer_text)

        # It really is outside what ordinary composition would have cited.
        self.assertGreater(len(chunks), 6)
        self.assertEqual(chunks[-1]["chunk_id"], "cAMT")
        # And it really is served.
        self.assertIn("cAMT", {r.chunk_id for r in result.evidence_results})

        self.assertFalse(verdict.answerable)
        self.assertEqual(verdict.unavailable_fields, (CONTRACT_AMOUNT,))
        self.assertIsNotNone(verdict.refusal_citation)
        self.assertIn(verdict.refusal_citation, answer)
        marker = {c.citation_id: c.chunk_id for c in generated.citations}
        self.assertEqual(marker[verdict.refusal_citation], "cAMT")

    def test_without_the_union_that_chunk_would_not_be_cited(self) -> None:
        """Mutation control: ordinary composition alone never reaches it."""

        chunks = self._blank_beyond_the_cap()
        pairs = [
            (
                CandidateChunk(str(c["chunk_id"]), str(c["doc_id"]), c, MetadataMatch()),
                RetrievalResult(str(c["chunk_id"]), str(c["doc_id"]), 1.0 / i, i, {}),
            )
            for i, c in enumerate(chunks, start=1)
        ]
        plan = _corporate_plan(self.QUESTION)
        # The same execution with no carried P0-B state: the producer declines,
        # so no union happens and the blank filing stays uncited.
        execution = SimpleNamespace(
            plan=plan,
            chunks=[candidate for candidate, _ in pairs],
            results=[result for _, result in pairs],
            event_expansion={},
            correction_expansion={},
        )
        result = AgentOrchestrator().run(self.QUESTION, plan, execution)
        generated = generate_answer(result.answer_draft)

        self.assertEqual(result.field_evidence, ())
        self.assertNotIn("cAMT", {c.chunk_id for c in generated.citations})

    def test_an_available_field_adds_no_citation(self) -> None:
        """The union is bounded to the state that needs grounding."""

        chunks = [
            *[_unrelated_chunk(f"f{i}", f"df{i}", rcept_dt=LATER_RECEIPT)
              for i in range(1, 7)],
            _contract_chunk("cAMT", "dC", amount="184,000,000,000"),
        ]
        result, generated, verdict = self._run(chunks, (_state("dC"),))

        self.assertTrue(verdict.answerable)
        self.assertNotIn("cAMT", {c.chunk_id for c in generated.citations})


class P1_2_CorrectionAbsenceTests(unittest.TestCase):
    """Missing correction metadata is not proof that a filing is final.

    An ordinary contract-amount question names no correction, so correction
    expansion reports ``not_requested`` and finds no groups.  Reading that
    zero as "this filing is its own final version" turned an absence of
    metadata into authority.
    """

    QUESTION = "계약금액은 얼마인가요?"

    def test_ordinary_question_reports_no_correction_group(self) -> None:
        """The precondition this closure rests on, asserted rather than assumed."""

        from app.retrieval.correction_expansion import _trace as correction_trace

        trace = correction_trace(None, "not_requested")
        self.assertEqual(trace["correction_group_count"], 0)
        self.assertEqual(trace["correction_status"], "not_requested")

    def test_a_superseded_member_is_not_authoritative_without_correction_intent(
        self,
    ) -> None:
        """P0-A already knows; the asker never has to say 정정."""

        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("cOLD", "dROOT", amount="100,000,000"),),
            states=(
                _state("dROOT", canonical="dLATEST", group="g1",
                       correction_status=RESOLVED, member_count=2),
            ),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        authoritative = [record for record in records if record.authoritative]
        self.assertEqual(len(authoritative), 1)
        self.assertIs(authoritative[0].status, FieldStatus.MISSING)
        self.assertEqual(authoritative[0].doc_id, "dLATEST")
        self.assertNotIn(
            "100,000,000", [record.value for record in authoritative]
        )

    def test_unresolved_correction_finality_fails_closed(self) -> None:
        for status in (AMBIGUOUS, UNRESOLVED):
            records = _corporate_evidence(
                self.QUESTION,
                (_contract_chunk("c1", "d1", amount="184,000,000,000"),),
                states=(
                    _state("d1", group="g1", correction_status=status, member_count=2),
                ),
            )
            self.assertEqual(
                [record.status for record in records], [FieldStatus.CONFLICT], status
            )

    def test_a_standalone_filing_is_positively_final(self) -> None:
        """P0-A looked and found no group; that is a finding, not an absence."""

        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("c1", "d1", amount="184,000,000,000"),),
            states=(_state("d1"),),
        )

        self.assertIs(records[0].status, FieldStatus.AVAILABLE)


class P1_3_CorrectionMiddleLinkTests(unittest.TestCase):
    """ROOT -> MIDDLE -> LATEST: only LATEST may answer."""

    QUESTION = "계약금액은 얼마인가요?"

    def _chain(self, *docs):
        return tuple(
            _state(doc, canonical="dLATEST", group="g1",
                   correction_status=RESOLVED, member_count=3)
            for doc in docs
        )

    def test_a_middle_member_is_never_authoritative(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("cMID", "dMID", amount="100,000,000"),),
            states=self._chain("dMID"),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        authoritative = [record for record in records if record.authoritative]
        self.assertEqual(len(authoritative), 1)
        self.assertIs(authoritative[0].status, FieldStatus.MISSING)
        self.assertEqual(authoritative[0].doc_id, "dLATEST")
        self.assertFalse(
            any(record.value == "100,000,000" and record.authoritative
                for record in records)
        )

    def test_a_root_member_is_never_authoritative(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (_contract_chunk("cROOT", "dROOT", amount="100,000,000"),),
            states=self._chain("dROOT"),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        self.assertTrue(all(
            record.status is FieldStatus.MISSING
            for record in records if record.authoritative
        ))

    def test_the_latest_member_answers_when_it_is_served(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (
                _contract_chunk("cMID", "dMID", amount="100,000,000"),
                _contract_chunk("cLATEST", "dLATEST", amount="184,000,000,000",
                                table_id="t0002"),
            ),
            states=self._chain("dMID", "dLATEST"),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        authoritative = [record for record in records if record.authoritative]
        self.assertEqual(len(authoritative), 1)
        self.assertIs(authoritative[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(authoritative[0].doc_id, "dLATEST")
        self.assertEqual(authoritative[0].value, "184,000,000,000")
        # The superseded filing is recorded and cannot decide.
        self.assertEqual(
            [record.doc_id for record in records if not record.authoritative], ["dMID"]
        )

    def test_the_authority_layer_reports_the_chain_latest(self) -> None:
        """The guard that makes the three cases above structural."""

        authority = selected_corporate_member(
            execution=_execution(states=self._chain("dMID"), seeds=["dMID"]),
            corp_code=CORP,
            served_doc_ids=["dMID"],
        )

        self.assertIsNotNone(authority.member)
        self.assertEqual(authority.member.authoritative_doc_id, "dLATEST")


class P1_4_EventIdentityTests(unittest.TestCase):
    """Which filing answers is P0-B's decision, not a company-and-date match."""

    QUESTION = "계약금액은 얼마인가요?"

    def test_an_unrelated_same_day_filing_is_irrelevant(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (
                _contract_chunk("c1", "dC", amount="184,000,000,000"),
                _unrelated_chunk("u1", "dU"),
            ),
            states=(_state("dC"),),
        )

        self.assertEqual(len(records), 1)
        self.assertIs(records[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(records[0].value, "184,000,000,000")
        self.assertEqual(records[0].doc_id, "dC")

    def test_two_genuine_contract_events_fail_closed(self) -> None:
        records = _corporate_evidence(
            self.QUESTION,
            (
                _contract_chunk("c1", "dC1", amount="184,000,000,000"),
                _contract_chunk("c2", "dC2", amount="205,000,000,000",
                                table_id="t0002"),
            ),
            states=(
                _state("dC1", event_id="evt-a"),
                _state("dC2", event_id="evt-b"),
            ),
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])

    def test_a_later_lifecycle_member_cannot_supply_the_original_amount(self) -> None:
        """The later filing is served, tempting and same-company -- and not it."""

        records = _corporate_evidence(
            self.QUESTION,
            (
                _contract_chunk("cORIG", "dORIG", amount="-"),
                _contract_chunk("cLATER", "dLATER", amount="205,000,000,000",
                                rcept_dt=RECEIPT, table_id="t0002",
                                section="단일판매ㆍ공급계약해지"),
            ),
            # Same lifecycle, same day: only the member role separates them, so
            # a date filter could not have rejected the later filing here.
            states=(
                _state("dORIG"),
                _state("dLATER", role=ROLE_TERMINATION),
            ),
        )

        self.assertEqual({record.doc_id for record in records}, {"dORIG"})
        self.assertIs(records[0].status, FieldStatus.UNAVAILABLE)


class P1_A_AuthorityCompletenessTests(unittest.TestCase):
    """Carried authority covers the retrieval seeds, not the served set.

    Those two can differ, and when they do a filing P0-B never described is not
    thereby irrelevant.  Selecting among the filings that happen to carry state
    would answer from whichever candidate the seed window kept -- turning a
    real two-event conflict into a confident single-event answer.
    """

    QUESTION = "계약금액은 얼마인가요?"

    def _amount(self, chunk_id, doc_id, amount, table_id="t0001"):
        return _contract_chunk(chunk_id, doc_id, amount=amount, table_id=table_id)

    def test_a_candidate_without_carried_authority_fails_closed(self) -> None:
        """The reported defect: dA is served and competing, dB alone has state."""

        served = (
            self._amount("cA", "dA", "184,000,000,000"),
            self._amount("cB", "dB", "99,000,000", table_id="t0002"),
        )

        records = _corporate_evidence(
            self.QUESTION, served,
            states=(_state("dB", event_id="evt-b"),), seeds=["dB"],
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])
        self.assertNotIn("99,000,000", [record.value for record in records])

    def test_the_same_shape_with_complete_authority_conflicts_for_its_own_reason(
        self,
    ) -> None:
        """Control: with both states carried this is an ordinary multi-event."""

        served = (
            self._amount("cA", "dA", "184,000,000,000"),
            self._amount("cB", "dB", "99,000,000", table_id="t0002"),
        )

        records = _corporate_evidence(
            self.QUESTION, served,
            states=(_state("dA", event_id="evt-a"), _state("dB", event_id="evt-b")),
            seeds=["dA", "dB"],
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])

    def test_the_seed_window_itself_reproduces_the_truncation(self) -> None:
        """Drive the real seeding helper rather than asserting a hand-cut set.

        ``_seed_doc_ids`` is what bounds the carried authority in production;
        narrowing its limit here reproduces the architectural trigger without
        touching the configured one.
        """

        served = (
            self._amount("cA", "dA", "184,000,000,000"),
            self._amount("cB", "dB", "99,000,000", table_id="t0002"),
        )
        results = [
            RetrievalResult(str(chunk["chunk_id"]), str(chunk["doc_id"]),
                            1.0 / index, index, {})
            for index, chunk in enumerate(served, start=1)
        ]
        seeds = _seed_doc_ids(results, 1)          # the seed window cuts dB out
        self.assertEqual(seeds, ["dA"])

        records = _corporate_evidence(
            self.QUESTION, served,
            states=(_state("dA", event_id="evt-a"),), seeds=seeds,
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])
        self.assertNotIn("184,000,000,000", [record.value for record in records])

    def test_an_unrelated_served_filing_does_not_break_completeness(self) -> None:
        """Positive control: the guard must not be over-broad."""

        served = (
            self._amount("cB", "dB", "184,000,000,000"),
            _unrelated_chunk("u1", "dU"),
        )

        records = _corporate_evidence(
            self.QUESTION, served, states=(_state("dB"),), seeds=["dB"],
        )

        self.assertEqual(len(records), 1)
        self.assertIs(records[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(records[0].value, "184,000,000,000")
        self.assertEqual(records[0].doc_id, "dB")

    def test_several_chunks_of_one_filing_are_one_candidate(self) -> None:
        served = (
            self._amount("cB1", "dB", "184,000,000,000"),
            _contract_chunk("cB2", "dB", amount="184,000,000,000", row_start=20),
            _contract_chunk("cB3", "dB", amount="184,000,000,000", row_start=40),
        )

        records = _corporate_evidence(
            self.QUESTION, served, states=(_state("dB"),), seeds=["dB"],
        )
        states = resolve_field_states(records, served={"cB1", "cB2", "cB3"})

        self.assertTrue(records)
        self.assertTrue(all(r.status is FieldStatus.AVAILABLE for r in records))
        self.assertIs(states[CONTRACT_AMOUNT].status, FieldStatus.AVAILABLE)

    def test_a_same_event_expansion_member_is_accounted_for(self) -> None:
        """Expansion pulls in members of the lifecycle the seeds established.

        Those were never seeds, so they carry no state -- but upstream said
        which event they belong to, and it is the event a candidate is already
        on.  They are not competing candidates and must not fail the request
        closed.
        """

        root = _contract_chunk("cROOT", "dROOT", amount="-")
        latest = _contract_chunk("cLATEST", "dLATEST", amount="184,000,000,000",
                                 table_id="t0002")

        records = _corporate_evidence(
            self.QUESTION, (root, latest),
            states=(_state("dROOT", canonical="dLATEST", group="g1",
                           correction_status=RESOLVED, member_count=2),),
            seeds=["dROOT"],
            expanded_events=({"event_id": EVENT, "target_doc_ids": ["dLATEST"]},),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        authoritative = [record for record in records if record.authoritative]
        self.assertEqual(len(authoritative), 1)
        self.assertIs(authoritative[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(authoritative[0].doc_id, "dLATEST")

    def test_an_expansion_member_of_another_event_is_not_accounted_for(self) -> None:
        """P1-B.  "An expansion added it" is not a statement about which event.

        One request expands every lifecycle its seeds touch.  A contract pulled
        in behind an unrelated termination seed is a genuinely competing event,
        and exempting it because some expansion produced it is how a confident
        single-event answer used to survive a real conflict.
        """

        served = (
            self._amount("cA", "dA", "184,000,000,000"),
            self._amount("cB", "dContractB", "99,000,000", table_id="t0002"),
        )

        records = _corporate_evidence(
            self.QUESTION, served,
            states=(
                _state("dA", event_id="evt-a"),
                _state("dTermB", event_id="evt-b", role=ROLE_TERMINATION),
            ),
            seeds=["dA", "dTermB"],
            expanded_events=({"event_id": "evt-b",
                              "target_doc_ids": ["dContractB"]},),
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])
        self.assertNotIn("184,000,000,000", [record.value for record in records])

    def test_only_the_carried_events_targets_are_explained(self) -> None:
        """The scoping itself, isolated from the rest of the selection."""

        execution = SimpleNamespace(event_expansion={
            "corporate_event_expansion": {"events": [
                {"event_id": "evt-a", "target_doc_ids": ["a2", "a3"]},
                {"event_id": "evt-b", "target_doc_ids": ["b2"]},
                {"event_id": "evt-c", "target_doc_ids": ["c2"]},
            ]},
        })

        self.assertEqual(
            sorted(_event_scoped_targets(execution, {"evt-a"})), ["a2", "a3"]
        )
        self.assertEqual(sorted(_event_scoped_targets(execution, set())), [])
        self.assertEqual(
            sorted(_event_scoped_targets(execution, {"evt-a", "evt-b"})),
            ["a2", "a3", "b2"],
        )

    def test_correction_targets_are_scoped_to_the_candidate_chain(self) -> None:
        """A chain the candidates are not on explains nothing.

        The correction trace reports its added filings as one list, so they are
        attributable only when a single chain expanded and the trace names it.
        Anything else stays unexplained rather than being assumed harmless.
        """

        def execution(groups, added):
            return SimpleNamespace(correction_expansion=correction_trace(
                "latest", "expanded", groups=groups, added_doc_ids=added))

        group_a = {"correction_group_id": "group-a", "root_doc_id": "dROOT",
                   "latest_doc_id": "dLATEST"}
        group_b = {"correction_group_id": "group-b", "root_doc_id": "dR2",
                   "latest_doc_id": "dL2"}

        self.assertEqual(
            sorted(_correction_scoped_targets(
                execution([group_a], ["dLATEST"]), {"group-a"})),
            ["dLATEST"],
        )
        # A different chain is a different contract.
        self.assertEqual(
            _correction_scoped_targets(execution([group_b], ["dL2"]), {"group-a"}),
            set(),
        )
        # Two chains expanded: the list cannot be attributed to either.
        self.assertEqual(
            _correction_scoped_targets(
                execution([group_a, group_b], ["dLATEST", "dL2"]), {"group-a"}),
            set(),
        )
        self.assertEqual(
            _correction_scoped_targets(execution([group_a], ["dLATEST"]), set()),
            set(),
        )

    def test_a_correction_member_of_another_chain_fails_closed(self) -> None:
        served = (
            self._amount("cA", "dA", "184,000,000,000"),
            self._amount("cB", "dB", "99,000,000", table_id="t0002"),
        )

        records = _corporate_evidence(
            self.QUESTION, served,
            states=(_state("dA", group="group-a", correction_status=RESOLVED,
                           member_count=2),),
            seeds=["dA"],
            correction=correction_trace(
                "latest", "expanded",
                groups=[{"correction_group_id": "group-b", "root_doc_id": "dR2",
                         "latest_doc_id": "dB"}],
                added_doc_ids=["dB"],
            ),
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])

    def test_expansion_membership_never_grants_field_authority(self) -> None:
        """Explained means "not a second event", never "may answer".

        The superseded root is on the carried event and its own chain, so it is
        explained -- and finality still decides that only the canonical filing
        may supply the value.
        """

        root = _contract_chunk("cROOT", "dROOT", amount="100,000,000")

        records = _corporate_evidence(
            self.QUESTION, (root,),
            states=(_state("dROOT", canonical="dLATEST", group="g1",
                           correction_status=RESOLVED, member_count=2),),
            seeds=["dROOT"],
            expanded_events=({"event_id": EVENT, "target_doc_ids": ["dLATEST"]},),
            plan=_corporate_plan(self.QUESTION, exact_receipt=False),
        )

        authoritative = [record for record in records if record.authoritative]
        self.assertEqual(len(authoritative), 1)
        self.assertIs(authoritative[0].status, FieldStatus.MISSING)
        self.assertEqual(authoritative[0].doc_id, "dLATEST")
        self.assertNotIn("100,000,000", [r.value for r in authoritative])

    def test_an_unexplained_extra_filing_still_fails_closed(self) -> None:
        """Mutation control for the expansion allowance above."""

        root = _contract_chunk("cROOT", "dROOT", amount="-")
        stranger = _contract_chunk("cX", "dX", amount="184,000,000,000",
                                   table_id="t0002")

        records = _corporate_evidence(
            self.QUESTION, (root, stranger),
            states=(_state("dROOT"),), seeds=["dROOT"],
        )

        self.assertEqual([record.status for record in records],
                         [FieldStatus.CONFLICT])

    def test_the_final_answer_is_not_answerable_from_the_surviving_candidate(
        self,
    ) -> None:
        """End to end: the collapsed answer must not reach the caller."""

        served = [
            self._amount("cA", "dA", "184,000,000,000"),
            self._amount("cB", "dB", "99,000,000", table_id="t0002"),
        ]
        pairs = [
            (
                CandidateChunk(str(c["chunk_id"]), str(c["doc_id"]), c, MetadataMatch()),
                RetrievalResult(str(c["chunk_id"]), str(c["doc_id"]), 1.0 / i, i, {}),
            )
            for i, c in enumerate(served, start=1)
        ]
        plan = _corporate_plan(self.QUESTION, exact_receipt=False)
        execution = SimpleNamespace(
            plan=plan, chunks=[candidate for candidate, _ in pairs],
            results=[result for _, result in pairs],
            event_expansion=_execution(states=(_state("dB", event_id="evt-b"),),
                                       seeds=["dB"]).event_expansion,
            correction_expansion={},
        )
        result = AgentOrchestrator().run(self.QUESTION, plan, execution)
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated, plan=plan, agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        answer = guarded_answer_text(verdict, generated.answer_text)

        self.assertFalse(verdict.answerable)
        self.assertNotIn("99,000,000", answer)
        self.assertNotIn("184,000,000,000", answer)


class P1_5_HoldingHeaderIdentityTests(unittest.TestCase):
    """The unit-price column is chosen by header meaning, never by position."""

    def _merged(self, text="취득/처분단가**"):
        return [[
            {"text": "성명(명칭)", "colspan": 1, "rowspan": 1, "is_header": True},
            {"text": "변동일", "colspan": 1, "rowspan": 1, "is_header": True},
            {"text": "취득/처분방법", "colspan": 1, "rowspan": 1, "is_header": True},
            {"text": "증감", "colspan": 1, "rowspan": 1, "is_header": True},
            {"text": text, "colspan": 2, "rowspan": 1, "is_header": True},
            {"text": "비 고", "colspan": 1, "rowspan": 1, "is_header": True},
        ]]

    def test_a_distinguished_acquisition_header_is_selected(self) -> None:
        headers = ["성명", "변동일", "방법", "증감",
                   "단가 / 취득", "단가 / 처분", "비 고"]

        self.assertEqual(_unit_price_column(headers), 4)

    def test_column_order_does_not_change_the_selection(self) -> None:
        headers = ["성명", "변동일", "방법", "증감",
                   "단가 / 처분", "단가 / 취득", "비 고"]

        self.assertEqual(_unit_price_column(headers), 5)

    def test_a_single_generic_unit_price_column_is_accepted(self) -> None:
        headers = ["성명", "변동일", "방법", "증감", "취득단가", "비 고"]

        self.assertEqual(_unit_price_column(headers), 4)

    def test_one_merged_header_cell_states_one_field(self) -> None:
        headers = ["성명", "변동일", "취득/처분방법", "증감",
                   "취득/처분단가**", "취득/처분단가**", "비 고"]

        self.assertEqual(_unit_price_column(headers, self._merged()), 4)

    def test_repeated_headers_that_are_not_one_merged_cell_fail_closed(self) -> None:
        """Same text, separate physical cells: nothing says which is which."""

        headers = ["성명", "변동일", "취득/처분방법", "증감",
                   "취득/처분단가**", "취득/처분단가**", "비 고"]
        separate = [[
            {"text": value, "colspan": 1, "rowspan": 1, "is_header": True}
            for value in headers
        ]]

        self.assertIsNone(_unit_price_column(headers, separate))

    def test_two_distinct_ambiguous_price_columns_fail_closed(self) -> None:
        headers = ["성명", "변동일", "방법", "증감", "단가 A", "단가 B", "비 고"]

        self.assertIsNone(_unit_price_column(headers))

    def test_no_price_column_at_all(self) -> None:
        self.assertIsNone(_unit_price_column(["성명", "변동일", "방법", "증감"]))


class P1_6_MultiDocumentPrecedenceTests(unittest.TestCase):
    """A supported single field never answers an incomplete document set."""

    def setUp(self) -> None:
        self.guard = AnswerabilityGuard()
        self.generated = _generated()
        self.execution = _served("c1")
        self.record = _record()

    def _incomplete(self):
        return MultiDocumentEvidence(facts=MultiDocumentFacts(
            plan_type="enumeration", complete=False, logical_count=8,
            unresolved_count=3,
        ))

    def test_m1_incomplete_set_outranks_a_supported_field(self) -> None:
        result = self.guard.evaluate(
            self.generated,
            agent_result=SimpleNamespace(field_evidence=(self.record,),
                                         resolution=None),
            execution=self.execution,
            multi_document=self._incomplete(),
        )

        self.assertIs(result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.reason, "incomplete_multi_document_set")
        self.assertFalse(result.answerable)

    def test_the_verdict_matches_p0c_alone(self) -> None:
        """Adding field evidence changes nothing about a blocking P0-C verdict."""

        alone = self.guard.evaluate(
            self.generated, agent_result=SimpleNamespace(resolution=None),
            execution=self.execution, multi_document=self._incomplete(),
        )
        with_field = self.guard.evaluate(
            self.generated,
            agent_result=SimpleNamespace(field_evidence=(self.record,),
                                         resolution=None),
            execution=self.execution, multi_document=self._incomplete(),
        )

        self.assertEqual(alone.to_public_dict(), with_field.to_public_dict())

    def test_a_complete_set_still_reaches_the_field_lane(self) -> None:
        complete = MultiDocumentEvidence(facts=MultiDocumentFacts(
            plan_type="enumeration", complete=True, logical_count=3,
        ))
        result = self.guard.evaluate(
            self.generated,
            agent_result=SimpleNamespace(
                field_evidence=(
                    _record(status=FieldStatus.UNAVAILABLE, value=None,
                            reason=FieldReason.NOT_STATED),
                ),
                resolution=None,
            ),
            execution=self.execution, multi_document=complete,
        )

        self.assertEqual(result.unavailable_fields, (CONTRACT_AMOUNT,))
        self.assertFalse(result.answerable)

    def test_m2_an_ordinary_single_event_request_still_works(self) -> None:
        result = self.guard.evaluate(
            self.generated,
            agent_result=SimpleNamespace(field_evidence=(self.record,),
                                         resolution=None),
            execution=self.execution,
        )

        self.assertIs(result.status, AnswerabilityStatus.ANSWERABLE)

    def test_m3_the_corporate_producer_declines_a_set_request(self) -> None:
        records = _corporate_evidence(
            "2024년 5월 20일 공시된 계약들 중에서 계약금액은 얼마인가요?",
            (_contract_chunk("c1", "d1", amount="184,000,000,000"),),
            states=(_state("d1"),),
            multi_document=self._incomplete(),
        )

        self.assertEqual(records, ())


# ================================================== end-to-end pipeline shape


class OrdinaryQuestionEndToEndTests(unittest.TestCase):
    """The real seam, not a hand-written trace.

    Correction safety has to hold for a question that never says 정정, so this
    drives the actual expansion writer and the actual orchestrator rather than
    a metadata shape invented for the test.
    """

    QUESTION = "계약금액은 얼마인가요?"

    def _answer(self, chunks, states):
        pairs = [
            (
                CandidateChunk(str(c["chunk_id"]), str(c["doc_id"]), c, MetadataMatch()),
                RetrievalResult(str(c["chunk_id"]), str(c["doc_id"]), 1.0 / i, i, {}),
            )
            for i, c in enumerate(chunks, start=1)
        ]
        plan = _corporate_plan(self.QUESTION, exact_receipt=False)
        seeds = [str(c["doc_id"]) for c in chunks]
        execution = SimpleNamespace(
            plan=plan,
            chunks=[candidate for candidate, _ in pairs],
            results=[result for _, result in pairs],
            event_expansion=_execution(states=states, seeds=seeds).event_expansion,
            correction_expansion={},
        )
        result = AgentOrchestrator().run(self.QUESTION, plan, execution)
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated, plan=plan, agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        return verdict, guarded_answer_text(verdict, generated.answer_text)

    def test_a_stale_correction_member_cannot_answer_without_the_word_correction(
        self,
    ) -> None:
        self.assertNotIn("정정", self.QUESTION)
        verdict, answer = self._answer(
            [_contract_chunk("cOLD", "dROOT", amount="100,000,000")],
            (_state("dROOT", canonical="dLATEST", group="g1",
                    correction_status=RESOLVED, member_count=2),),
        )

        self.assertFalse(verdict.answerable)
        self.assertEqual(verdict.missing_fields, (CONTRACT_AMOUNT,))
        self.assertNotIn("100,000,000", answer)

    def test_an_ordinary_standalone_filing_answers(self) -> None:
        verdict, _answer = self._answer(
            [_contract_chunk("c1", "d1", amount="184,000,000,000")],
            (_state("d1"),),
        )

        self.assertTrue(verdict.answerable)
        self.assertEqual(verdict.confirmed_fields, (CONTRACT_AMOUNT,))

    def test_the_measured_t18_corporate_shape_stays_closed(self) -> None:
        """Blank formal field, real nearby numbers, a later filing served."""

        verdict, answer = self._answer(
            [
                _contract_chunk("cORIG", "dORIG", amount="-"),
                _contract_chunk("cLATER", "dLATER", amount="205,000,000,000",
                                rcept_dt=LATER_RECEIPT, table_id="t0002"),
            ],
            (_state("dORIG"), _state("dLATER", role=ROLE_TERMINATION)),
        )

        self.assertFalse(verdict.answerable)
        self.assertEqual(verdict.unavailable_fields, (CONTRACT_AMOUNT,))
        self.assertIsNotNone(verdict.refusal_citation)
        self.assertIn(verdict.refusal_citation, answer)
        self.assertNotIn("1,482,000,000,000", answer)
        self.assertNotIn("205,000,000,000", answer)


# ------------------------------------------------------------- holding side


HOLDING_HEADERS = [
    "성명(명칭)", "생년월일 또는사업자등록번호 등", "변동일*", "취득/처분방법",
    "주식등의종류", "변동 내역 / 변동전", "변동 내역 / 증감", "변동 내역 / 변동후",
    "취득/처분단가**", "취득/처분단가**", "비 고",
]

HOLDING_HEADER_ROWS = [[
    {"text": value, "colspan": 1, "rowspan": 1, "is_header": True}
    for value in HOLDING_HEADERS[:8]
] + [
    {"text": "취득/처분단가**", "colspan": 2, "rowspan": 1, "is_header": True},
    {"text": "비 고", "colspan": 1, "rowspan": 1, "is_header": True},
]]

ACQUIRED_WITH_PRICE = _row(
    "성명주식회사", "120-86-78223", "2024.03.07", "장내매수(+)",
    "의결권있는 주식", "2,098,811", "868,948", "2,967,759", "120,000", "-", "-",
)
ACQUIRED_BLANK_PRICE = _row(
    "성명주식회사", "120-86-78223", "2024.03.07", "장내매수(+)",
    "의결권있는 주식", "2,098,811", "868,948", "2,967,759", "-", "-", "-",
)
ACQUIRED_PRICE_OMITTED = _row(
    "성명주식회사", "120-86-78223", "2024.03.07", "장내매수(+)",
    "의결권있는 주식", "2,098,811", "868,948", "2,967,759", "-", "-",
    "최초 보고이므로 취득단가 미기재",
)
DISPOSED_WITH_PRICE = _row(
    "성명주식회사", "120-86-78223", "2024.03.07", "장내매도(-)",
    "의결권있는 주식", "5,635,483", "-2,320,493", "3,314,990", "9,000", "-", "-",
)


def _detail_chunk(
    chunk_id: str,
    doc_id: str,
    row: list[dict[str, object]],
    *,
    table_id: str = "t0019",
    row_index: int = 2,
    headers: list[str] | None = None,
    header_rows: list | None = None,
    corp_code: str = CORP,
) -> dict[str, object]:
    cells = [str(cell["text"]) for cell in row]
    labels = ("보고자/보유자", "기준일/보고일", "직전 보유주식수", "증감주식수",
              "보유주식수")
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": corp_code,
        "corp_name": "예시전자",
        "doc_group": "holding",
        "chunk_type": "table_projection",
        "rcept_dt": "20240314",
        "section_path": ["세부변동내역"],
        "content": "세부 변동 내역",
        "retrieval_text": "세부 변동 내역",
        "projection_type": "holding_detail_row",
        "table_id": table_id,
        "source_table_id": table_id,
        "row_start": row_index,
        "row_end": row_index,
        "column_headers": HOLDING_HEADERS if headers is None else headers,
        "header_rows": HOLDING_HEADER_ROWS if header_rows is None else header_rows,
        "table_rows": [row],
        "source_refs": [{"table_id": table_id, "row_start": row_index,
                         "row_end": row_index}],
        "projection_fields": dict(
            zip(labels, (cells[0], cells[2], cells[5], cells[6], cells[7]))
        ),
        "projection_field_refs": {
            label: [{"table_id": table_id, "row_start": row_index,
                     "row_end": row_index}]
            for label in labels
        },
    }


def _report_chunk(
    chunk_id: str,
    doc_id: str,
    *,
    corp_code: str = CORP,
    reporter: str = "성명주식회사",
    reference_date: str = "2025-08-04",
) -> dict[str, object]:
    fields = {
        "보고자/보유자": reporter,
        "기준일/보고일": reference_date,
        "직전 보유주식수": "62,792,705",
        "증감주식수": "3,333",
        "보유주식수": "62,796,038",
        "보유비율": "30.67",
    }
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": corp_code,
        "corp_name": "예시전자",
        "doc_group": "holding",
        "chunk_type": "table_projection",
        "rcept_dt": reference_date.replace("-", ""),
        "section_path": ["주식등의 대량보유상황보고서"],
        "content": "보고자 기준일 직전 보유주식수 증감주식수 보유주식수 보유비율",
        "retrieval_text": "보고자 기준일 보유주식수 보유비율",
        "projection_type": "holding_report",
        "table_id": "t0013",
        "source_table_id": "t0013",
        "row_start": 3,
        "row_end": 3,
        "projection_fields": fields,
        "projection_field_refs": {
            label: [{"table_id": "t0013", "row_start": 3, "row_end": 3}]
            for label in fields
        },
        "source_refs": [{"table_id": "t0013", "row_start": 3, "row_end": 3}],
    }


def _report_field_evidence(
    question: str,
    chunk: dict[str, object],
    *,
    corp_code: str = CORP,
    reporter: str = "성명주식회사",
    day: str = "2025-08-04",
):
    candidate = CandidateChunk(
        str(chunk["chunk_id"]), str(chunk["doc_id"]), chunk, MetadataMatch()
    )
    result = RetrievalResult(
        str(chunk["chunk_id"]), str(chunk["doc_id"]), 1.0, 1, {}
    )
    plan = QueryPlan(
        query=question,
        raw_query=question,
        company="예시전자",
        corp_code=corp_code,
        task_type="holding_change",
        metric=ACQUISITION_UNIT_PRICE,
        reporter=reporter,
        period=QueryPeriod(
            from_date=day,
            to_date=day,
            period_type="holding_reference_date",
        ),
    )
    evidence = build_evidence_set(
        question=question,
        query_plan=plan,
        candidates=[candidate],
        results=[result],
        grouping_intent="holding_change",
    )
    resolution = resolve_holding_events(evidence, query_plan=plan)
    records = holding_field_evidence(
        question=question,
        plan=plan,
        resolution=resolution,
        evidence_items=evidence.served_items,
    )
    return records, resolution, plan, candidate, result


def _holding(question: str, chunks: tuple[dict[str, object], ...], *,
             reporter: str | None = None, day: str | None = None):
    pairs = [
        (
            CandidateChunk(str(chunk["chunk_id"]), str(chunk["doc_id"]), chunk,
                           MetadataMatch()),
            RetrievalResult(str(chunk["chunk_id"]), str(chunk["doc_id"]),
                            1.0 / index, index, {}),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    plan = QueryPlan(
        query=question,
        raw_query=question,
        task_type="holding_change",
        reporter=reporter,
        period=(
            None
            if day is None
            else QueryPeriod(from_date=day, to_date=day,
                             period_type="holding_reference_date")
        ),
    )
    evidence = build_evidence_set(
        question=question,
        query_plan=plan,
        candidates=[candidate for candidate, _ in pairs],
        results=[result for _, result in pairs],
        grouping_intent="holding_change",
    )
    resolution = resolve_holding_events(evidence, query_plan=plan)
    return holding_field_evidence(
        question=question,
        resolution=resolution,
        evidence_items=evidence.served_items,
    ), resolution


class HoldingRequestedFieldContractTests(unittest.TestCase):
    def test_unit_price_wording_activates(self) -> None:
        for question in ("취득단가는 얼마인가요?", "취득 단가를 알려주세요"):
            self.assertEqual(
                requested_holding_fields(question), (ACQUISITION_UNIT_PRICE,), question
            )

    def test_h8_existing_acquisition_questions_do_not_activate(self) -> None:
        for question in ("취득일은 언제인가요?", "취득 수량은 얼마인가요?", "취득한 주식 수는?"):
            self.assertEqual(requested_holding_fields(question), (), question)


class HoldingFieldEvidenceTests(unittest.TestCase):
    QUESTION = "취득단가는 얼마인가요?"

    def test_h1_a_row_that_states_its_omission_is_omitted(self) -> None:
        records, _ = _holding(
            self.QUESTION, (_detail_chunk("h1", "d1", ACQUIRED_PRICE_OMITTED),)
        )

        self.assertEqual(len(records), 1)
        self.assertIs(records[0].status, FieldStatus.UNAVAILABLE)
        self.assertIs(records[0].reason, FieldReason.OMITTED)
        self.assertEqual(records[0].chunk_id, "h1")
        self.assertEqual(records[0].table_id, "t0019")
        self.assertEqual(records[0].row_start, 2)

    def test_h2_a_real_unit_price_is_available(self) -> None:
        records, _ = _holding(
            self.QUESTION, (_detail_chunk("h1", "d1", ACQUIRED_WITH_PRICE),)
        )

        self.assertIs(records[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(records[0].value, "120,000")

    def test_h3_a_blank_unit_price_without_a_reason_is_not_stated(self) -> None:
        records, _ = _holding(
            self.QUESTION, (_detail_chunk("h1", "d1", ACQUIRED_BLANK_PRICE),)
        )

        self.assertIs(records[0].status, FieldStatus.UNAVAILABLE)
        self.assertIs(records[0].reason, FieldReason.NOT_STATED)

    def test_h4_an_omission_note_in_another_report_does_not_reach_this_row(self) -> None:
        """The other report explains its own blank, not this one."""

        selected = _detail_chunk("h1", "d1", ACQUIRED_BLANK_PRICE)
        other_row = list(ACQUIRED_PRICE_OMITTED)
        other_row[2] = _cell("2023.11.02")
        elsewhere = _detail_chunk("h2", "d2", other_row, table_id="t0031", row_index=5)

        records, _ = _holding(
            "2024년 3월 7일 취득단가는 얼마인가요?", (selected, elsewhere),
            day="2024-03-07",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].chunk_id, "h1")
        self.assertIs(records[0].status, FieldStatus.UNAVAILABLE)
        self.assertIs(records[0].reason, FieldReason.NOT_STATED)

    def test_h5_unrelated_holding_citations_do_not_change_the_state(self) -> None:
        selected = _detail_chunk("h1", "d1", ACQUIRED_BLANK_PRICE)
        unrelated = dict(_detail_chunk("h2", "d1", ACQUIRED_WITH_PRICE,
                                       table_id="t0044", row_index=9))
        unrelated["projection_type"] = "holding_report"

        alone, _ = _holding(self.QUESTION, (selected,))
        with_noise, _ = _holding(self.QUESTION, (selected, unrelated))

        self.assertEqual(
            [record.outcome for record in alone],
            [record.outcome for record in with_noise],
        )

    def test_h6_a_reporter_that_does_not_match_emits_nothing(self) -> None:
        records, resolution = _holding(
            self.QUESTION,
            (_detail_chunk("h1", "d1", ACQUIRED_BLANK_PRICE),),
            reporter="다른보고자",
        )

        self.assertEqual(records, ())
        self.assertTrue(
            all(event.matches_query is not True for event in resolution.events)
        )

    def test_h7_two_acquisitions_answering_one_question_fail_closed(self) -> None:
        first = _detail_chunk("h1", "d1", ACQUIRED_WITH_PRICE)
        second_row = _row(
            "다른보고자", "110-11-11111", "2024.03.07", "장내매수(+)",
            "의결권있는 주식", "1,000", "500", "1,500", "99,000", "-", "-",
        )
        second = _detail_chunk("h2", "d2", second_row, table_id="t0021", row_index=4)

        records, _ = _holding(self.QUESTION, (first, second))

        self.assertEqual(records, ())

    def test_a_disposal_row_never_answers_for_an_acquisition_price(self) -> None:
        records, _ = _holding(
            self.QUESTION, (_detail_chunk("h1", "d1", DISPOSED_WITH_PRICE),)
        )

        self.assertEqual(records, ())

    def test_an_absent_unit_price_column_is_missing(self) -> None:
        headers = [header for header in HOLDING_HEADERS if "단가" not in header]
        row = [
            cell for index, cell in enumerate(ACQUIRED_WITH_PRICE)
            if "단가" not in HOLDING_HEADERS[index]
        ]

        records, _ = _holding(
            self.QUESTION,
            (_detail_chunk("h1", "d1", row, headers=headers, header_rows=[]),),
        )

        self.assertEqual(len(records), 1)
        self.assertIs(records[0].status, FieldStatus.MISSING)
        self.assertIsNone(records[0].chunk_id)

    def test_a_distinguished_acquisition_column_is_read(self) -> None:
        """Multi-row headers, and the acquisition price is not the first 단가."""

        headers = [*HOLDING_HEADERS[:8], "단가 / 처분", "단가 / 취득", "비 고"]
        row = _row(
            "성명주식회사", "120-86-78223", "2024.03.07", "장내매수(+)",
            "의결권있는 주식", "2,098,811", "868,948", "2,967,759",
            "9,000", "120,000", "-",
        )

        records, _ = _holding(
            self.QUESTION,
            (_detail_chunk("h1", "d1", row, headers=headers, header_rows=[]),),
        )

        self.assertIs(records[0].status, FieldStatus.AVAILABLE)
        self.assertEqual(records[0].value, "120,000")

    def test_c094_exact_report_without_unit_price_is_grounded_unavailable(self) -> None:
        records, _resolution, _plan, _candidate, _result = _report_field_evidence(
            "성명주식회사가 보유한 예시전자 주식의 "
            "2025년 8월 4일 기준 취득 단가는?",
            _report_chunk("h-target", "holding-target"),
        )

        self.assertEqual(len(records), 1)
        self.assertIs(records[0].status, FieldStatus.UNAVAILABLE)
        self.assertIs(records[0].reason, FieldReason.NOT_STATED)
        self.assertEqual(records[0].doc_id, "holding-target")
        self.assertEqual(records[0].chunk_id, "h-target")
        self.assertEqual(records[0].table_id, "t0013")

    def test_report_identity_mismatch_is_missing_and_never_negative(self) -> None:
        records, _resolution, _plan, _candidate, _result = _report_field_evidence(
            "성명주식회사가 보유한 예시전자 주식의 "
            "2025년 8월 4일 기준 취득 단가는?",
            _report_chunk(
                "h-wrong", "holding-wrong", reporter="다른보고자"
            ),
        )

        self.assertEqual(len(records), 1)
        self.assertIs(records[0].status, FieldStatus.MISSING)
        self.assertIsNone(records[0].chunk_id)


class HoldingNegativeCitationTests(unittest.TestCase):
    """The holding refusal is grounded through the same shared plumbing."""

    QUESTION = "취득단가는 얼마인가요?"

    def test_the_measured_t18_holding_shape_stays_closed(self) -> None:
        chunks = [_detail_chunk("hSEL", "dSEL", ACQUIRED_PRICE_OMITTED)]
        pairs = [
            (
                CandidateChunk(str(c["chunk_id"]), str(c["doc_id"]), c, MetadataMatch()),
                RetrievalResult(str(c["chunk_id"]), str(c["doc_id"]), 1.0 / i, i, {}),
            )
            for i, c in enumerate(chunks, start=1)
        ]
        plan = QueryPlan(query=self.QUESTION, raw_query=self.QUESTION,
                         task_type="holding_change")
        execution = SimpleNamespace(
            plan=plan, chunks=[candidate for candidate, _ in pairs],
            results=[result for _, result in pairs],
        )
        result = AgentOrchestrator().run(self.QUESTION, plan, execution)
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated, plan=plan, agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        answer = guarded_answer_text(verdict, generated.answer_text)

        self.assertFalse(verdict.answerable)
        self.assertEqual(verdict.unavailable_fields, (ACQUISITION_UNIT_PRICE,))
        self.assertIsNotNone(verdict.refusal_citation)
        self.assertIn(verdict.refusal_citation, answer)
        marker = {c.citation_id: c.chunk_id for c in generated.citations}
        self.assertEqual(marker[verdict.refusal_citation], "hSEL")
        self.assertIn("기재되지 않은 것으로 명시", answer)

    def test_c094_report_bound_refusal_keeps_the_target_citation(self) -> None:
        question = (
            "성명주식회사가 보유한 예시전자 주식의 "
            "2025년 8월 4일 기준 취득 단가는?"
        )
        source = _report_chunk("h-c094", "holding-c094")
        source.pop("projection_type")
        source["projection_fields"] = {}
        source["projection_field_refs"] = {}
        _records, _resolution, plan, candidate, retrieval = _report_field_evidence(
            question, source
        )
        execution = SimpleNamespace(
            plan=plan,
            chunks=(candidate,),
            results=(retrieval,),
            routing={},
            correction_expansion={},
            event_expansion={},
        )

        selected = SimpleNamespace(
            issuer_corp_code=CORP,
            reporter_key="성명주식회사",
            raw_reporter="성명주식회사",
            reference_date="20250804",
            doc_id="holding-c094",
            projection_chunk_id="not-served",
        )
        index = SimpleNamespace(
            select_report=lambda *args, **kwargs: SimpleNamespace(
                resolved=True, selected=selected
            )
        )
        result = AgentOrchestrator(holding_report_index=index).run(
            question, plan, execution
        )
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated,
            plan=plan,
            agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        answer = guarded_answer_text(verdict, generated.answer_text)

        self.assertFalse(verdict.answerable)
        self.assertIn("no_holding_event_groups", result.warnings)
        self.assertEqual(verdict.unavailable_fields, (ACQUISITION_UNIT_PRICE,))
        self.assertIsNotNone(verdict.refusal_citation)
        self.assertIn(verdict.refusal_citation, answer)
        marker = {citation.citation_id: citation.chunk_id for citation in generated.citations}
        self.assertEqual(marker[verdict.refusal_citation], "h-c094")

    def test_m4_several_reports_cannot_be_answered_as_one(self) -> None:
        """Report-relative and multi-report safety is not bypassed."""

        first = _detail_chunk("h1", "d1", ACQUIRED_WITH_PRICE)
        second_row = list(ACQUIRED_WITH_PRICE)
        second_row[2] = _cell("2023.11.02")
        second = _detail_chunk("h2", "d2", second_row, table_id="t0021", row_index=4)

        records, resolution = _holding(self.QUESTION, (first, second))

        self.assertEqual(records, ())
        self.assertTrue(resolution.temporal_ambiguity)


class HoldingFrozenSemanticsTests(unittest.TestCase):
    """H8.  The acquisition fields this producer sits beside are untouched."""

    def test_acquisition_date_and_shares_keep_their_values(self) -> None:
        _records, resolution = _holding(
            "취득일과 취득 수량은?", (_detail_chunk("h1", "d1", ACQUIRED_WITH_PRICE),)
        )
        event = next(
            event for event in resolution.events if event.matches_query is not False
        )

        self.assertEqual(event.acquisition_date, "2024-03-07")
        self.assertEqual(event.transaction_method, "장내매수(+)")
        self.assertEqual(event.acquired_shares.raw, "868,948")


# --------------------------------------------------------- served-set model


def _record(
    field: str = CONTRACT_AMOUNT,
    *,
    status: FieldStatus = FieldStatus.AVAILABLE,
    chunk_id: str | None = "c1",
    semantic_key: str = "k",
    reason: FieldReason | None = None,
    value: str | None = "1,000",
) -> FieldEvidence:
    return FieldEvidence(
        field=field,
        status=status,
        domain=DOMAIN_CORPORATE_EVENT,
        semantic_key=semantic_key,
        reason=reason,
        value=value if status is FieldStatus.AVAILABLE else None,
        chunk_id=chunk_id,
    )


class FieldEvidenceModelTests(unittest.TestCase):
    def test_a_stated_value_must_name_its_source(self) -> None:
        with self.assertRaises(ValueError):
            _record(chunk_id=None)

    def test_a_stated_blank_must_name_its_source(self) -> None:
        """A grounded negative is a reading of a source, so it needs one."""

        with self.assertRaises(ValueError):
            _record(status=FieldStatus.UNAVAILABLE, chunk_id=None, value=None,
                    reason=FieldReason.NOT_STATED)

    def test_an_absence_may_omit_a_source(self) -> None:
        record = _record(status=FieldStatus.MISSING, chunk_id=None, value=None)

        self.assertIsNone(record.chunk_id)

    def test_only_an_unavailable_field_carries_a_reason(self) -> None:
        with self.assertRaises(ValueError):
            FieldEvidence(
                field=CONTRACT_AMOUNT, status=FieldStatus.MISSING,
                domain=DOMAIN_CORPORATE_EVENT, semantic_key="k",
                reason=FieldReason.NOT_STATED,
            )


class ServedMembershipTests(unittest.TestCase):
    """P1-F.  Membership is tested against the served set, always."""

    def test_a_record_outside_the_served_set_is_rejected(self) -> None:
        self.assertEqual(
            accepted_field_evidence((_record(chunk_id="c9"),), {"c1", "c2"}), ()
        )

    def test_an_empty_served_set_accepts_nothing(self) -> None:
        for record in (
            _record(),
            _record(status=FieldStatus.MISSING, chunk_id=None, value=None),
            _record(status=FieldStatus.UNAVAILABLE, chunk_id="c1", value=None,
                    reason=FieldReason.NOT_STATED),
        ):
            self.assertEqual(accepted_field_evidence((record,), ()), (), record.status)

    def test_an_unserved_negative_cannot_ground_a_refusal_either(self) -> None:
        record = _record(status=FieldStatus.UNAVAILABLE, chunk_id="c9", value=None,
                         reason=FieldReason.NOT_STATED)

        self.assertEqual(accepted_field_evidence((record,), {"c1"}), ())

    def test_an_absence_is_accepted_when_something_was_served(self) -> None:
        record = _record(status=FieldStatus.MISSING, chunk_id=None, value=None)

        self.assertEqual(accepted_field_evidence((record,), {"c1"}), (record,))

    def test_a_non_authoritative_record_never_decides(self) -> None:
        superseded = FieldEvidence(
            field=CONTRACT_AMOUNT, status=FieldStatus.AVAILABLE,
            domain=DOMAIN_CORPORATE_EVENT, semantic_key="k", value="1,000",
            chunk_id="c1", authoritative=False,
        )

        self.assertEqual(accepted_field_evidence((superseded,), {"c1"}), ())

    def test_two_identities_for_one_field_conflict(self) -> None:
        states = resolve_field_states(
            (
                _record(semantic_key="event-a", chunk_id="c1"),
                _record(semantic_key="event-b", chunk_id="c2", value="2,000"),
            ),
            served={"c1", "c2"},
        )

        self.assertIs(states[CONTRACT_AMOUNT].status, FieldStatus.CONFLICT)

    def test_one_identity_stated_twice_is_one_finding(self) -> None:
        states = resolve_field_states(
            (_record(chunk_id="c1"), _record(chunk_id="c2")), served={"c1", "c2"}
        )

        self.assertIs(states[CONTRACT_AMOUNT].status, FieldStatus.AVAILABLE)


class FieldInstanceProvenanceTests(unittest.TestCase):
    """Two instances of one field in one filing are two instances."""

    QUESTION = "계약금액은 얼마인가요?"

    def test_each_amount_row_is_bound_to_its_own_row(self) -> None:
        chunk = _contract_chunk("c1", "d1", amount="-")
        chunk["table_rows"] = [
            *list(chunk["table_rows"]),
            _row("9. 계약금액(원)", "184,000,000,000"),
        ]

        records = _corporate_evidence(self.QUESTION, (chunk,), states=(_state("d1"),))
        rows = sorted(record.row_start for record in records)

        self.assertEqual(len(records), 2)
        # Distinct rows, not the chunk's whole span repeated.
        self.assertEqual(rows, [2, 7])
        self.assertEqual(len({record.instance_key for record in records}), 2)

    def test_distinct_instances_that_disagree_conflict(self) -> None:
        chunk = _contract_chunk("c1", "d1", amount="-")
        chunk["table_rows"] = [
            *list(chunk["table_rows"]),
            _row("9. 계약금액(원)", "184,000,000,000"),
        ]

        records = _corporate_evidence(self.QUESTION, (chunk,), states=(_state("d1"),))
        states = resolve_field_states(records, served={"c1"})

        self.assertIs(states[CONTRACT_AMOUNT].status, FieldStatus.CONFLICT)


# ------------------------------------------------------------ guard behaviour


def _generated(text: str = "확인된 사실입니다.", *, answerable: bool = True,
               chunk_ids: tuple[str, ...] = ("c1",)):
    return SimpleNamespace(
        answerable=answerable,
        answer_text=text,
        citations=tuple(
            SimpleNamespace(chunk_id=chunk_id, citation_id=f"[{index}]")
            for index, chunk_id in enumerate(chunk_ids, start=1)
        ),
    )


def _served(*chunk_ids: str):
    return SimpleNamespace(
        results=tuple(SimpleNamespace(chunk_id=chunk_id) for chunk_id in chunk_ids)
    )


class AnswerabilityFieldLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = AnswerabilityGuard()

    def _evaluate(self, records, *, generated=None, served=("c1",)):
        return self.guard.evaluate(
            generated if generated is not None else _generated(),
            agent_result=SimpleNamespace(field_evidence=tuple(records),
                                         resolution=None),
            execution=_served(*served),
        )

    def test_a1_an_unavailable_field_overrides_a_citable_positive_answer(self) -> None:
        result = self._evaluate(
            (_record(status=FieldStatus.UNAVAILABLE, value=None,
                     reason=FieldReason.NOT_STATED),),
            generated=_generated("계약금액은 1,482억원입니다. [1]"),
        )

        self.assertFalse(result.answerable)
        self.assertEqual(result.confirmed_fields, ())
        self.assertEqual(result.unavailable_fields, (CONTRACT_AMOUNT,))

    def test_a2_a_plausible_number_in_prose_is_not_field_support(self) -> None:
        result = self._evaluate(
            (_record(status=FieldStatus.UNAVAILABLE, value=None,
                     reason=FieldReason.NOT_STATED),),
            generated=_generated("계약금액은 약 1,800억원 규모입니다. [1]"),
        )

        self.assertIs(result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)
        self.assertNotIn(CONTRACT_AMOUNT, result.confirmed_fields)

    def test_a3_an_available_field_is_answerable(self) -> None:
        result = self._evaluate((_record(),))

        self.assertIs(result.status, AnswerabilityStatus.ANSWERABLE)
        self.assertEqual(result.confirmed_fields, (CONTRACT_AMOUNT,))

    def test_a4_a_missing_field_is_not_answerable(self) -> None:
        result = self._evaluate(
            (_record(status=FieldStatus.MISSING, chunk_id=None, value=None),)
        )

        self.assertFalse(result.answerable)
        self.assertEqual(result.missing_fields, (CONTRACT_AMOUNT,))

    def test_a5_a_conflict_is_not_answerable(self) -> None:
        result = self._evaluate(
            (_record(semantic_key="a"), _record(semantic_key="b", chunk_id="c2")),
            served=("c1", "c2"),
        )

        self.assertFalse(result.answerable)

    def test_a6_evidence_outside_the_served_set_cannot_support_the_answer(self) -> None:
        result = self._evaluate((_record(chunk_id="c9"),), served=("c1",))

        self.assertEqual(result.confirmed_fields, ())
        self.assertEqual(result.unavailable_evidence, ())

    def test_a7_an_empty_served_set_confirms_nothing(self) -> None:
        result = self._evaluate((_record(),), served=())

        self.assertEqual(result.confirmed_fields, ())
        self.assertEqual(result.evidence_count, 0)

    def test_a9_independent_state_per_requested_field(self) -> None:
        result = self._evaluate(
            (
                _record(),
                _record("contract_period", status=FieldStatus.UNAVAILABLE,
                        chunk_id="c2", value=None, reason=FieldReason.NOT_STATED),
            ),
            served=("c1", "c2"),
        )

        self.assertEqual(result.confirmed_fields, (CONTRACT_AMOUNT,))
        self.assertEqual(result.missing_fields, ("contract_period",))
        self.assertEqual(result.unavailable_fields, ("contract_period",))
        self.assertFalse(result.answerable)

    def test_a8_a_general_evidence_request_keeps_its_legacy_verdict(self) -> None:
        legacy = self.guard.evaluate(
            _generated(), agent_result=SimpleNamespace(resolution=None),
            execution=_served("c1"),
        )
        unchanged = self.guard.evaluate(
            _generated(),
            agent_result=SimpleNamespace(field_evidence=(), resolution=None),
            execution=_served("c1"),
        )

        self.assertIs(legacy.status, AnswerabilityStatus.ANSWERABLE)
        self.assertEqual(legacy.reason, "requested_evidence_is_citable")
        self.assertEqual(legacy.to_public_dict(), unchanged.to_public_dict())


class CitedRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = AnswerabilityGuard()

    def _refusal(self, reason: FieldReason, *, chunk_id: str = "c1",
                 cited: tuple[str, ...] = ("c1",)) -> str:
        result = self.guard.evaluate(
            _generated("계약금액은 1,000원입니다. [1]", chunk_ids=cited),
            agent_result=SimpleNamespace(
                field_evidence=(
                    _record(status=FieldStatus.UNAVAILABLE, chunk_id=chunk_id,
                            value=None, reason=reason),
                ),
                resolution=None,
            ),
            execution=_served("c1"),
        )
        return guarded_answer_text(result, "계약금액은 1,000원입니다. [1]")

    def test_each_reason_states_what_the_evidence_says(self) -> None:
        not_stated = self._refusal(FieldReason.NOT_STATED)
        omitted = self._refusal(FieldReason.OMITTED)
        deferred = self._refusal(FieldReason.WITHHELD_OR_DEFERRED)

        self.assertTrue(not_stated.endswith("[1]"))
        self.assertIn("제시되어 있지 않아", not_stated)
        self.assertIn("기재되지 않은 것으로 명시", omitted)
        self.assertIn("추후 공개될 예정", deferred)
        self.assertEqual(len({not_stated, omitted, deferred}), 3)

    def test_a_refusal_never_repeats_the_number_it_refused(self) -> None:
        self.assertNotIn("1,000", self._refusal(FieldReason.NOT_STATED))

    def test_an_uncited_source_produces_no_citation(self) -> None:
        text = self._refusal(FieldReason.NOT_STATED, cited=("c2",))

        self.assertNotIn("[1]", text)
        self.assertIn("확인하기 어렵습니다", text)

    def test_a_missing_field_never_fabricates_a_citation(self) -> None:
        result = self.guard.evaluate(
            _generated(),
            agent_result=SimpleNamespace(
                field_evidence=(
                    _record(status=FieldStatus.MISSING, chunk_id=None, value=None),
                ),
                resolution=None,
            ),
            execution=_served("c1"),
        )
        text = guarded_answer_text(result, "")

        self.assertIsNone(result.refusal_citation)
        self.assertNotIn("[1]", text)


class PriorFailureClosureTests(unittest.TestCase):
    """The STEP 11-A failures, each still shut."""

    def setUp(self) -> None:
        self.guard = AnswerabilityGuard()

    def test_p1c_the_guard_never_reads_the_evidence_text(self) -> None:
        records = (
            _record(status=FieldStatus.UNAVAILABLE, value=None,
                    reason=FieldReason.NOT_STATED),
        )
        plain = self.guard.evaluate(
            _generated("확인된 사실입니다."),
            agent_result=SimpleNamespace(field_evidence=records, resolution=None),
            execution=_served("c1"),
        )
        with_table = self.guard.evaluate(
            _generated(
                "| 계약금액(원) | 184,000,000,000 |\n"
                "| 최근매출액(원) | 1,482,000,000,000 |\n"
                "계약금액은 1,840억원입니다. [1]"
            ),
            agent_result=SimpleNamespace(field_evidence=records, resolution=None),
            execution=_served("c1"),
        )

        self.assertEqual(plain.to_public_dict(), with_table.to_public_dict())
        self.assertFalse(with_table.answerable)

    def test_p1d_temporal_match_is_not_semantic_authority(self) -> None:
        chunk = _contract_chunk("c1", "d1", amount="-")
        question = "계약금액은 얼마인가요?"
        baseline = _corporate_evidence(question, (chunk,), states=(_state("d1"),))
        items = _items(chunk)

        for value in (True, False, None):
            flipped = tuple(
                SimpleNamespace(
                    chunk_id=item.chunk_id, doc_id=item.doc_id,
                    corp_code=item.corp_code, rcept_dt=item.rcept_dt,
                    provenance=item.provenance, temporal_match=value,
                )
                for item in items
            )
            records = corporate_event_field_evidence(
                question=question,
                plan=_corporate_plan(question),
                execution=_execution(states=(_state("d1"),), seeds=["d1"]),
                evidence_items=flipped,
            )
            self.assertEqual(
                [record.outcome for record in records],
                [record.outcome for record in baseline],
                value,
            )


class TraceCompatibilityTests(unittest.TestCase):
    def test_a_question_outside_the_fielded_lanes_adds_no_trace_content(self) -> None:
        result = AnswerabilityGuard().evaluate(
            _generated(), agent_result=SimpleNamespace(resolution=None),
            execution=_served("c1"),
        )
        trace = result.to_public_dict()

        self.assertEqual(trace["unavailable_fields"], [])
        self.assertEqual(trace["unavailable_evidence"], [])

    def test_a_fielded_verdict_reports_normalized_diagnostics(self) -> None:
        result = AnswerabilityGuard().evaluate(
            _generated(),
            agent_result=SimpleNamespace(
                field_evidence=(
                    _record(status=FieldStatus.UNAVAILABLE, value=None,
                            reason=FieldReason.NOT_STATED),
                ),
                resolution=None,
            ),
            execution=_served("c1"),
        )
        trace = result.to_public_dict()

        self.assertEqual(trace["unavailable_fields"], [CONTRACT_AMOUNT])
        self.assertEqual(
            trace["unavailable_evidence"][0]["reason"], FieldReason.NOT_STATED.value
        )
        self.assertEqual(trace["unavailable_evidence"][0]["chunk_id"], "c1")

    def test_carried_event_state_stays_out_of_the_public_block(self) -> None:
        """The pass-through is an internal seam, not an API addition."""

        trace = _execution(states=(_state("d1"),), seeds=["d1"]).event_expansion

        self.assertIn("event_member_states", trace)
        self.assertNotIn(
            "event_member_states", trace.get("corporate_event_expansion", {})
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class CorporateSilentBasisAuthorityTests(unittest.TestCase):
    """An exact receipt date whose basis helper stayed silent.

    ``period_type == "receipt_date"`` with one date is set from the question's
    own receipt wording.  ``date_basis`` is a narrower secondary signal that
    only fires when its marker sits inside a tight window after the date
    expression, so production can legitimately produce an exact receipt period
    with no basis at all.  Every plan here is built by ``QueryUnderstanding``
    so that combination is real rather than asserted: a fixture that sets the
    two together cannot express the state this covers.
    """

    COMPANY = "가상건설"
    #: Receipt wording the period reader recognizes, with the marker pushed
    #: past the basis helper's window by an intervening disclosure name.
    SILENT_BASIS = "가상건설의 2023년 6월 26일 공급계약 체결 건 공시의 계약금액은?"
    #: The same request with the marker adjacent, so the basis is named.
    NAMED_BASIS = "가상건설이 2023년 6월 26일 공시한 공급계약의 계약금액은?"
    AMOUNT = "1,234,567,890"
    LATER_AMOUNT = "9,876,543,210"
    HISTORICAL = "doc_hist_a"
    CANONICAL = "doc_canon_b"

    def understood(self, question: str) -> QueryPlan:
        return QueryUnderstanding(
            {self.COMPANY: {self.COMPANY}}
        ).understand(question)

    def plan(self, question: str | None = None, **overrides) -> QueryPlan:
        # Only the corp code is supplied: resolving it needs a corpus this test
        # does not carry.  Period, basis and correction policy stay exactly as
        # query understanding produced them.
        base = replace(
            self.understood(question or self.SILENT_BASIS), corp_code=CORP
        )
        return replace(base, **overrides) if overrides else base

    def chain_state(self, correction_status: str = RESOLVED):
        """The historical filing, whose state names a later chain member."""

        return _state(
            self.CANONICAL,
            canonical=self.CANONICAL,
            group="chain-1",
            correction_status=correction_status,
            member_count=2,
        )

    def evidence(self, chunks, states, *, plan=None, question=None):
        return corporate_event_field_evidence(
            question=question or self.SILENT_BASIS,
            plan=plan if plan is not None else self.plan(),
            execution=_repository_shaped_execution(states),
            evidence_items=_items(*chunks),
        )

    def _run(self, served, states, *, plan=None):
        """The served filings through the real orchestrator and guard."""

        selected = plan if plan is not None else self.plan()
        candidates = [
            CandidateChunk(str(chunk["chunk_id"]), str(chunk["doc_id"]), chunk,
                           MetadataMatch())
            for chunk in served
        ]
        results = [
            RetrievalResult(str(chunk["chunk_id"]), str(chunk["doc_id"]),
                            1.0 / index, index, {})
            for index, chunk in enumerate(served, start=1)
        ]
        carried = _repository_shaped_execution(states)
        execution = SimpleNamespace(
            plan=selected,
            chunks=candidates,
            results=results,
            event_expansion=carried.event_expansion,
            correction_expansion=carried.correction_expansion,
        )
        result = AgentOrchestrator().run(self.SILENT_BASIS, selected, execution)
        generated = generate_answer(result.answer_draft)
        verdict = AnswerabilityGuard().evaluate(
            generated,
            plan=selected,
            agent_result=result,
            execution=SimpleNamespace(results=result.evidence_results),
        )
        return result, verdict, guarded_answer_text(verdict, generated.answer_text)

    # ------------------------------------------------------------------ A
    def test_receipt_date_plan_can_have_unspecified_date_basis(self) -> None:
        plan = self.plan()

        self.assertIsNotNone(plan.period)
        self.assertEqual(plan.period.period_type, "receipt_date")
        self.assertIsNotNone(plan.period.from_date)
        self.assertEqual(plan.period.from_date, plan.period.to_date)
        self.assertEqual(
            getattr(plan.date_basis, "value", plan.date_basis), "unspecified"
        )
        # The period already fixed which filing was asked for, so a silent
        # basis may not take that authority away.
        self.assertTrue(_selection_intent(plan).exact_historical_receipt_date)

        named = self.plan(self.NAMED_BASIS)
        self.assertEqual(
            getattr(named.date_basis, "value", named.date_basis), "receipt_date"
        )
        self.assertTrue(_selection_intent(named).exact_historical_receipt_date)

    # ------------------------------------------------------------------ B
    def test_exact_receipt_date_value_bearing_contract_stays_authoritative(
        self,
    ) -> None:
        historical = _contract_chunk(
            "hist-amount", self.HISTORICAL, amount=self.AMOUNT
        )

        records = self.evidence((historical,), {self.HISTORICAL: self.chain_state()})

        self.assertEqual(len(records), 1)
        found = records[0]
        self.assertIs(found.status, FieldStatus.AVAILABLE)
        self.assertEqual(found.value, self.AMOUNT)
        # The later canonical pointer may not take the filing the question
        # named, and the answer must stay citable.
        self.assertEqual(found.doc_id, self.HISTORICAL)
        self.assertNotEqual(found.doc_id, self.CANONICAL)
        self.assertEqual(found.chunk_id, "hist-amount")
        self.assertTrue(found.authoritative)

    # ------------------------------------------------------------------ C
    def test_exact_receipt_date_withheld_amount_is_grounded_unavailable(
        self,
    ) -> None:
        historical = _contract_chunk(
            "hist-amount", self.HISTORICAL, amount="-",
            extra_rows=(
                _row("8. 비고", "계약금액은 유보기간 종료 후 추후 공시 예정"),
            ),
        )

        result, verdict, answer = self._run(
            (historical,), {self.HISTORICAL: self.chain_state()}
        )

        self.assertEqual(len(result.field_evidence), 1)
        found = result.field_evidence[0]
        self.assertIs(found.status, FieldStatus.UNAVAILABLE)
        self.assertIs(found.reason, FieldReason.WITHHELD_OR_DEFERRED)
        self.assertEqual(found.doc_id, self.HISTORICAL)
        self.assertIsNotNone(found.chunk_id)
        self.assertIsNotNone(found.table_id)
        self.assertIsNotNone(found.row_start)
        self.assertTrue(found.authoritative)
        self.assertIsNone(found.value)

        self.assertFalse(verdict.answerable)
        self.assertEqual(verdict.unavailable_fields, (CONTRACT_AMOUNT,))
        self.assertIsNotNone(verdict.refusal_citation)
        # The genuine numbers sitting beside the blank say nothing about the
        # contract amount, and none of them may stand in for it.
        for neighbour in ("1,482,000,000,000", "12.4"):
            self.assertNotEqual(found.value, neighbour)
            self.assertNotIn(neighbour, answer)

    # ------------------------------------------------------------------ D
    def test_explicit_latest_intent_still_uses_canonical_authority(self) -> None:
        historical = _contract_chunk(
            "hist-amount", self.HISTORICAL, amount=self.AMOUNT
        )
        corrected = _contract_chunk(
            "canon-amount", self.CANONICAL, amount=self.LATER_AMOUNT,
            rcept_dt=LATER_RECEIPT, table_id="t0002",
        )
        final = self.chain_state()
        states = {self.HISTORICAL: final, self.CANONICAL: final}
        latest = self.plan(
            correction_policy="latest_preferred",
            route_evidence={"is_correction": "explicit latest"},
        )
        corrected_only = self.plan(correction_policy="corrected_only")

        for label, plan in (("latest", latest), ("corrected", corrected_only)):
            with self.subTest(intent=label):
                # A silent basis must not turn an explicit latest/corrected
                # request into a historical read.
                self.assertFalse(
                    _selection_intent(plan).exact_historical_receipt_date
                )
                records = self.evidence(
                    (historical, corrected), states, plan=plan
                )
                authoritative = [item for item in records if item.authoritative]
                self.assertEqual(len(authoritative), 1)
                self.assertEqual(authoritative[0].doc_id, self.CANONICAL)
                self.assertEqual(authoritative[0].value, self.LATER_AMOUNT)

    # ------------------------------------------------------------------ E
    def test_contract_date_basis_does_not_claim_receipt_date_authority(
        self,
    ) -> None:
        historical = _contract_chunk(
            "hist-amount", self.HISTORICAL, amount=self.AMOUNT
        )
        for basis in (DateBasis.CONTRACT_DATE, DateBasis.PERIOD_START):
            with self.subTest(date_basis=basis.value):
                plan = self.plan(date_basis=basis)
                # The period is untouched: only the basis positively names a
                # different real-world date, and that is a real veto.
                self.assertEqual(plan.period.period_type, "receipt_date")
                self.assertEqual(plan.period.from_date, plan.period.to_date)
                self.assertFalse(
                    _selection_intent(plan).exact_historical_receipt_date
                )

                records = self.evidence(
                    (historical,), {self.HISTORICAL: self.chain_state()},
                    plan=plan,
                )
                found = records[0]
                self.assertEqual(found.doc_id, self.CANONICAL)
                self.assertIs(found.status, FieldStatus.MISSING)

    # ------------------------------------------------------------------ F
    def test_non_exact_periods_do_not_activate_historical_authority(self) -> None:
        exact = self.plan().period
        cases = (
            (
                "receipt range",
                QueryPeriod(from_date="2023-06-01", to_date="2023-06-30",
                            period_type="receipt_date"),
            ),
            (
                "non receipt period type",
                QueryPeriod(from_date=exact.from_date, to_date=exact.to_date,
                            period_type="date_range"),
            ),
            ("no period", None),
        )
        for label, period in cases:
            with self.subTest(period=label):
                plan = self.plan(period=period)
                self.assertFalse(
                    _selection_intent(plan).exact_historical_receipt_date
                )

    # ------------------------------------------------- correction finality
    def test_unresolved_finality_fails_closed_without_a_named_basis(self) -> None:
        historical = _contract_chunk(
            "hist-amount", self.HISTORICAL, amount=self.AMOUNT
        )
        for status in (AMBIGUOUS, UNRESOLVED):
            with self.subTest(correction_status=status):
                # P0-A could not say which filing is final.  A silent basis is
                # not permission to read either end of the chain.
                records = self.evidence(
                    (historical,),
                    {self.HISTORICAL: self.chain_state(correction_status=status)},
                )
                self.assertEqual(
                    [item.status for item in records], [FieldStatus.CONFLICT]
                )
