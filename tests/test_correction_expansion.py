"""Correction expansion: the serving path that reaches the rest of a chain.

The bug these tests pin down: a question anchored on the original's receipt date
turns that date into a hard candidate filter, so a correction filed later can
never enter the candidate set no matter what the correction graph knows.  The
fixtures reproduce that exactly -- initial retrieval sees only the root -- and
then assert the graph puts the missing documents back.
"""

import unittest

from app.reasoning.correction_graph import (
    CorrectionGraphUnavailable,
    CorrectionNotice,
    DisclosureRecord,
    build_correction_graph,
)
from app.reasoning.query_plan import QueryExecutor, QueryPlan
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.router import QueryRouter
from app.retrieval.correction_expansion import (
    PROVENANCE_KEY,
    CorrectionExpander,
    build_default_expander,
)
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


COMPANY = "현대건설"
CORP = "00164779"

# root -> correction1 -> correction2, each naming the previous filing's date.
CHAIN = (
    ("exchange_root", "2023-06-26", "20230626800002", False, "단일판매ㆍ공급계약체결"),
    ("exchange_c1", "2024-02-15", "20240215801246", True, "[기재정정]단일판매ㆍ공급계약체결"),
    ("exchange_c2", "2026-01-20", "20260120800597", True, "[기재정정]단일판매ㆍ공급계약체결"),
)
CONTRACT_AMOUNT = {
    "exchange_root": "계약금액 100,000,000,000원 계약기간 2023-06-26 ~ 2025-12-31",
    "exchange_c1": "계약금액 120,000,000,000원 계약기간 2024-02-15 ~ 2026-06-30",
    "exchange_c2": "계약금액 155,000,000,000원 계약기간 2026-01-20 ~ 2027-12-31",
}


def _records(corp_code: str = CORP) -> list[DisclosureRecord]:
    return [
        DisclosureRecord(
            doc_id=doc_id,
            corp_code=corp_code,
            doc_group="exchange",
            report_nm=report_nm,
            rcept_no=rcept_no,
            rcept_dt=rcept_dt,
            doc_subtype="단일판매공급계약체결",
            is_correction=is_correction,
        )
        for doc_id, rcept_dt, rcept_no, is_correction, report_nm in CHAIN
    ]


def _notices() -> dict[str, CorrectionNotice]:
    return {
        "exchange_c1": CorrectionNotice(
            doc_id="exchange_c1",
            target_submitted_on="2023-06-26",
            target_report_nm="단일판매ㆍ공급계약 체결",
        ),
        "exchange_c2": CorrectionNotice(
            doc_id="exchange_c2",
            target_submitted_on="2024-02-15",
            target_report_nm="단일판매ㆍ공급계약 체결",
        ),
    }


def _graph(corp_code: str = CORP):
    return build_correction_graph(_records(corp_code), _notices())


class _Backend:
    """Metadata, chunk, and lexical behaviour of the production backend.

    ``get_candidate_documents`` returns the company's disclosures and lets the
    router apply the question's own date window, exactly as PostgreSQL plus
    ``QueryRouter.filter_documents`` do.  ``fetch_documents`` looks documents up
    by identity and applies no filter at all.
    """

    def __init__(self, records=None) -> None:
        self.records = records or _records()
        self.fetch_calls: list[list[str]] = []

    def _document(self, record: DisclosureRecord) -> CandidateDocument:
        return CandidateDocument(
            doc_id=record.doc_id,
            metadata={
                "doc_id": record.doc_id,
                "corp_code": record.corp_code,
                "corp_name": COMPANY,
                "doc_group": record.doc_group,
                "doc_subtype": record.doc_subtype,
                "report_nm": record.report_nm,
                "rcept_dt": record.rcept_dt,
                "is_correction": record.is_correction,
            },
            metadata_match=MetadataMatch(),
        )

    def get_candidate_documents(self, **_filters) -> list[CandidateDocument]:
        return [self._document(record) for record in self.records]

    def fetch_documents(self, doc_ids) -> list[CandidateDocument]:
        wanted = sorted({str(doc_id) for doc_id in doc_ids})
        self.fetch_calls.append(wanted)
        return [
            self._document(record)
            for record in self.records
            if record.doc_id in wanted
        ]

    def get_candidate_chunks(self, documents) -> list[CandidateChunk]:
        chunks: list[CandidateChunk] = []
        for document in documents:
            doc_id = document.doc_id
            for index, body in enumerate(
                (
                    CONTRACT_AMOUNT.get(doc_id, "계약 내용"),
                    f"{doc_id} 부속 서류 및 기타 참고 사항",
                )
            ):
                chunks.append(
                    CandidateChunk(
                        chunk_id=f"{doc_id}:c{index}",
                        doc_id=doc_id,
                        chunk={
                            **document.metadata,
                            "content": body,
                            "retrieval_text": body,
                            "section_path": ["계약 내용"],
                        },
                        metadata_match=document.metadata_match,
                    )
                )
        return chunks

    def retrieve(self, query, candidates, *, top_k=None):
        terms = [term for term in str(query).split() if term]
        scored = []
        for candidate in candidates:
            text = str(candidate.chunk.get("retrieval_text") or "")
            score = sum(1.0 for term in terms if term in text)
            scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        if top_k is not None:
            scored = scored[:top_k]
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                bm25_score=float(score),
                rank=rank,
                metadata_match=candidate.metadata_match.to_dict(),
            )
            for rank, (score, candidate) in enumerate(scored, start=1)
        ]


def _plan(question: str) -> QueryPlan:
    return QueryUnderstanding({COMPANY: {COMPANY}}).understand(question)


#: ``expander=None`` means "run without expansion"; the default builds one.
_AUTO = object()


def _execute(question: str, *, graph=None, backend=None, expander=_AUTO):
    backend = backend or _Backend()
    graph = _graph() if graph is None else graph
    if expander is _AUTO:
        expander = build_default_expander(graph, backend, backend, backend)
    executor = QueryExecutor(
        backend,
        router=QueryRouter(correction_graph=graph),
        correction_expander=expander,
    )
    return backend, executor.execute(_plan(question))


ORIGINAL_Q = (
    "현대건설이 2023년 6월 26일 최초 공시한 단일판매·공급계약체결의 주요 계약 내용을 알려줘"
)
LATEST_Q = (
    "현대건설의 2023년 6월 26일 단일판매·공급계약체결 공시는 최종 정정 기준으로 "
    "계약 내용이 어떻게 되어 있어?"
)
HISTORY_Q = (
    "현대건설이 2023년 6월 26일 공시한 단일판매·공급계약체결은 이후 어떻게 정정되었어? "
    "최초 공시부터 최종 정정까지 변경 이력을 시간순으로 정리해줘"
)


class TheBugTests(unittest.TestCase):
    """Without expansion the anchor date makes the rest of the chain unreachable."""

    def test_the_anchor_date_is_a_hard_filter_on_the_candidate_set(self) -> None:
        _backend, execution = _execute(LATEST_Q, expander=None)
        self.assertEqual(
            [document.doc_id for document in execution.documents], ["exchange_root"]
        )
        self.assertEqual(
            sorted({result.doc_id for result in execution.results}), ["exchange_root"]
        )
        # The graph knows the final version, yet retrieval cannot reach it.
        self.assertEqual(_graph().get_latest_report("exchange_root"), "exchange_c2")


class OriginalIntentTests(unittest.TestCase):
    """A: the original question keeps answering from the original."""

    def test_original_intent_is_not_replaced_by_the_latest_document(self) -> None:
        _backend, execution = _execute(ORIGINAL_Q)
        doc_ids = sorted({result.doc_id for result in execution.results})
        self.assertEqual(doc_ids, ["exchange_root"])
        self.assertFalse(execution.correction_expansion["correction_expanded"])
        self.assertEqual(
            execution.correction_expansion["correction_status"], "not_requested"
        )
        self.assertEqual(execution.plan.correction_policy, "original_only")

    def test_original_intent_never_fetches_chain_documents(self) -> None:
        backend, _execution = _execute(ORIGINAL_Q)
        self.assertEqual(backend.fetch_calls, [])


class LatestIntentTests(unittest.TestCase):
    """B: the latest question reaches the final version through the graph."""

    def setUp(self) -> None:
        self.backend, self.execution = _execute(LATEST_Q)
        self.trace = self.execution.correction_expansion

    def test_the_final_document_is_added_to_the_evidence_context(self) -> None:
        doc_ids = {result.doc_id for result in self.execution.results}
        self.assertIn("exchange_c2", doc_ids)
        self.assertIn("exchange_root", doc_ids)
        self.assertIn(
            "exchange_c2", {candidate.doc_id for candidate in self.execution.chunks}
        )

    def test_the_added_document_carries_readable_chunks_not_just_metadata(self) -> None:
        added = [
            candidate
            for candidate in self.execution.chunks
            if candidate.doc_id == "exchange_c2"
        ]
        self.assertTrue(added)
        self.assertIn("155,000,000,000", added[0].chunk["content"])

    def test_only_the_relevant_chunks_of_the_added_document_are_used(self) -> None:
        added = [
            result for result in self.execution.results if result.doc_id == "exchange_c2"
        ]
        self.assertLessEqual(len(added), 2)
        # The contract chunk, not the attachments boilerplate, ranks first.
        self.assertEqual(added[0].chunk_id, "exchange_c2:c0")

    def test_the_anchor_date_is_not_reapplied_to_the_added_document(self) -> None:
        self.assertEqual(self.backend.fetch_calls, [["exchange_c2"]])
        added = next(
            candidate
            for candidate in self.execution.chunks
            if candidate.doc_id == "exchange_c2"
        )
        self.assertEqual(added.chunk["rcept_dt"], "2026-01-20")

    def test_added_evidence_is_distinguishable_from_retrieved_evidence(self) -> None:
        added = next(
            result for result in self.execution.results if result.doc_id == "exchange_c2"
        )
        retrieved = next(
            result
            for result in self.execution.results
            if result.doc_id == "exchange_root"
        )
        self.assertIn(PROVENANCE_KEY, added.metadata_match)
        self.assertNotIn(PROVENANCE_KEY, retrieved.metadata_match)

    def test_the_trace_records_the_chain_it_came_from(self) -> None:
        self.assertTrue(self.trace["correction_expanded"])
        self.assertEqual(self.trace["correction_intent"], "latest")
        self.assertEqual(self.trace["correction_group_id"], "exchange_root")
        self.assertEqual(self.trace["correction_root_doc_id"], "exchange_root")
        self.assertEqual(self.trace["correction_latest_doc_id"], "exchange_c2")
        self.assertEqual(self.trace["correction_added_doc_ids"], ["exchange_c2"])

    def test_the_retrieved_ranking_is_left_untouched(self) -> None:
        _backend, plain = _execute(LATEST_Q, expander=None)
        expanded = self.execution.results
        self.assertEqual(
            [(r.rank, r.chunk_id) for r in expanded[: len(plain.results)]],
            [(r.rank, r.chunk_id) for r in plain.results],
        )


class HistoryIntentTests(unittest.TestCase):
    """C: the history question reaches every document of the chain."""

    def setUp(self) -> None:
        self.backend, self.execution = _execute(HISTORY_Q)
        self.trace = self.execution.correction_expansion

    def test_every_chain_member_is_available_as_evidence(self) -> None:
        doc_ids = {result.doc_id for result in self.execution.results}
        self.assertEqual(
            doc_ids & {"exchange_root", "exchange_c1", "exchange_c2"},
            {"exchange_root", "exchange_c1", "exchange_c2"},
        )

    def test_the_chain_is_available_in_receipt_order(self) -> None:
        chain = _graph().get_correction_chain("exchange_root")
        self.assertEqual(
            [member.doc_id for member in chain],
            ["exchange_root", "exchange_c1", "exchange_c2"],
        )
        self.assertEqual([member.correction_order for member in chain], [0, 1, 2])

    def test_only_the_missing_members_are_fetched(self) -> None:
        self.assertEqual(self.backend.fetch_calls, [["exchange_c1", "exchange_c2"]])

    def test_the_trace_names_the_added_documents(self) -> None:
        self.assertEqual(self.trace["correction_intent"], "history")
        self.assertEqual(
            sorted(self.trace["correction_added_doc_ids"]),
            ["exchange_c1", "exchange_c2"],
        )


class AmbiguousChainTests(unittest.TestCase):
    """D: an unverified correction is never expanded into a chain."""

    def _ambiguous_graph(self):
        # Two identical prior titles and no notice: the correction cannot be
        # attached to either, so it is stored as a group of one.
        records = [
            DisclosureRecord("a1", CORP, "exchange", "단일판매ㆍ공급계약체결", "1", "2023-01-05",
                             "단일판매공급계약체결"),
            DisclosureRecord("a2", CORP, "exchange", "단일판매ㆍ공급계약체결", "2", "2023-02-05",
                             "단일판매공급계약체결"),
            DisclosureRecord("a3", CORP, "exchange", "[기재정정]단일판매ㆍ공급계약체결", "3",
                             "2023-03-01", "단일판매공급계약체결", is_correction=True),
        ]
        return build_correction_graph(records), records

    def test_an_ambiguous_singleton_is_not_expanded(self) -> None:
        graph, records = self._ambiguous_graph()
        state = graph.document_states(["a3"])["a3"]
        self.assertTrue(state.is_latest)
        self.assertFalse(state.is_resolved)

        backend = _Backend(records)
        expander = build_default_expander(graph, backend, backend, backend)
        results = [
            RetrievalResult("a3:c0", "a3", 1.0, 1, MetadataMatch().to_dict())
        ]
        expansion = expander.expand(
            _plan(LATEST_Q), documents=[], chunks=[], results=results
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(expansion.trace["correction_status"], "no_resolved_group")
        self.assertEqual(backend.fetch_calls, [])

    def test_a_document_outside_any_group_is_not_expanded(self) -> None:
        backend = _Backend()
        expander = build_default_expander(_graph(), backend, backend, backend)
        results = [
            RetrievalResult("other:c0", "other_doc", 1.0, 1, MetadataMatch().to_dict())
        ]
        expansion = expander.expand(
            _plan(LATEST_Q), documents=[], chunks=[], results=results
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(backend.fetch_calls, [])


class ExpansionSafetyTests(unittest.TestCase):
    """Guards that must hold whatever the graph or the backend does."""

    def test_another_company_is_never_pulled_into_the_chain(self) -> None:
        backend = _Backend()
        # A damaged graph row pointing at a disclosure of another company.
        foreign = DisclosureRecord(
            "foreign_doc", "99999999", "exchange", "단일판매ㆍ공급계약체결", "9",
            "2026-02-01", "단일판매공급계약체결", is_correction=True,
        )
        backend.records = [*backend.records, foreign]

        class _ForeignGraph:
            def document_states(self, doc_ids):
                return _graph().document_states(doc_ids)

            def get_correction_chain(self, doc_id):
                chain = list(_graph().get_correction_chain(doc_id))
                from dataclasses import replace

                return [*chain[:-1], replace(chain[-1], doc_id="foreign_doc")]

        expander = CorrectionExpander(_ForeignGraph(), backend, backend, backend)
        results = [
            RetrievalResult(
                "exchange_root:c0", "exchange_root", 1.0, 1, MetadataMatch().to_dict()
            )
        ]
        with self.assertLogs("app.retrieval.correction_expansion", level="WARNING"):
            expansion = expander.expand(
                _plan(LATEST_Q),
                documents=[backend._document(backend.records[0])],
                chunks=[],
                results=results,
            )
        self.assertFalse(expansion.expanded)
        self.assertNotIn(
            "foreign_doc", [d.doc_id for d in expansion.added_documents]
        )

    def test_a_chain_beyond_the_safety_cap_is_refused_whole(self) -> None:
        backend = _Backend()
        expander = CorrectionExpander(
            _graph(), backend, backend, backend, max_documents=1
        )
        expansion = expander.expand(
            _plan(HISTORY_Q),
            documents=[],
            chunks=[],
            results=[
                RetrievalResult(
                    "exchange_root:c0", "exchange_root", 1.0, 1,
                    MetadataMatch().to_dict(),
                )
            ],
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(expansion.trace["correction_status"], "too_many_documents")

    def test_the_cap_leaves_room_for_the_longest_real_chain(self) -> None:
        backend = _Backend()
        expander = CorrectionExpander(_graph(), backend, backend, backend)
        # The longest legitimate chain in the corpus holds fifteen documents.
        self.assertGreaterEqual(expander.max_documents, 15)

    def test_an_unavailable_graph_degrades_instead_of_failing(self) -> None:
        class _Unavailable:
            def document_states(self, doc_ids):
                raise CorrectionGraphUnavailable("db/006 not applied")

        backend = _Backend()
        expander = CorrectionExpander(_Unavailable(), backend, backend, backend)
        with self.assertLogs("app.retrieval.correction_expansion", level="WARNING"):
            expansion = expander.expand(
                _plan(LATEST_Q),
                documents=[],
                chunks=[],
                results=[
                    RetrievalResult(
                        "exchange_root:c0", "exchange_root", 1.0, 1,
                        MetadataMatch().to_dict(),
                    )
                ],
            )
        self.assertFalse(expansion.expanded)
        self.assertEqual(expansion.trace["correction_status"], "graph_unavailable")

    def test_a_backend_without_identity_lookup_disables_expansion(self) -> None:
        class _NoLookup:
            def get_candidate_chunks(self, documents):
                return []

            def retrieve(self, query, candidates, *, top_k=None):
                return []

        self.assertIsNone(
            build_default_expander(_graph(), _NoLookup(), _NoLookup(), _NoLookup())
        )


class UnrelatedQueryRegressionTests(unittest.TestCase):
    """E: an ordinary question is byte-for-byte what it was before."""

    QUESTIONS = (
        "현대건설 2024년 매출액",
        "현대건설 단일판매·공급계약체결 계약금액",
        "현대건설 2023년 6월 26일 단일판매·공급계약체결 계약 내용",
        "현대건설 정정 공시만 계약금액",
    )

    def _snapshot(self, execution) -> tuple:
        return (
            [document.doc_id for document in execution.documents],
            [candidate.chunk_id for candidate in execution.chunks],
            [(r.rank, r.chunk_id, r.doc_id) for r in execution.results],
        )

    def test_questions_without_a_correction_intent_are_unchanged(self) -> None:
        for question in self.QUESTIONS:
            with self.subTest(question=question):
                _b1, expanded = _execute(question)
                _b2, plain = _execute(question, expander=None)
                self.assertEqual(self._snapshot(expanded), self._snapshot(plain))
                self.assertFalse(
                    expanded.correction_expansion["correction_expanded"]
                )

    def test_no_chain_documents_are_fetched_for_an_ordinary_question(self) -> None:
        for question in self.QUESTIONS:
            with self.subTest(question=question):
                backend, _execution = _execute(question)
                self.assertEqual(backend.fetch_calls, [])


class PointInTimeTests(unittest.TestCase):
    """F: a date-window question is not silently overwritten by the latest."""

    def test_a_plain_date_window_question_is_never_expanded(self) -> None:
        backend, execution = _execute(
            "현대건설 2023년 6월 26일 공시된 단일판매·공급계약체결 계약금액"
        )
        doc_ids = {result.doc_id for result in execution.results}
        self.assertEqual(doc_ids, {"exchange_root"})
        self.assertNotIn("exchange_c2", doc_ids)
        self.assertEqual(backend.fetch_calls, [])
        self.assertIsNone(
            execution.correction_expansion["correction_intent"]
        )

    def test_the_original_survives_alongside_the_added_latest(self) -> None:
        """Expansion adds; it never removes or replaces what was retrieved."""

        _backend, execution = _execute(LATEST_Q)
        doc_ids = [result.doc_id for result in execution.results]
        self.assertIn("exchange_root", doc_ids)
        self.assertIn("exchange_c2", doc_ids)
        self.assertLess(
            doc_ids.index("exchange_root"), doc_ids.index("exchange_c2")
        )


if __name__ == "__main__":
    unittest.main()
