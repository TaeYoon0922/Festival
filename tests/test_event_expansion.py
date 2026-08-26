"""Corporate event evidence expansion at the retrieval boundary.

Two layers, as in the repository tests.  The fake-backend tests run everywhere
and pin the contract: what expands, what must never expand, and what the trace
records.  The live-PostgreSQL tests at the bottom replay real corpus lifecycles
and are skipped unless ``FESTIVAL_TEST_DATABASE_URL`` points at a disposable
database.
"""

from __future__ import annotations

import os
import unittest

from app.reasoning.corporate_event import (
    CorporateEvent,
    CorporateEventMember,
    EventFamily,
    EventLifecycleStatus,
    EventMemberRole,
    EventResolutionStatus,
)
from app.reasoning.corporate_event_graph import (
    CorporateEventGraph,
    CorporateEventGraphUnavailable,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.event_expansion import (
    CONTRACT_EVENT_TYPES,
    DEFAULT_EVENT_EVIDENCE_LIMIT,
    EventExpander,
    build_default_event_expander,
)
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


CORP = "00126380"
OTHER_CORP = "00999999"
#: A row a real 해지 filing states, so the structured-evidence preference has
#: something to prefer.
TERMINATION_ROW = (
    "- 해지계약명 전기차 배터리 공급계약 | 계약상대 Ford Motor Company | "
    "해지금액(원) 9,603,075,000,000 | 해지일자 2025-12-17 | "
    "해지 주요사유 거래 상대방의 계약 해지 통보 | 관련공시 2024-10-15"
)
CONTRACT_ROW = (
    "- 체결계약명 전기차 배터리 공급계약 | 계약상대 Ford Motor Company | "
    "계약금액(원) - | 시작일 2027-01-01 | 종료일 2032-12-31 | "
    "계약(수주)일자 2024-10-14"
)


def _plan(event_type: str | None = "supply_contract", query: str = "공급계약 해지") -> QueryPlan:
    return QueryPlan(
        query=query,
        raw_query=query,
        company="테스트",
        corp_code=CORP,
        task_type="disclosure_lookup",
        period=QueryPeriod(),
        event_type=event_type,
        top_k=10,
    )


def _document(doc_id: str, corp_code: str = CORP) -> CandidateDocument:
    return CandidateDocument(
        doc_id=doc_id,
        metadata={"doc_id": doc_id, "corp_code": corp_code, "corp_name": "테스트"},
        metadata_match=MetadataMatch(),
    )


def _chunk(doc_id: str, text: str, suffix: str = "c1") -> CandidateChunk:
    return CandidateChunk(
        chunk_id=f"{doc_id}:{suffix}",
        doc_id=doc_id,
        chunk={"chunk_id": f"{doc_id}:{suffix}", "doc_id": doc_id, "retrieval_text": text},
        metadata_match=MetadataMatch(),
    )


def _result(doc_id: str, rank: int, suffix: str = "c1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{doc_id}:{suffix}",
        doc_id=doc_id,
        bm25_score=1.0,
        rank=rank,
        metadata_match={},
    )


def _member(doc_id, role, order, *, corp_code=CORP, chain=(), collapsed=None,
            group=None, status=None):
    return CorporateEventMember(
        corp_code=corp_code,
        doc_id=doc_id,
        canonical_doc_id=doc_id,
        member_role=role,
        member_order=order,
        event_date=f"2025-01-{order + 1:02d}",
        root_doc_id=chain[0] if chain else None,
        correction_group_id=group,
        correction_resolution_status=status,
        correction_chain=tuple(chain),
        provenance={"collapsed_doc_ids": list(collapsed or [doc_id])},
    )


def _event(members, *, corp_code=CORP, resolution=EventResolutionStatus.RESOLVED,
           lifecycle=EventLifecycleStatus.TERMINATED, root=None):
    return CorporateEvent.create(
        corp_code=corp_code,
        event_family=EventFamily.SUPPLY_CONTRACT,
        root_logical_key=root or members[0].logical_key,
        lifecycle_status=lifecycle,
        resolution_status=resolution,
        resolution_source="related_reference_corroborated",
        members=tuple(members),
        opened_at="2024-01-01",
        closed_at="2025-12-17" if lifecycle == EventLifecycleStatus.TERMINATED else None,
        confidence=0.95,
    )


class _Backend:
    """Fetches documents and chunks by identity.  It cannot search."""

    def __init__(self, texts):
        self._texts = texts
        self.fetch_calls: list[list[str]] = []
        self.retrieve_calls = 0
        self.retrieve_args: list[tuple[str, list]] = []

    def fetch_documents(self, doc_ids):
        self.fetch_calls.append(list(doc_ids))
        return [_document(doc_id) for doc_id in doc_ids if doc_id in self._texts]

    def get_candidate_chunks(self, documents):
        chunks = []
        for document in documents:
            for index, text in enumerate(self._texts.get(document.doc_id, [])):
                chunks.append(_chunk(document.doc_id, text, f"c{index + 1}"))
        return chunks

    def retrieve(self, query, candidates, *, top_k=None):
        self.retrieve_calls += 1
        self.retrieve_args.append((query, list(candidates)))
        return [
            RetrievalResult(
                chunk_id=item.chunk_id, doc_id=item.doc_id,
                bm25_score=1.0, rank=index + 1, metadata_match={},
            )
            for index, item in enumerate(candidates[: top_k or len(candidates)])
        ]


def _expander(graph, backend, **kwargs):
    return EventExpander(graph, backend, backend, backend, **kwargs)


def _block(expansion):
    return expansion.to_dict().get("corporate_event_expansion") or {}


class ExpansionDirectionTests(unittest.TestCase):
    """A seed reaches the rest of its own lifecycle, in both directions."""

    def setUp(self) -> None:
        self.graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member("A", EventMemberRole.CONTRACT, 0),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ]
                )
            ]
        )
        self.backend = _Backend({"A": [CONTRACT_ROW], "T": [TERMINATION_ROW]})

    def _run(self, seed):
        expander = _expander(self.graph, self.backend)
        return expander.expand(
            _plan(),
            documents=[_document(seed)],
            chunks=[_chunk(seed, "seed")],
            results=[_result(seed, 1)],
        )

    def test_contract_seed_adds_the_termination(self) -> None:
        expansion = self._run("A")
        self.assertTrue(expansion.expanded)
        self.assertEqual(
            [d.doc_id for d in expansion.added_documents], ["T"]
        )
        block = _block(expansion)
        self.assertTrue(block["applied"])
        self.assertEqual(block["seed_doc_ids"], ["A"])
        self.assertEqual(block["added_doc_ids"], ["T"])
        self.assertFalse(block["truncated"])

    def test_termination_seed_adds_the_contract(self) -> None:
        expansion = self._run("T")
        self.assertTrue(expansion.expanded)
        self.assertEqual([d.doc_id for d in expansion.added_documents], ["A"])

    def test_expansion_never_issues_a_new_search(self) -> None:
        """Ranking happens inside already-fetched chunks; nothing is searched.

        The ranker may be used to order what was fetched, which is a rerank, not
        a retrieval retry: it only ever sees chunks of the document the graph
        already named, and no vector search is reachable at all.
        """

        self._run("A")
        # Documents are reached by identity, never by a query.
        self.assertEqual(self.backend.fetch_calls, [["T"]])
        for _, candidates in self.backend.retrieve_args:
            self.assertTrue(candidates)
            self.assertEqual({item.doc_id for item in candidates}, {"T"})
        # The expander holds no vector retriever to call.
        self.assertFalse(hasattr(self.backend, "vector_search"))

    def test_added_evidence_is_marked_as_relation_derived(self) -> None:
        expansion = self._run("A")
        provenance = expansion.added_results[0].metadata_match["event_expansion"]
        self.assertEqual(provenance["retrieval_source"], "event_expansion")
        self.assertEqual(provenance["member_role"], "termination")
        self.assertEqual(provenance["lifecycle_status"], "terminated")
        self.assertGreater(provenance["lifecycle_field_hits"], 0)

    def test_structured_lifecycle_row_is_preferred_over_prose(self) -> None:
        backend = _Backend({"A": [CONTRACT_ROW], "T": ["보도자료 요약입니다", TERMINATION_ROW]})
        expansion = _expander(self.graph, backend).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        first = expansion.added_chunks[0]
        self.assertIn("해지계약명", first.chunk["retrieval_text"])


class ExpansionGatingTests(unittest.TestCase):
    """Only a contract question expands."""

    def setUp(self) -> None:
        self.graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member("A", EventMemberRole.CONTRACT, 0),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ]
                )
            ]
        )
        self.backend = _Backend({"A": [CONTRACT_ROW], "T": [TERMINATION_ROW]})

    def _run(self, event_type):
        return _expander(self.graph, self.backend).expand(
            _plan(event_type=event_type),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )

    def test_contract_routes_expand(self) -> None:
        for event_type in sorted(CONTRACT_EVENT_TYPES):
            with self.subTest(event_type=event_type):
                self.assertTrue(self._run(event_type).expanded)

    def test_unrelated_routes_do_not_expand(self) -> None:
        for event_type in (None, "merger", "capital_increase", "treasury_share_disposal"):
            with self.subTest(event_type=event_type):
                expansion = self._run(event_type)
                self.assertFalse(expansion.expanded)
                self.assertEqual(expansion.to_dict()["event_status"], "not_requested")
                self.assertEqual(self.backend.fetch_calls, [])

    def test_periodic_and_holding_questions_do_not_expand(self) -> None:
        for query, event_type in (("2024년 매출액", None), ("국민연금 지분율", None)):
            with self.subTest(query=query):
                plan = _plan(event_type=event_type, query=query)
                expansion = _expander(self.graph, self.backend).expand(
                    plan,
                    documents=[_document("A")],
                    chunks=[_chunk("A", "seed")],
                    results=[_result("A", 1)],
                )
                self.assertFalse(expansion.expanded)


class ExpansionSafetyTests(unittest.TestCase):
    def test_unresolved_lifecycle_pulls_in_nothing(self) -> None:
        """The real outside-corpus termination shape."""

        graph = CorporateEventGraph(
            [
                _event(
                    [_member("T", EventMemberRole.TERMINATION, 0)],
                    resolution=EventResolutionStatus.UNRESOLVED,
                ),
                _event(
                    [_member("OTHER", EventMemberRole.CONTRACT, 0)],
                    resolution=EventResolutionStatus.UNRESOLVED,
                    lifecycle=EventLifecycleStatus.OPEN,
                ),
            ]
        )
        backend = _Backend({"T": [TERMINATION_ROW], "OTHER": [CONTRACT_ROW]})
        expansion = _expander(graph, backend).expand(
            _plan(),
            documents=[_document("T")],
            chunks=[_chunk("T", "seed")],
            results=[_result("T", 1)],
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(expansion.to_dict()["event_status"], "no_resolved_event")
        block = _block(expansion)
        self.assertEqual(block["skipped_doc_ids"], ["T"])
        self.assertEqual(
            block["skipped"][0]["reason"], "lifecycle_not_resolved"
        )
        self.assertEqual(backend.fetch_calls, [])

    def test_a_different_event_is_never_added(self) -> None:
        graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member("A", EventMemberRole.CONTRACT, 0),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ]
                ),
                _event(
                    [
                        _member("B", EventMemberRole.CONTRACT, 0),
                        _member("T2", EventMemberRole.TERMINATION, 1),
                    ],
                    root="B",
                ),
            ]
        )
        backend = _Backend({k: [CONTRACT_ROW] for k in ("A", "T", "B", "T2")})
        expansion = _expander(graph, backend).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        self.assertEqual([d.doc_id for d in expansion.added_documents], ["T"])
        self.assertNotIn("B", _block(expansion)["added_doc_ids"])
        self.assertNotIn("T2", _block(expansion)["added_doc_ids"])

    def test_cross_company_document_is_dropped(self) -> None:
        graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member("A", EventMemberRole.CONTRACT, 0),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ]
                )
            ]
        )

        class _Rogue(_Backend):
            def fetch_documents(self, doc_ids):
                self.fetch_calls.append(list(doc_ids))
                return [_document(doc_id, corp_code=OTHER_CORP) for doc_id in doc_ids]

        backend = _Rogue({"A": [CONTRACT_ROW], "T": [TERMINATION_ROW]})
        expansion = _expander(graph, backend).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(
            expansion.to_dict()["event_status"], "documents_unavailable"
        )

    def test_already_retrieved_member_is_not_added_again(self) -> None:
        graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member("A", EventMemberRole.CONTRACT, 0),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ]
                )
            ]
        )
        backend = _Backend({"A": [CONTRACT_ROW], "T": [TERMINATION_ROW]})
        expansion = _expander(graph, backend).expand(
            _plan(),
            documents=[_document("A"), _document("T")],
            chunks=[_chunk("A", "seed"), _chunk("T", "seed")],
            results=[_result("A", 1), _result("T", 2)],
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(expansion.to_dict()["event_status"], "already_retrieved")


class SupersededSeedTests(unittest.TestCase):
    """A seed that is a superseded correction must not duplicate its own group."""

    def setUp(self) -> None:
        self.graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member(
                            "A2", EventMemberRole.CONTRACT, 0,
                            chain=("A", "A1", "A2"),
                            collapsed=["A", "A1", "A2"],
                            group="g1",
                            status=EventResolutionStatus.RESOLVED,
                        ),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ],
                    root="g1",
                )
            ]
        )
        self.backend = _Backend(
            {"A": [CONTRACT_ROW], "A1": [CONTRACT_ROW], "A2": [CONTRACT_ROW],
             "T": [TERMINATION_ROW]}
        )

    def test_superseded_seed_reaches_the_termination(self) -> None:
        for seed in ("A", "A1", "A2"):
            with self.subTest(seed=seed):
                expansion = _expander(self.graph, self.backend).expand(
                    _plan(),
                    documents=[_document(seed)],
                    chunks=[_chunk(seed, "seed")],
                    results=[_result(seed, 1)],
                )
                self.assertEqual(
                    [d.doc_id for d in expansion.added_documents], ["T"], seed
                )

    def test_superseded_seed_does_not_re_add_its_own_representative(self) -> None:
        """A1 already stands for A2; adding A2 would duplicate the evidence."""

        expansion = _expander(self.graph, self.backend).expand(
            _plan(),
            documents=[_document("A1")],
            chunks=[_chunk("A1", "seed")],
            results=[_result("A1", 1)],
        )
        added = _block(expansion)["added_doc_ids"]
        self.assertEqual(added, ["T"])
        self.assertNotIn("A2", added)
        self.assertNotIn("A", added)

    def test_correction_provenance_is_kept_in_the_trace(self) -> None:
        expansion = _expander(self.graph, self.backend).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        block = _block(expansion)
        self.assertEqual(block["seed_member_doc_ids"], {"A": "A2"})
        event = block["events"][0]
        self.assertEqual(event["seed_doc_id"], "A")
        self.assertEqual(event["seed_member_doc_id"], "A2")
        self.assertEqual(event["correction_group_id"], "g1")


class AmbiguousCorrectionSeedTests(unittest.TestCase):
    def test_ambiguous_member_is_not_given_a_verified_latest(self) -> None:
        graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member(
                            "A1", EventMemberRole.CONTRACT, 0,
                            chain=("A1",), collapsed=["A1"], group="A1",
                            status=EventResolutionStatus.AMBIGUOUS,
                        )
                    ],
                    resolution=EventResolutionStatus.UNRESOLVED,
                    lifecycle=EventLifecycleStatus.OPEN,
                )
            ]
        )
        backend = _Backend({"A1": [CONTRACT_ROW]})
        expansion = _expander(graph, backend).expand(
            _plan(),
            documents=[_document("A1")],
            chunks=[_chunk("A1", "seed")],
            results=[_result("A1", 1)],
        )
        self.assertFalse(expansion.expanded)
        self.assertEqual(backend.fetch_calls, [])


class BudgetAndTraceTests(unittest.TestCase):
    def _big_event(self):
        members = [_member("A", EventMemberRole.CONTRACT, 0)]
        members += [
            _member(f"T{i}", EventMemberRole.TERMINATION, i)
            for i in range(1, 6)
        ]
        return CorporateEventGraph([_event(members)])

    def test_document_limit_truncates_deterministically(self) -> None:
        graph = self._big_event()
        texts = {"A": [CONTRACT_ROW]}
        texts.update({f"T{i}": [TERMINATION_ROW] for i in range(1, 6)})
        backend = _Backend(texts)
        expansion = _expander(graph, backend, max_documents=2).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        block = _block(expansion)
        self.assertEqual(len(block["added_doc_ids"]), 2)
        self.assertTrue(block["truncated"])
        # Deterministic: the earliest members by timeline order, not a sample.
        self.assertEqual(block["added_doc_ids"], ["T1", "T2"])

    def test_total_evidence_is_bounded(self) -> None:
        graph = self._big_event()
        texts = {"A": [CONTRACT_ROW]}
        texts.update({f"T{i}": [TERMINATION_ROW] * 6 for i in range(1, 6)})
        backend = _Backend(texts)
        expansion = _expander(
            graph, backend, chunks_per_document=6, max_evidence=4
        ).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        self.assertLessEqual(len(expansion.added_results), 4)

    def test_default_evidence_limit_is_bounded(self) -> None:
        self.assertLessEqual(DEFAULT_EVENT_EVIDENCE_LIMIT, 32)

    def test_trace_carries_the_required_keys_and_metrics(self) -> None:
        graph = CorporateEventGraph(
            [
                _event(
                    [
                        _member("A", EventMemberRole.CONTRACT, 0),
                        _member("T", EventMemberRole.TERMINATION, 1),
                    ]
                )
            ]
        )
        backend = _Backend({"A": [CONTRACT_ROW], "T": [TERMINATION_ROW]})
        expansion = _expander(graph, backend).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )
        block = _block(expansion)
        for key in (
            "applied", "reason", "seed_doc_ids", "matched_event_ids",
            "added_doc_ids", "skipped_doc_ids", "truncated", "metrics",
        ):
            self.assertIn(key, block)
        metrics = block["metrics"]
        for key in (
            "seed_count", "graph_lookup_count", "added_doc_count",
            "added_evidence_count", "elapsed_ms",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["seed_count"], 1)
        self.assertEqual(metrics["added_doc_count"], 1)
        self.assertGreaterEqual(metrics["elapsed_ms"], 0.0)
        # JSON-clean: no enum members leak into the trace.
        import json

        json.dumps(block)


class FailureHandlingTests(unittest.TestCase):
    def _run(self, view):
        backend = _Backend({"A": [CONTRACT_ROW]})
        return EventExpander(view, backend, backend, backend).expand(
            _plan(),
            documents=[_document("A")],
            chunks=[_chunk("A", "seed")],
            results=[_result("A", 1)],
        )

    def test_unavailable_graph_degrades(self) -> None:
        class _Unavailable:
            def event_states(self, doc_ids):
                raise CorporateEventGraphUnavailable("db/007 not applied")

            def get_event_state(self, doc_id):
                raise CorporateEventGraphUnavailable("db/007 not applied")

            def get_event(self, doc_id):
                raise CorporateEventGraphUnavailable("db/007 not applied")

        expansion = self._run(_Unavailable())
        self.assertFalse(expansion.expanded)
        self.assertEqual(expansion.to_dict()["event_status"], "graph_unavailable")

    def test_programming_error_propagates(self) -> None:
        class _Broken:
            def event_states(self, doc_ids):
                raise RuntimeError("programming error")

            def get_event_state(self, doc_id):
                raise RuntimeError("programming error")

            def get_event(self, doc_id):
                raise RuntimeError("programming error")

        with self.assertRaises(RuntimeError):
            self._run(_Broken())

    def test_builder_returns_none_without_a_repository(self) -> None:
        backend = _Backend({})
        self.assertIsNone(
            build_default_event_expander(None, backend, backend, backend)
        )

    def test_builder_returns_none_without_fetch_documents(self) -> None:
        class _NoFetch:
            def get_candidate_chunks(self, documents):
                return []

        graph = CorporateEventGraph([])
        self.assertIsNone(
            build_default_event_expander(graph, _NoFetch(), _NoFetch(), _NoFetch())
        )


class SingleOwnerTests(unittest.TestCase):
    """One query must trigger event expansion exactly once."""

    class _CountingExpander:
        def __init__(self):
            self.calls = 0

        def expand(self, plan, *, documents, chunks, results):
            from app.retrieval.event_expansion import EventExpansion

            self.calls += 1
            return EventExpansion()

    def test_query_executor_expands_once(self) -> None:
        from app.reasoning.query_plan import QueryExecutor

        class _Backend2(_Backend):
            def get_candidate_documents(self, **kwargs):
                return [_document("A")]

        backend = _Backend2({"A": [CONTRACT_ROW]})
        expander = self._CountingExpander()
        QueryExecutor(backend, event_expander=expander).execute(_plan())
        self.assertEqual(expander.calls, 1)

    def test_executors_are_alternatives_not_a_chain(self) -> None:
        """The serving path uses one executor; neither wraps the other."""

        import inspect

        from app.reasoning.query_plan import QueryExecutor
        from app.retrieval.hybrid import HybridQueryExecutor

        hybrid_source = inspect.getsource(HybridQueryExecutor)
        self.assertNotIn("QueryExecutor(", hybrid_source)
        self.assertNotIn("HybridQueryExecutor(", inspect.getsource(QueryExecutor))

    def test_pipeline_wires_exactly_one_event_expander(self) -> None:
        import inspect

        from app.api import pipeline

        source = inspect.getsource(pipeline)
        self.assertEqual(source.count("event_expander=build_default_event_expander"), 1)


_LIVE_DSN = os.environ.get("FESTIVAL_TEST_DATABASE_URL")


@unittest.skipUnless(_LIVE_DSN, "FESTIVAL_TEST_DATABASE_URL is not set")
class LiveCorpusExpansionTests(unittest.TestCase):
    """Real corpus lifecycles, expanded through the PostgreSQL repository."""

    @classmethod
    def setUpClass(cls) -> None:
        from app.retrieval.corporate_event_repository import (
            PostgresCorporateEventRepository,
        )
        from app.retrieval.postgres_backend import PostgresBackend

        os.environ.setdefault("DATABASE_URL", _LIVE_DSN)
        cls.repo = PostgresCorporateEventRepository(PostgresBackend())

    def _targets(self, seed):
        from app.reasoning.corporate_event_resolver import seed_expansion_targets

        wanted, events, info = seed_expansion_targets(self.repo, [seed], limit=8)
        return wanted, events, info

    def test_samsung_contract_seed_reaches_its_termination(self) -> None:
        wanted, events, _ = self._targets("exchange_20240612800459")
        self.assertEqual(wanted, ["exchange_20250618800387"])
        self.assertEqual(events[0]["seed_member_role"].value, "contract")

    def test_samsung_termination_seed_reaches_its_contract(self) -> None:
        wanted, events, _ = self._targets("exchange_20250618800387")
        self.assertEqual(wanted, ["exchange_20240612800459"])
        self.assertEqual(events[0]["seed_member_role"].value, "termination")

    def test_samsung_sibling_contract_is_never_pulled_in(self) -> None:
        wanted, _, _ = self._targets("exchange_20240612800459")
        self.assertNotIn("exchange_20240612800468", wanted)

    def test_lges_root_and_canonical_seeds_reach_one_termination(self) -> None:
        for seed in ("exchange_20240401800927", "exchange_20251226800767"):
            with self.subTest(seed=seed):
                wanted, events, info = self._targets(seed)
                self.assertEqual(wanted, ["exchange_20251226800706"])
                # The superseded root never re-adds its own representative.
                self.assertNotIn("exchange_20251226800767", wanted)
                self.assertEqual(
                    info["seed_member_doc_ids"][seed], "exchange_20251226800767"
                )

    def test_lges_termination_seed_reaches_the_canonical_contract(self) -> None:
        wanted, _, _ = self._targets("exchange_20251226800706")
        self.assertEqual(wanted, ["exchange_20251226800767"])

    def test_outside_corpus_termination_expands_nothing(self) -> None:
        for seed in (
            "exchange_20230926800443",
            "exchange_20240826800108",
            "exchange_20260313801644",
        ):
            with self.subTest(seed=seed):
                wanted, events, info = self._targets(seed)
                self.assertEqual(wanted, [])
                self.assertEqual(events, [])
                self.assertEqual(
                    info["skipped"][0]["reason"], "lifecycle_not_resolved"
                )

    def test_trust_lifecycle_expands_in_both_directions(self) -> None:
        forward, _, _ = self._targets("major_20230727000001")
        self.assertEqual(forward, ["major_20240126000018"])
        reverse, _, _ = self._targets("major_20240126000018")
        self.assertEqual(reverse, ["major_20230727000001"])

    def test_every_resolved_lifecycle_expands_within_the_document_budget(self) -> None:
        graph = self.repo.load_graph()
        checked = 0
        for event in graph.events:
            if not event.is_resolved:
                continue
            seed = event.members[0].doc_id
            wanted, _, _ = self._targets(seed)
            self.assertLessEqual(len(wanted), 8)
            for target in wanted:
                self.assertIn(target, [m.doc_id for m in event.members])
            checked += 1
            if checked >= 40:
                break
        self.assertGreater(checked, 0)
