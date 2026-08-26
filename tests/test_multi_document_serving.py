"""P0-C Step 5: serving integration.

Two properties carry the whole step:

* a question P0-C declines is served *exactly* as before -- same
  ``retrieved_context``, same ``think_trace``, no P0-C block, no repository call;
* a question P0-C accepts states its counts from the metadata enumeration, never
  from the documents that happened to fit in the context budget.

The response contract stays exactly five top-level fields.
"""

from __future__ import annotations

import csv
import json
import unicodedata
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.api.pipeline import AnswerPipeline
from app.api.settings import ApiSettings
from app.reasoning.corporate_event import CorporateEventState
from app.reasoning.corporate_event_graph import CorporateEventGraphUnavailable
from app.reasoning.multi_document_evidence import (
    LIFECYCLE_EXISTS,
    LIFECYCLE_NONE,
    LIFECYCLE_NO_MEMBERS,
    LIFECYCLE_UNDETERMINED,
    MAX_MULTI_DOC_EVIDENCE,
    PROVENANCE_KEY,
    MultiDocumentEvidenceBuilder,
    lifecycle_answer,
)
from app.reasoning.multi_document_executor import MultiDocumentExecutor
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


REPO = Path(__file__).resolve().parents[1]
GOLD60 = (
    REPO / "reports/evaluation/gold60/2026-08-21-agent-90pct/gold60_agent_questions.jsonl"
)
UNIVERSE = REPO / "data/corpus/universe.csv"
CORP = "00126478"

COUNT_Q = "삼성중공업이 2025년에 체결한 주요 공급계약은 모두 몇 건인가?"
LIFECYCLE_Q = "삼성중공업이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"
RECEIPT_Q = "삼성중공업이 2025년에 공시한 공급계약은 모두 몇 건인가?"


def _resolver():
    names = {}
    if UNIVERSE.exists():
        with UNIVERSE.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for key in ("corp_name", "listed_name"):
                    value = (row.get(key) or "").replace(" ", "")
                    if value:
                        names[unicodedata.normalize("NFC", value)] = row["corp_code"]
    else:  # pragma: no cover
        names = {"삼성중공업": CORP}

    def resolve(query: str):
        text = unicodedata.normalize("NFC", (query or "").replace(" ", ""))
        best = None
        for name, code in names.items():
            if name in text and (best is None or len(name) > len(best[0])):
                best = (name, code)
        return {"corp_code": best[1], "corp_name": best[0]} if best else None

    return resolve


def _state(event_id, *, terminated=False, source="single_document", role="contract",
           doc_id=None):
    return CorporateEventState(
        doc_id=doc_id or f"contract_{event_id}",
        event_id=event_id,
        corp_code=CORP,
        event_family="supply_contract",
        member_role=role,
        lifecycle_status="terminated" if terminated else "open",
        resolution_status="unresolved",
        canonical_doc_id=doc_id or f"contract_{event_id}",
        member_count=1,
        opened_at="2025-03-01",
        resolution_source=source,
    )


class _EventRepo:
    def __init__(self, contracts=(), terminations=()):
        self.contracts = tuple(contracts)
        self.terminations = tuple(terminations)
        self.calls = []

    def enumerate_events(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("member_role") == "termination":
            return self.terminations
        return self.contracts

    def event_states(self, doc_ids):
        self.calls.append({"event_states": tuple(doc_ids)})
        return {s.event_id: s for s in self.contracts if s.event_id in set(doc_ids)}


class _Store:
    """Metadata + chunk backend for hydration."""

    def __init__(self):
        self.fetched = []

    def fetch_documents(self, doc_ids):
        self.fetched.append(tuple(doc_ids))
        return [
            CandidateDocument(
                doc_id=d,
                metadata={"report_nm": f"보고서 {d}", "rcept_dt": "2025-03-02",
                          "corp_code": CORP},
                metadata_match=MetadataMatch(),
            )
            for d in doc_ids
        ]

    def get_candidate_chunks(self, documents):
        out = []
        for document in documents:
            for i in range(3):
                out.append(
                    CandidateChunk(
                        chunk_id=f"{document.doc_id}#c{i}",
                        doc_id=document.doc_id,
                        chunk={
                            "content": f"body {document.doc_id} {i}",
                            "retrieval_text": f"body {document.doc_id} {i}",
                            "report_nm": document.metadata.get("report_nm"),
                            "rcept_dt": document.metadata.get("rcept_dt"),
                            "corp_code": CORP,
                            "chunk_type": "text",
                            "section_path": ["s"],
                            "source_refs": [],
                            "provenance": {},
                        },
                        metadata_match=MetadataMatch(),
                    )
                )
        return out


class _StubRetrieval:
    def execute(self, plan):
        chunks, results = [], []
        for i in range(3):
            cid = f"base_{i}"
            chunks.append(
                CandidateChunk(
                    chunk_id=cid, doc_id=f"base_doc_{i}",
                    chunk={"content": f"base {i}", "retrieval_text": f"base {i}",
                           "report_nm": "기존", "rcept_dt": "2025-01-01",
                           "corp_code": CORP, "chunk_type": "text",
                           "section_path": ["s"], "source_refs": [], "provenance": {}},
                    metadata_match=MetadataMatch(),
                )
            )
            results.append(
                RetrievalResult(chunk_id=cid, doc_id=f"base_doc_{i}",
                                bm25_score=1.0, rank=i + 1, metadata_match={})
            )
        return SimpleNamespace(plan=plan, documents=[], chunks=chunks, results=results,
                               correction_expansion={}, event_expansion={})


def _pipeline(event_repo=None, *, store=None, evidence=True, executor=None):
    store = store if store is not None else _Store()
    builder = (
        MultiDocumentEvidenceBuilder(
            event_repository=event_repo, metadata_backend=store, chunk_backend=store
        )
        if evidence
        else None
    )
    return AnswerPipeline(
        settings=ApiSettings(),
        understanding=QueryUnderstanding(company_resolver=_resolver()),
        executor=_StubRetrieval(),
        multi_document_planner=MultiDocumentPlanner(),
        multi_document_executor=executor
        or MultiDocumentExecutor(event_repository=event_repo),
        multi_document_evidence=builder,
    )


def _trace(payload):
    return payload["think_trace"].get("multi_document_planner")


# ------------------------------------------------------------- non-engagement


class NonEngagementTests(unittest.TestCase):
    def test_gold60_serving_output_is_unchanged(self) -> None:
        """§23 -- P0-C wired vs not wired must produce identical payloads."""

        if not GOLD60.exists():
            self.skipTest("gold60 artifact not present")

        class _Boom:
            def enumerate_events(self, **kwargs):
                raise AssertionError("P0-C reached a repository for a Gold60 question")

        baseline = AnswerPipeline(
            settings=ApiSettings(),
            understanding=QueryUnderstanding(company_resolver=_resolver()),
            executor=_StubRetrieval(),
        )
        wired = _pipeline(_Boom())
        rows = [
            json.loads(line)
            for line in GOLD60.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 60)
        for row in rows:
            before = baseline.answer(row["question_id"], row["question"])
            after = wired.answer(row["question_id"], row["question"])
            self.assertEqual(after, before, row["question_id"])
            self.assertIsNone(_trace(after), row["question_id"])

    def test_five_field_contract_holds(self) -> None:
        payload = _pipeline(_EventRepo([_state("e1")])).answer("Q", COUNT_Q)
        self.assertEqual(
            sorted(payload),
            ["answer", "question", "question_id", "retrieved_context", "think_trace"],
        )

    def test_declined_question_adds_no_stage(self) -> None:
        payload = _pipeline(_EventRepo()).answer("Q", "삼성중공업 2025년 공급계약")
        self.assertIsNone(_trace(payload))
        self.assertNotIn("multi_document_planner", payload["think_trace"]["stages"])


# ------------------------------------------------------------------- applied


class AppliedServingTests(unittest.TestCase):
    def test_i1_cardinality_uses_the_logical_count(self) -> None:
        repo = _EventRepo([_state(f"e{i:02d}") for i in range(14)])
        payload = _pipeline(repo).answer("I1", COUNT_Q)
        trace = _trace(payload)
        self.assertTrue(trace["applied"])
        self.assertEqual(trace["plan_type"], "enumeration")
        self.assertEqual(trace["logical_count"], 14)
        self.assertTrue(trace["complete"])
        self.assertEqual(trace["stop_reason"], "all_slots_complete")

    def test_i2_lifecycle_positive(self) -> None:
        contracts = [_state(f"e{i:02d}") for i in range(12)] + [
            _state("e12", terminated=True), _state("e13", terminated=True)
        ]
        terminations = [
            _state("e12", role="termination", doc_id="term_e12"),
            _state("e13", role="termination", doc_id="term_e13"),
        ]
        repo = _EventRepo(contracts, terminations)
        payload = _pipeline(repo).answer("I2", LIFECYCLE_Q)
        trace = _trace(payload)
        self.assertEqual(trace["plan_type"], "enumeration_plus_event")
        self.assertEqual(trace["logical_count"], 14)
        self.assertEqual(trace["terminated_count"], 2)
        self.assertEqual(trace["open_count"], 12)
        self.assertEqual(trace["lifecycle_answer"], LIFECYCLE_EXISTS)
        self.assertTrue(trace["complete"])
        # Both sides of the termination claim are citable.
        doc_ids = {row["doc_id"] for row in payload["retrieved_context"]}
        self.assertIn("term_e12", doc_ids)
        self.assertIn("term_e13", doc_ids)
        self.assertIn("contract_e12", doc_ids)

    def test_i3_lifecycle_negative(self) -> None:
        repo = _EventRepo([_state(f"e{i:02d}") for i in range(14)])
        trace = _trace(_pipeline(repo).answer("I3", LIFECYCLE_Q))
        self.assertEqual(trace["terminated_count"], 0)
        self.assertEqual(trace["unresolved_count"], 0)
        self.assertTrue(trace["complete"])
        self.assertEqual(trace["lifecycle_answer"], LIFECYCLE_NONE)

    def test_i4_unresolved_is_never_a_confident_no(self) -> None:
        contracts = [_state(f"e{i:02d}") for i in range(13)] + [
            _state("e13", source="related_reference_not_in_corpus")
        ]
        trace = _trace(_pipeline(_EventRepo(contracts)).answer("I4", LIFECYCLE_Q))
        self.assertEqual(trace["terminated_count"], 0)
        self.assertGreater(trace["unresolved_count"], 0)
        self.assertFalse(trace["complete"])
        self.assertEqual(trace["lifecycle_answer"], LIFECYCLE_UNDETERMINED)
        self.assertNotEqual(trace["lifecycle_answer"], LIFECYCLE_NONE)

    def test_i5_empty_set(self) -> None:
        trace = _trace(_pipeline(_EventRepo([])).answer("I5", LIFECYCLE_Q))
        self.assertEqual(trace["logical_count"], 0)
        self.assertTrue(trace["complete"])
        self.assertEqual(trace["lifecycle_answer"], LIFECYCLE_NO_MEMBERS)

    def test_i6_receipt_axis_never_substitutes_opened_at(self) -> None:
        class _Backend(_Store):
            def enumerate_disclosures(self, **kwargs):
                self.kwargs = kwargs
                return tuple(
                    CandidateDocument(doc_id=f"d{i}", metadata={}, metadata_match=MetadataMatch())
                    for i in range(5)
                )

        class _Corr:
            def document_states(self, doc_ids):
                return {}

        store = _Backend()
        executor = MultiDocumentExecutor(
            disclosure_backend=store, correction_repository=_Corr()
        )
        pipeline = _pipeline(None, store=store, executor=executor)
        trace = _trace(pipeline.answer("I6", RECEIPT_Q))
        self.assertEqual(trace["logical_count"], 5)
        self.assertEqual(store.kwargs["doc_subtype"], "단일판매공급계약체결")
        self.assertEqual(store.kwargs["date_from"], "2025-01-01")
        self.assertEqual(store.kwargs["date_to"], "2026-01-01")
        slot = trace["slots"][0]
        self.assertEqual(slot["date_field"], "rcept_dt")
        self.assertNotEqual(slot["date_field"], "opened_at")

    def test_i7_bare_fallback_is_visible_in_the_trace(self) -> None:
        repo = _EventRepo([_state("e1")])
        payload = _pipeline(repo).answer(
            "I7", "삼성중공업이 2025년에 체결한 주요 계약은 모두 몇 건인가?"
        )
        self.assertEqual(_trace(payload)["family_resolution"], "bare_contract_fallback")
        # Auditable in the trace, invisible to the reader: an internal
        # diagnostic must never surface as user-facing wording.
        answer = payload["answer"]
        for leaked in ("fallback", "bare_contract", "family_resolution", "slot"):
            self.assertNotIn(leaked, answer, leaked)


# ------------------------------------------------------------------- details


class EvidenceAndTraceTests(unittest.TestCase):
    def test_trace_never_carries_identifiers(self) -> None:
        contracts = [_state(f"e{i:02d}") for i in range(4)] + [
            _state("e04", terminated=True)
        ]
        payload = _pipeline(
            _EventRepo(contracts, [_state("e04", role="termination", doc_id="term_e04")])
        ).answer("T", LIFECYCLE_Q)
        text = json.dumps(_trace(payload), ensure_ascii=False)
        for forbidden in ("expected_ids", "found_ids", "terminated_ids",
                          "open_ids", "e00", "term_e04"):
            self.assertNotIn(forbidden, text, forbidden)

    def test_added_rows_carry_planner_provenance(self) -> None:
        repo = _EventRepo([_state("e1")])
        payload = _pipeline(repo).answer("P", COUNT_Q)
        added = [
            row for row in payload["retrieved_context"]
            if row["doc_id"].startswith("contract_")
        ]
        self.assertTrue(added)
        self.assertGreaterEqual(len(payload["retrieved_context"]), 4)

    def test_existing_retrieval_rows_keep_their_rank_and_order(self) -> None:
        repo = _EventRepo([_state("e1")])
        payload = _pipeline(repo).answer("R", COUNT_Q)
        base = [row for row in payload["retrieved_context"]
                if row["doc_id"].startswith("base_doc_")]
        self.assertEqual([row["rank"] for row in base], [1, 2, 3])
        self.assertEqual([row["doc_id"] for row in base],
                         ["base_doc_0", "base_doc_1", "base_doc_2"])

    def test_hydration_is_bounded(self) -> None:
        repo = _EventRepo([_state(f"e{i:03d}") for i in range(60)])
        payload = _pipeline(repo).answer("B", COUNT_Q)
        added = [row for row in payload["retrieved_context"]
                 if row["doc_id"].startswith("contract_")]
        self.assertLessEqual(len(added), MAX_MULTI_DOC_EVIDENCE)
        # Completeness is still measured over the whole set.
        self.assertEqual(_trace(payload)["logical_count"], 60)

    def test_hydrated_rows_are_deduped(self) -> None:
        repo = _EventRepo([_state("e1"), _state("e1")])
        payload = _pipeline(repo).answer("D", COUNT_Q)
        chunk_ids = [row["chunk_id"] for row in payload["retrieved_context"]]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_citations_keep_document_provenance(self) -> None:
        repo = _EventRepo([_state("e1")])
        payload = _pipeline(repo).answer("C", COUNT_Q)
        for row in payload["retrieved_context"]:
            if row["doc_id"].startswith("contract_"):
                self.assertTrue(row["report_nm"])
                self.assertTrue(row["rcept_dt"])
                self.assertNotIn("evt_", row["doc_id"])

    def test_stages_name_the_planner_when_applied(self) -> None:
        repo = _EventRepo([_state("e1")])
        payload = _pipeline(repo).answer("S", COUNT_Q)
        stages = payload["think_trace"]["stages"]
        self.assertIn("multi_document_planner", stages)
        self.assertIn("multi_document_executor", stages)


class DegradationTests(unittest.TestCase):
    def test_repository_unavailable_falls_back_to_existing_evidence(self) -> None:
        class _Down:
            def enumerate_events(self, **kwargs):
                raise CorporateEventGraphUnavailable("db/007 not applied")

        payload = _pipeline(_Down()).answer("U", COUNT_Q)
        self.assertIsNone(_trace(payload))
        self.assertEqual(len(payload["retrieved_context"]), 3)
        self.assertTrue(payload["answer"])

    def test_programming_errors_propagate(self) -> None:
        class _Buggy:
            def enumerate_events(self, **kwargs):
                raise TypeError("bad call")

        with self.assertRaises(Exception):
            _pipeline(_Buggy()).answer("E", COUNT_Q)


class LifecycleSemanticsTests(unittest.TestCase):
    """§13 -- "checked and none" and "could not check" are different answers."""

    def test_truth_table(self) -> None:
        cases = [
            (dict(logical_count=5, terminated_count=2, unresolved_count=0, complete=True),
             LIFECYCLE_EXISTS),
            (dict(logical_count=5, terminated_count=0, unresolved_count=0, complete=True),
             LIFECYCLE_NONE),
            (dict(logical_count=5, terminated_count=0, unresolved_count=1, complete=False),
             LIFECYCLE_UNDETERMINED),
            (dict(logical_count=5, terminated_count=0, unresolved_count=0, complete=False),
             LIFECYCLE_UNDETERMINED),
            (dict(logical_count=0, terminated_count=0, unresolved_count=0, complete=True),
             LIFECYCLE_NO_MEMBERS),
            # A termination that *was* found still answers "exists" even if
            # another member is unresolved.
            (dict(logical_count=5, terminated_count=1, unresolved_count=1, complete=False),
             LIFECYCLE_EXISTS),
        ]
        for kwargs, expected in cases:
            self.assertEqual(lifecycle_answer(**kwargs), expected, kwargs)


if __name__ == "__main__":
    unittest.main()
