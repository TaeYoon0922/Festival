from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from app.api.pipeline import AnswerPipeline, align_public_citations
from app.generation.answer_generator import GeneratedAnswer, GeneratedCitation
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult
from evaluation.independent_v2.evaluator import IndependentV2Evaluator


QUESTION = "공시 근거를 알려줘"


def _citation(
    marker: str,
    chunk_id: str,
    doc_id: str,
    *,
    evidence_type: str = "holding_event",
) -> GeneratedCitation:
    return GeneratedCitation(
        citation_id=marker,
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_refs=({"table_id": "t1", "row_start": 1, "row_end": 1},),
        section="근거",
        evidence_type=evidence_type,
    )


def _context(rank: int, chunk_id: str, doc_id: str) -> dict[str, object]:
    return {"rank": rank, "chunk_id": chunk_id, "doc_id": doc_id}


def _pair(rank: int, chunk_id: str, doc_id: str):
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_type": "table_projection",
        "section_path": ["근거"],
        "content": f"{doc_id} evidence",
        "retrieval_text": f"{doc_id} evidence",
        "source_refs": [{"table_id": "t1", "row_start": 1, "row_end": 1}],
    }
    return (
        CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
        RetrievalResult(chunk_id, doc_id, 1.0 / rank, rank, {}),
    )


class _Understanding:
    def __init__(self, plan: QueryPlan) -> None:
        self.plan = plan

    def understand(self, question: str, *, top_k: int):
        del question, top_k
        return self.plan


class _Executor:
    def __init__(self, execution) -> None:
        self.execution = execution

    def execute(self, plan):
        del plan
        return self.execution


class _Orchestrator:
    def run(self, question, plan, execution, *, multi_document=None):
        del question, plan, execution, multi_document
        return SimpleNamespace(
            answer_draft=object(),
            resolution={},
            task_decision=SimpleNamespace(task_type="holding_event"),
            execution_trace=("holding_event_resolver",),
            evidence_set=SimpleNamespace(selected_evidence_count=2),
        )


class _Generator:
    def __init__(self, generated: GeneratedAnswer) -> None:
        self.generated = generated

    def generate(self, draft):
        del draft
        return self.generated


def _pipeline_payload(
    generated: GeneratedAnswer,
    pairs: list[tuple[CandidateChunk, RetrievalResult]],
) -> dict[str, object]:
    plan = QueryPlan(query=QUESTION, task_type="holding_change")
    execution = SimpleNamespace(
        plan=plan,
        chunks=[candidate for candidate, _result in pairs],
        results=[result for _candidate, result in pairs],
    )
    pipeline = AnswerPipeline(
        understanding=_Understanding(plan),
        executor=_Executor(execution),
        orchestrator=_Orchestrator(),
        generator=_Generator(generated),
        verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
    )
    return pipeline.answer("citation-rank", QUESTION)


class PublicCitationAlignmentTests(unittest.TestCase):
    def test_ordinals_are_rewritten_to_exact_served_ranks_in_one_pass(self) -> None:
        citations = (
            _citation("[1]", "chunk-a", "doc-a"),
            _citation("[2]", "chunk-b", "doc-b"),
        )
        context = (
            _context(1, "unrelated-1", "other-1"),
            _context(2, "chunk-a", "doc-a"),
            _context(3, "unrelated-3", "other-3"),
            _context(4, "chunk-b", "doc-b"),
        )
        answer = "original [1]\nfinal [2]\nfooter [1]\nfooter [2]"

        aligned, diagnostic = align_public_citations(answer, citations, context)

        self.assertEqual(
            aligned,
            "original [2]\nfinal [4]\nfooter [2]\nfooter [4]",
        )
        self.assertEqual(
            [
                (row["internal_marker"], row["public_rank"])
                for row in diagnostic["mapped"]
            ],
            [(1, 2), (2, 4)],
        )
        self.assertEqual(diagnostic["status"], "aligned")

    def test_an_already_aligned_answer_is_byte_compatible(self) -> None:
        answer = "value [1]\nfooter [1]"

        aligned, diagnostic = align_public_citations(
            answer,
            (_citation("[1]", "chunk-a", "doc-a"),),
            (_context(1, "chunk-a", "doc-a"),),
        )

        self.assertEqual(aligned, answer)
        self.assertEqual(diagnostic["mapped"][0]["public_rank"], 1)

    def test_same_document_does_not_override_exact_chunk_identity(self) -> None:
        aligned, diagnostic = align_public_citations(
            "selected [1]",
            (_citation("[1]", "chunk-selected", "doc-shared"),),
            (
                _context(2, "chunk-other", "doc-shared"),
                _context(4, "chunk-selected", "doc-shared"),
            ),
        )

        self.assertEqual(aligned, "selected [4]")
        self.assertEqual(diagnostic["mapped"][0]["chunk_id"], "chunk-selected")

    def test_unmapped_marker_is_removed_instead_of_using_same_document(self) -> None:
        aligned, diagnostic = align_public_citations(
            "value [1]",
            (_citation("[1]", "chunk-missing", "doc-shared"),),
            (_context(7, "chunk-other", "doc-shared"),),
        )

        self.assertEqual(aligned.strip(), "value")
        self.assertNotRegex(aligned, r"\[\d{1,2}\]")
        self.assertEqual(diagnostic["status"], "unmapped")
        self.assertEqual(diagnostic["unmapped"][0]["reason"], "identity_not_served")

    def test_unknown_public_marker_is_removed(self) -> None:
        aligned, diagnostic = align_public_citations(
            "value [2]",
            (_citation("[1]", "chunk-a", "doc-a"),),
            (_context(1, "chunk-a", "doc-a"),),
        )

        self.assertEqual(aligned.strip(), "value")
        self.assertEqual(diagnostic["unmapped"][0]["reason"], "marker_not_generated")

    def test_duplicate_citations_collapse_to_one_served_rank_consistently(self) -> None:
        aligned, diagnostic = align_public_citations(
            "first [1], repeated [1], duplicate [2]",
            (
                _citation("[1]", "chunk-a", "doc-a"),
                _citation("[2]", "chunk-a", "doc-a"),
            ),
            (_context(4, "chunk-a", "doc-a"),),
        )

        self.assertEqual(aligned, "first [4], repeated [4], duplicate [4]")
        self.assertEqual(
            [row["public_rank"] for row in diagnostic["mapped"]],
            [4, 4],
        )

    def test_duplicate_served_identity_at_different_ranks_fails_closed(self) -> None:
        aligned, diagnostic = align_public_citations(
            "value [1]",
            (_citation("[1]", "chunk-a", "doc-a"),),
            (
                _context(2, "chunk-a", "doc-a"),
                _context(4, "chunk-a", "doc-a"),
            ),
        )

        self.assertNotRegex(aligned, r"\[\d{1,2}\]")
        self.assertEqual(
            diagnostic["unmapped"][0]["reason"], "identity_rank_ambiguous"
        )

    def test_non_holding_citation_uses_the_same_public_contract(self) -> None:
        aligned, diagnostic = align_public_citations(
            "periodic fact [1]",
            (
                _citation(
                    "[1]",
                    "periodic-chunk",
                    "periodic-doc",
                    evidence_type="periodic",
                ),
            ),
            (_context(6, "periodic-chunk", "periodic-doc"),),
        )

        self.assertEqual(aligned, "periodic fact [6]")
        self.assertEqual(diagnostic["status"], "aligned")


class PublicCitationPipelineTests(unittest.TestCase):
    def test_multi_event_answer_and_footer_use_public_ranks(self) -> None:
        citations = (
            _citation("[1]", "original-chunk", "original-doc"),
            _citation("[2]", "final-chunk", "final-doc"),
        )
        internal_answer = (
            "original event [1]\n"
            "final event [2]\n\n"
            "sources\n"
            "[1]\n"
            "doc_id: original-doc\n"
            "chunk_id: original-chunk\n"
            "[2]\n"
            "doc_id: final-doc\n"
            "chunk_id: final-chunk"
        )
        generated = GeneratedAnswer(
            question=QUESTION,
            answer_text=internal_answer,
            citations=citations,
            sections=(),
            warnings=(),
            confidence={},
            answerable=True,
        )
        pairs = [
            _pair(1, "unrelated-1", "other-1"),
            _pair(2, "original-chunk", "original-doc"),
            _pair(3, "unrelated-3", "other-3"),
            _pair(4, "final-chunk", "final-doc"),
        ]

        payload = _pipeline_payload(generated, pairs)

        self.assertEqual(
            payload["answer"],
            (
                "original event [2]\n"
                "final event [4]\n\n"
                "sources\n"
                "[2]\n"
                "doc_id: original-doc\n"
                "chunk_id: original-chunk\n"
                "[4]\n"
                "doc_id: final-doc\n"
                "chunk_id: final-chunk"
            ),
        )
        self.assertEqual(
            [(row["rank"], row["chunk_id"]) for row in payload["retrieved_context"]],
            [
                (1, "unrelated-1"),
                (2, "original-chunk"),
                (3, "unrelated-3"),
                (4, "final-chunk"),
            ],
        )
        self.assertEqual(
            IndependentV2Evaluator._cited_docs(payload),
            {"original-doc", "final-doc"},
        )
        self.assertNotIn(
            "citation_alignment_unmapped", payload["think_trace"]["warnings"]
        )

    def test_unmapped_public_marker_warns_and_fails_closed(self) -> None:
        generated = GeneratedAnswer(
            question=QUESTION,
            answer_text="supported value [1]",
            citations=(_citation("[1]", "missing-chunk", "shared-doc"),),
            sections=(),
            warnings=(),
            confidence={},
            answerable=True,
        )

        payload = _pipeline_payload(
            generated,
            [_pair(1, "different-chunk", "shared-doc")],
        )

        self.assertNotRegex(payload["answer"], r"\[\d{1,2}\]")
        self.assertIn(
            "citation_alignment_unmapped", payload["think_trace"]["warnings"]
        )
        self.assertEqual(IndependentV2Evaluator._cited_docs(payload), set())

    def test_alignment_does_not_mutate_generated_answer_or_context_input(self) -> None:
        citations = (_citation("[1]", "chunk-a", "doc-a"),)
        context = [_context(3, "chunk-a", "doc-a")]
        before = copy.deepcopy(context)

        aligned, _diagnostic = align_public_citations("value [1]", citations, context)

        self.assertEqual(aligned, "value [3]")
        self.assertEqual(context, before)
        self.assertEqual(citations[0].citation_id, "[1]")


if __name__ == "__main__":
    unittest.main()
