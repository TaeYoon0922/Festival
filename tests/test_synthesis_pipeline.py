"""Synthesis serves facts plus a code verdict, or falls through without retrying."""

import unittest
from dataclasses import replace
from unittest.mock import Mock

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import (
    DIRECT_ANSWER_TITLE,
    CitationAwareAnswerGenerator,
    GeneratedSection,
)
from app.generation.answer_lead import AnswerLeadWriter
from app.generation.answer_narration import AnswerNarrator, NarrationOutcome
from app.generation.answer_synthesis import AnswerSynthesizer
from app.generation.hcx_verbalizer import HcxSettings
from tests.test_answer_api import (
    QUESTION, _ask, _client, _pipeline, _plan_and_execution,
)
from tests.test_answer_synthesis import _settings, _StubTransport
from tests.wire_contract import structured


FACT = "공시 내용을 확인했습니다. [1]"
BAD = "효성중공업의 보유 주식 수가 더 큽니다. [1]"


def pipeline_with_reply(reply, *, verdict=None, reverse_citations=False):
    pipeline = _pipeline()
    pipeline.synthesizer = AnswerSynthesizer(
        _settings(), transport=_StubTransport(reply), enabled=True,
    )
    pipeline.narrator = Mock(spec=AnswerNarrator)
    pipeline.narrator.narrate.return_value = NarrationOutcome(None, "disabled")
    pipeline.lead_writer = AnswerLeadWriter(HcxSettings(enabled=False))
    if verdict is not None:
        plan, execution = _plan_and_execution()
        result = AgentOrchestrator().run(QUESTION, plan, execution)
        generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
        citations = generated.citations
        if reverse_citations:
            citations = tuple(
                replace(citation, citation_id=f"[{len(citations) - index}]")
                for index, citation in enumerate(citations)
            )
        pipeline.generator = Mock(spec=CitationAwareAnswerGenerator)
        pipeline.generator.generate.return_value = replace(
            generated,
            citations=citations,
            sections=(*generated.sections, GeneratedSection(
                title=DIRECT_ANSWER_TITLE, content=verdict, citations=("[1]", "[2]"),
            )),
            answer_text=f"{generated.answer_text}\n\n{verdict}",
        )
    return pipeline


class SynthesisPipelineTests(unittest.TestCase):
    def test_success_appends_each_code_verdict_before_complete_citations(self):
        verdicts = (
            "매출액 기준으로는 A전자가 B전자보다 200백만원 더 큽니다. [1] [2]",
            "두 회사의 매출액은 같습니다. [1] [2]",
            "A전자와 B전자의 매출액은 공시 원문에 단위 표기가 없어 "
            "어느 쪽이 큰지 직접 비교하지 않았습니다. [1] [2]",
        )
        for verdict in verdicts:
            with self.subTest(verdict=verdict):
                pipeline = pipeline_with_reply(FACT, verdict=verdict, reverse_citations=True)
                response = _ask(_client(lambda: pipeline))

                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()), 5)
                self.assertTrue(all(isinstance(v, str) for v in response.json().values()))
                payload = structured(response)
                answer = payload["answer"]
                expected = verdict.replace("[1] [2]", "[2] [1]")
                self.assertTrue(answer.startswith(f"{FACT}\n\n{expected}\n\n인용\n"))
                self.assertEqual(answer.count(expected), 1)
                # The model only cited [1]; the appended verdict also cites [2].
                self.assertIn("[1]\ndoc_id: h23\nchunk_id: h23:ch_report", answer)
                self.assertIn("[2]\ndoc_id: h24\nchunk_id: h24:ch_report", answer)
                self.assertEqual(payload["think_trace"]["answer_synthesis"]["status"], "success")
                pipeline.narrator.narrate.assert_not_called()

    def test_success_without_a_comparison_keeps_the_synthesized_answer(self):
        pipeline = pipeline_with_reply(FACT)

        payload = structured(_ask(_client(lambda: pipeline)))

        self.assertTrue(payload["answer"].startswith(f"{FACT}\n\n인용\n"))
        pipeline.narrator.narrate.assert_not_called()

    def test_rejected_verdict_falls_through_to_narration(self):
        pipeline = pipeline_with_reply(BAD)
        pipeline.narrator.narrate.return_value = NarrationOutcome("나레이션 답변 [1]", "success")

        payload = structured(_ask(_client(lambda: pipeline)))

        self.assertEqual(payload["answer"], "나레이션 답변 [1]")
        self.assertEqual(payload["think_trace"]["answer_synthesis"]["status"],
                         "rejected:comparison_verdict:더 큽")
        self.assertNotIn(BAD, str(payload))
        pipeline.narrator.narrate.assert_called_once()
        self.assertEqual(pipeline.synthesizer.call_count, 1)
        self.assertEqual(len(pipeline.synthesizer.transport.payloads), 1)

    def test_rejected_narration_preserves_the_deterministic_answer(self):
        pipeline = pipeline_with_reply(BAD)
        pipeline.synthesizer.enabled = False
        baseline = structured(_ask(_client(lambda: pipeline)))["answer"]
        pipeline.synthesizer.enabled = True
        pipeline.narrator.narrate.reset_mock()
        pipeline.narrator.narrate.return_value = NarrationOutcome(None, "rejected:placeholder_integrity")

        payload = structured(_ask(_client(lambda: pipeline)))

        self.assertEqual(payload["answer"], baseline)
        self.assertNotIn(BAD, str(payload))
        pipeline.narrator.narrate.assert_called_once()
        self.assertEqual(pipeline.synthesizer.call_count, 1)
