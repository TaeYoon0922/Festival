from __future__ import annotations

import json
import unittest

from app.generation.hcx_verbalizer import HcxSettings
from app.reasoning.clarification_request import ClarificationCandidate
from app.reasoning.hcx_clarification_classifier import (
    HCX_CLARIFICATION_SYSTEM_PROMPT,
    HcxClarificationClassifier,
    parse_hcx_clarification_result,
)


def _candidate(identifier: str, label: str) -> ClarificationCandidate:
    return ClarificationCandidate(
        identifier,
        label,
        "metric",
        "deterministic_test_taxonomy",
        identifier,
    )


CANDIDATES = (
    _candidate("M1", "금융비용"),
    _candidate("M2", "기타비용"),
)


def _settings(*, enabled: bool = True) -> HcxSettings:
    return HcxSettings(
        enabled=enabled,
        endpoint="https://clova.example/v1/chat/completions",
        api_key="test-key",
        max_tokens=900,
    )


class _Transport:
    def __init__(self, content=None, *, error=None) -> None:
        self.content = content
        self.error = error
        self.calls = 0
        self.payload = None

    def post_json(self, url, *, headers, payload, timeout_seconds):
        del url, headers, timeout_seconds
        self.calls += 1
        self.payload = payload
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.content}}]}


def _payload(decision: str, ids=(), reason="bounded classification") -> str:
    return json.dumps(
        {"decision": decision, "candidate_ids": list(ids), "reason": reason}
    )


class HcxClarificationClassifierTests(unittest.TestCase):
    def test_prompt_forbids_candidate_invention_and_answer_generation(self) -> None:
        self.assertIn("only", HCX_CLARIFICATION_SYSTEM_PROMPT)
        self.assertIn("candidate_interpretations", HCX_CLARIFICATION_SYSTEM_PROMPT)
        self.assertIn("Do not add facts", HCX_CLARIFICATION_SYSTEM_PROMPT)
        self.assertIn("Do not", HCX_CLARIFICATION_SYSTEM_PROMPT)
        self.assertNotIn("unsupported", HCX_CLARIFICATION_SYSTEM_PROMPT)

    def test_valid_resolved_candidate_is_accepted(self) -> None:
        transport = _Transport(_payload("resolved", ("M1",)))
        classifier = HcxClarificationClassifier(_settings(), transport=transport)

        outcome = classifier.classify("영업외비용은?", CANDIDATES)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.result.decision, "resolved")
        self.assertEqual(outcome.result.candidate_ids, ("M1",))
        self.assertEqual(transport.calls, 1)
        self.assertEqual(transport.payload["temperature"], 0.0)
        self.assertEqual(transport.payload["max_tokens"], 256)
        supplied = json.loads(transport.payload["messages"][1]["content"])
        self.assertEqual(
            supplied["candidate_interpretations"],
            [{"id": "M1", "label": "금융비용"}, {"id": "M2", "label": "기타비용"}],
        )

    def test_clarify_is_accepted(self) -> None:
        outcome = HcxClarificationClassifier(
            _settings(), transport=_Transport(_payload("clarify", ("M1", "M2")))
        ).classify("질문", CANDIDATES)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.result.decision, "clarify")
        self.assertEqual(outcome.result.candidate_ids, ("M1", "M2"))

    def test_legacy_unsupported_is_rejected_and_can_only_fall_back(self) -> None:
        outcome = HcxClarificationClassifier(
            _settings(), transport=_Transport(_payload("unsupported", ()))
        ).classify("질문", CANDIDATES)

        self.assertEqual(outcome.status, "malformed_response")
        self.assertEqual(outcome.parse_status, "schema_invalid")
        self.assertEqual(outcome.schema_error_code, "invalid_enum")
        self.assertIsNone(outcome.result)

    def test_hallucinated_candidate_id_is_rejected(self) -> None:
        outcome = HcxClarificationClassifier(
            _settings(), transport=_Transport(_payload("resolved", ("M99",)))
        ).classify("질문", CANDIDATES)

        self.assertEqual(outcome.status, "malformed_response")
        self.assertEqual(outcome.parse_status, "schema_invalid")
        self.assertEqual(outcome.schema_error_code, "unknown_candidate")
        self.assertIsNone(outcome.result)

    def test_unknown_keys_and_invalid_cardinality_are_rejected(self) -> None:
        payloads = (
            json.dumps(
                {
                    "decision": "clarify",
                    "candidate_ids": ["M1", "M2"],
                    "reason": "ambiguous",
                    "answer": "invented",
                }
            ),
            _payload("resolved", ("M1", "M2")),
            _payload("clarify", ("M1",)),
        )
        for content in payloads:
            with self.subTest(content=content):
                outcome = HcxClarificationClassifier(
                    _settings(), transport=_Transport(content)
                ).classify("질문", CANDIDATES)

                self.assertEqual(outcome.status, "malformed_response")
                self.assertEqual(outcome.parse_status, "schema_invalid")
                self.assertIsNone(outcome.result)

    def test_malformed_json_and_markdown_fence_are_rejected(self) -> None:
        for content in ("not-json", f"```json\n{_payload('clarify', ('M1', 'M2'))}\n```"):
            with self.subTest(content=content[:8]):
                outcome = HcxClarificationClassifier(
                    _settings(), transport=_Transport(content)
                ).classify("질문", CANDIDATES)
                self.assertEqual(outcome.status, "malformed_response")
                self.assertEqual(outcome.parse_status, "invalid_json")
                self.assertIsNone(outcome.result)

    def test_timeout_and_error_return_safe_failure_statuses(self) -> None:
        cases = (
            (TimeoutError("slow"), "transport_failure"),
            (RuntimeError("broken"), "classifier_error"),
        )
        for error, expected in cases:
            with self.subTest(expected=expected):
                outcome = HcxClarificationClassifier(
                    _settings(), transport=_Transport(error=error)
                ).classify("질문", CANDIDATES)
                self.assertEqual(outcome.status, expected)
                self.assertIsNone(outcome.result)

    def test_disabled_classifier_does_not_call_transport(self) -> None:
        transport = _Transport(_payload("clarify", ("M1", "M2")))
        outcome = HcxClarificationClassifier(
            _settings(enabled=False), transport=transport
        ).classify("질문", CANDIDATES)

        self.assertEqual(outcome.status, "disabled")
        self.assertEqual(transport.calls, 0)

    def test_reason_is_validated_but_not_stored_as_semantic_truth(self) -> None:
        result = parse_hcx_clarification_result(
            _payload("resolved", ("M1",), reason="free model prose"),
            allowed_candidate_ids=("M1", "M2"),
        )

        self.assertFalse(hasattr(result, "reason"))

    def test_empty_reason_is_rejected_by_the_strict_schema(self) -> None:
        outcome = HcxClarificationClassifier(
            _settings(),
            transport=_Transport(_payload("clarify", ("M1", "M2"), reason="   ")),
        ).classify("질문", CANDIDATES)

        self.assertEqual(outcome.status, "malformed_response")
        self.assertEqual(outcome.schema_error_code, "invalid_type")


if __name__ == "__main__":
    unittest.main()
