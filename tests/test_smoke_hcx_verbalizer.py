from __future__ import annotations

import json
import unittest

from scripts.smoke_hcx_verbalizer import (
    FIXTURE_COMPANY,
    FIXTURE_TEXT,
    _diagnostic,
    _RecordingTransport,
)
from app.generation.answer_validator import validate_verbalized_answer


DIAGNOSTIC_KEYS = {
    "reference_text",
    "raw_hcx_candidate",
    "reference_citations",
    "candidate_citations",
    "reference_numeric_tokens",
    "candidate_numeric_tokens",
    "validator_reason",
    "citations_only_in_reference",
    "citations_only_in_candidate",
    "numbers_only_in_reference",
    "numbers_only_in_candidate",
}


def _diagnose(candidate: str | None) -> dict:
    result = (
        validate_verbalized_answer(
            candidate, reference=FIXTURE_TEXT, required_terms=(FIXTURE_COMPANY,)
        )
        if candidate is not None
        else None
    )
    return _diagnostic(candidate, result)


class _EchoTransport:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.calls.append(
            {"url": url, "headers": dict(headers), "timeout_seconds": timeout_seconds}
        )
        return self.response


class RecordingTransportTests(unittest.TestCase):
    def test_passes_the_response_through(self) -> None:
        inner = _EchoTransport({"choices": [{"message": {"content": "안녕"}}]})
        recorder = _RecordingTransport(inner)

        returned = recorder.post_json(
            "https://clova.example",
            headers={"Authorization": "Bearer secret-key"},
            payload={"model": "HCX-005"},
            timeout_seconds=5.0,
        )

        self.assertEqual(returned, inner.response)
        self.assertEqual(recorder.response, inner.response)

    def test_forwards_headers_without_retaining_them(self) -> None:
        inner = _EchoTransport({"choices": []})
        recorder = _RecordingTransport(inner)

        recorder.post_json(
            "https://clova.example",
            headers={"Authorization": "Bearer secret-key"},
            payload={},
            timeout_seconds=5.0,
        )

        # The inner transport still receives the credential it needs...
        self.assertEqual(
            inner.calls[0]["headers"], {"Authorization": "Bearer secret-key"}
        )
        # ...but the recorder keeps only the reply.
        self.assertNotIn("secret-key", repr(vars(recorder)))


class DiagnosticTests(unittest.TestCase):
    def test_reports_a_dropped_citation(self) -> None:
        candidate = FIXTURE_TEXT.replace("[1]", "")

        payload = _diagnose(candidate)

        self.assertEqual(payload["validator_reason"], "citation_marker_changed")
        self.assertEqual(payload["reference_citations"], ["1"])
        self.assertEqual(payload["candidate_citations"], [])
        self.assertEqual(payload["citations_only_in_reference"], ["1"])
        self.assertEqual(payload["citations_only_in_candidate"], [])

    def test_reports_an_added_citation(self) -> None:
        payload = _diagnose(FIXTURE_TEXT + " 추가 근거입니다.[2]")

        self.assertEqual(payload["validator_reason"], "citation_marker_changed")
        self.assertEqual(payload["citations_only_in_candidate"], ["2"])
        self.assertEqual(payload["citations_only_in_reference"], [])

    def test_reports_a_relocated_citation_as_unchanged(self) -> None:
        candidate = "[1] 국민연금기금의 효성중공업 보유주식수는 2023년 03월 07일 기준 655,490주입니다."

        payload = _diagnose(candidate)

        self.assertEqual(payload["citations_only_in_reference"], [])
        self.assertEqual(payload["citations_only_in_candidate"], [])

    def test_reports_a_changed_number(self) -> None:
        payload = _diagnose(FIXTURE_TEXT.replace("655,490", "655,491"))

        self.assertEqual(payload["validator_reason"], "numeric_token_changed")
        self.assertEqual(payload["numbers_only_in_reference"], ["655,490"])
        self.assertEqual(payload["numbers_only_in_candidate"], ["655,491"])

    def test_carries_the_reference_and_candidate_text(self) -> None:
        candidate = FIXTURE_TEXT.replace("[1]", "")

        payload = _diagnose(candidate)

        self.assertEqual(payload["reference_text"], FIXTURE_TEXT)
        self.assertEqual(payload["raw_hcx_candidate"], candidate)

    def test_handles_a_missing_candidate(self) -> None:
        payload = _diagnose(None)

        self.assertIsNone(payload["raw_hcx_candidate"])
        self.assertIsNone(payload["candidate_citations"])
        self.assertIsNone(payload["candidate_numeric_tokens"])
        self.assertIsNone(payload["validator_reason"])

    def test_exposes_no_field_outside_the_permitted_set(self) -> None:
        payload = _diagnose(FIXTURE_TEXT.replace("[1]", ""))

        self.assertTrue(set(payload).issubset(DIAGNOSTIC_KEYS))

    def test_never_carries_credentials_or_headers(self) -> None:
        body = json.dumps(_diagnose(FIXTURE_TEXT), ensure_ascii=False).lower()

        for forbidden in ("authorization", "bearer", "api_key", "apikey", "x-ncp"):
            self.assertNotIn(forbidden, body)


class ServedVersusCandidateTests(unittest.TestCase):
    """The served answer says nothing about whether HCX succeeded."""

    def test_validating_the_fallback_is_always_true(self) -> None:
        served = validate_verbalized_answer(
            FIXTURE_TEXT, reference=FIXTURE_TEXT, required_terms=(FIXTURE_COMPANY,)
        )

        self.assertTrue(served.valid)

    def test_candidate_validation_is_what_explains_a_fallback(self) -> None:
        candidate = validate_verbalized_answer(
            FIXTURE_TEXT.replace("[1]", ""),
            reference=FIXTURE_TEXT,
            required_terms=(FIXTURE_COMPANY,),
        )

        self.assertFalse(candidate.valid)
        self.assertEqual(candidate.reason, "citation_marker_changed")


if __name__ == "__main__":
    unittest.main()
