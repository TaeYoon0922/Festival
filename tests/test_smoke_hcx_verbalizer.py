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
from app.generation.protected_literals import (
    check_placeholder_integrity,
    protect_literals,
    restore_literals,
)


DIAGNOSTIC_KEYS = {
    "reference_text",
    "masked_reference",
    "raw_hcx_candidate",
    "restored_hcx_candidate",
    "expected_placeholders",
    "found_placeholders",
    "placeholder_integrity_reason",
    "placeholder_integrity_detail",
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

PROTECTION = protect_literals(FIXTURE_TEXT)
P_DATE, P_NUMBER, P_CITATION = PROTECTION.placeholders


def _diagnose(raw: str | None) -> dict:
    """Mirror the stages the smoke script runs, for one raw (masked) reply."""

    integrity = (
        check_placeholder_integrity(raw, PROTECTION) if raw is not None else None
    )
    restored = (
        restore_literals(raw, PROTECTION)
        if raw is not None and integrity is not None and integrity.valid
        else None
    )
    result = (
        validate_verbalized_answer(
            restored, reference=FIXTURE_TEXT, required_terms=(FIXTURE_COMPANY,)
        )
        if restored is not None
        else None
    )
    return _diagnostic(raw, restored, PROTECTION, integrity, result)


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


MASKED = PROTECTION.masked

MASKED_FAITHFUL = (
    f"효성중공업에 대한 국민연금기금의 보유주식수는 {P_DATE} 기준으로 "
    f"{P_NUMBER}주입니다.{P_CITATION}"
)


class DiagnosticTests(unittest.TestCase):
    """The reply now arrives masked, so the diagnostic reports both stages."""

    def test_faithful_reply_shows_no_drift(self) -> None:
        payload = _diagnose(MASKED_FAITHFUL)

        self.assertIsNone(payload["placeholder_integrity_reason"])
        self.assertIsNone(payload["validator_reason"])
        self.assertEqual(payload["citations_only_in_reference"], [])
        self.assertEqual(payload["citations_only_in_candidate"], [])
        self.assertEqual(payload["numbers_only_in_reference"], [])
        self.assertEqual(payload["numbers_only_in_candidate"], [])

    def test_reports_a_dropped_placeholder(self) -> None:
        payload = _diagnose(MASKED_FAITHFUL.replace(P_CITATION, ""))

        self.assertEqual(payload["placeholder_integrity_reason"], "placeholder_missing")
        self.assertIn(P_CITATION, payload["placeholder_integrity_detail"])
        self.assertIsNone(payload["restored_hcx_candidate"])
        self.assertNotIn(P_CITATION, payload["found_placeholders"])

    def test_explains_the_observed_live_failure(self) -> None:
        """The reply HCX actually returned before literals were protected."""

        payload = _diagnose(
            "국민연금기금의 효성중공업 보유주식 수는 **2023년 3월 7일** 기준 "
            "**655,490주**입니다."
        )

        self.assertEqual(payload["placeholder_integrity_reason"], "placeholder_missing")
        self.assertEqual(payload["found_placeholders"], [])
        self.assertEqual(
            payload["expected_placeholders"], [P_DATE, P_NUMBER, P_CITATION]
        )
        self.assertIsNone(payload["restored_hcx_candidate"])

    def test_reports_a_citation_the_model_typed_itself(self) -> None:
        payload = _diagnose(MASKED_FAITHFUL + " 추가 근거입니다.[2]")

        self.assertIsNone(payload["placeholder_integrity_reason"])
        self.assertEqual(payload["validator_reason"], "citation_marker_changed")
        self.assertEqual(payload["citations_only_in_candidate"], ["2"])
        self.assertEqual(payload["citations_only_in_reference"], [])

    def test_reports_a_hallucinated_number(self) -> None:
        payload = _diagnose(MASKED_FAITHFUL + " 전년 대비 99,999주 늘었습니다.")

        self.assertIsNone(payload["placeholder_integrity_reason"])
        self.assertEqual(payload["validator_reason"], "numeric_token_changed")
        self.assertEqual(payload["numbers_only_in_candidate"], ["99,999"])
        self.assertEqual(payload["numbers_only_in_reference"], [])

    def test_carries_reference_masked_and_restored_text(self) -> None:
        payload = _diagnose(MASKED_FAITHFUL)

        self.assertEqual(payload["reference_text"], FIXTURE_TEXT)
        self.assertEqual(payload["masked_reference"], MASKED)
        self.assertEqual(payload["raw_hcx_candidate"], MASKED_FAITHFUL)
        self.assertIn("2023년 03월 07일", payload["restored_hcx_candidate"])
        self.assertIn("655,490", payload["restored_hcx_candidate"])
        self.assertIn("[1]", payload["restored_hcx_candidate"])

    def test_handles_a_missing_candidate(self) -> None:
        payload = _diagnose(None)

        self.assertIsNone(payload["raw_hcx_candidate"])
        self.assertIsNone(payload["restored_hcx_candidate"])
        self.assertIsNone(payload["candidate_citations"])
        self.assertIsNone(payload["candidate_numeric_tokens"])
        self.assertIsNone(payload["validator_reason"])

    def test_exposes_no_field_outside_the_permitted_set(self) -> None:
        for reply in (MASKED_FAITHFUL, MASKED_FAITHFUL.replace(P_CITATION, ""), None):
            with self.subTest(reply=reply):
                self.assertTrue(set(_diagnose(reply)).issubset(DIAGNOSTIC_KEYS))

    def test_never_carries_credentials_or_headers(self) -> None:
        body = json.dumps(_diagnose(MASKED_FAITHFUL), ensure_ascii=False).lower()

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
