from __future__ import annotations

import unittest

from app.generation.compact_claim import ClaimCitation, ClaimField, CompactClaim, _render
from app.generation.lossless_verbalization import (
    CITATION_ATTACHMENT_FAILED,
    FORBIDDEN_LANGUAGE,
    GENERATED_CITATION,
    INFERENCE_MARKER,
    STRUCTURED_TEXT_LEAKAGE,
    UNPROTECTED_NUMERIC,
    attach_detached_citations,
    claim_event_count,
    detach_claim_citations,
    expected_attached_answer,
    group_event_fields,
    unprotected_numeric_tokens,
    unprotected_text_literals,
    verify_lossless_candidate,
)
from app.generation.protected_literals import PLACEHOLDER_PATTERN


def _claim(*events) -> CompactClaim:
    """Build a claim from structured field values, as the adapter does."""

    fields: list[ClaimField] = []
    citations: list[ClaimCitation] = []
    seen: set[str] = set()
    for event_index, event in enumerate(events):
        for field_index, (name, label, value, marker) in enumerate(event):
            chunk_id = f"doc-{event_index}:ch-{field_index}"
            fields.append(
                ClaimField(
                    name=name,
                    label=label,
                    value=value,
                    marker=marker,
                    chunk_id=chunk_id,
                )
            )
            if marker not in seen:
                seen.add(marker)
                citations.append(
                    ClaimCitation(
                        marker=marker,
                        chunk_id=chunk_id,
                        doc_id=f"doc-{event_index}",
                        source_refs=(
                            {"table_id": f"t{event_index}", "row_start": 1, "row_end": 1},
                        ),
                    )
                )
    return CompactClaim(
        question="테스트 질문",
        company="테스트회사",
        reporter="국민연금기금",
        fields=tuple(fields),
        citations=tuple(citations),
        deterministic_text=_render("테스트회사", "국민연금기금", fields),
    )


SINGLE = _claim(
    (
        ("reference_date", "변동일", "2023-06-30", "[1]"),
        ("after_shares", "변동 후 주식수", "1,000주", "[1]"),
    )
)

WITH_TEXT = _claim(
    (
        ("reference_date", "변동일", "2023-06-30", "[1]"),
        ("change_direction", "변동 방향", "감소", "[1]"),
    )
)

TWO_EVENTS = _claim(
    (
        ("reference_date", "변동일", "2023-06-30", "[1]"),
        ("after_shares", "변동 후 주식수", "1,000주", "[1]"),
    ),
    (
        ("reference_date", "변동일", "2024-06-30", "[2]"),
        ("after_shares", "변동 후 주식수", "2,000주", "[2]"),
    ),
)


class EventGroupingTests(unittest.TestCase):
    def test_a_repeated_field_name_starts_a_new_event(self) -> None:
        self.assertEqual(len(group_event_fields(TWO_EVENTS.fields)), 2)

    def test_one_event_stays_one_group(self) -> None:
        self.assertEqual(claim_event_count(SINGLE), 1)

    def test_event_count_matches_the_detached_attachments(self) -> None:
        self.assertEqual(
            claim_event_count(TWO_EVENTS),
            detach_claim_citations(TWO_EVENTS).event_count,
        )


class DetachmentTests(unittest.TestCase):
    def test_the_model_input_has_no_citation_marker(self) -> None:
        detached = detach_claim_citations(SINGLE)

        self.assertNotIn("[1]", detached.text)
        self.assertNotIn("[1]", detached.protection.masked)

    def test_every_field_value_is_masked(self) -> None:
        detached = detach_claim_citations(SINGLE)

        self.assertNotIn("2023-06-30", detached.protection.masked)
        self.assertNotIn("1,000", detached.protection.masked)
        self.assertEqual(len(detached.protection.literals), 2)

    def test_a_date_and_a_number_keep_their_kinds(self) -> None:
        detached = detach_claim_citations(SINGLE)

        self.assertEqual(
            [literal.kind for literal in detached.protection.literals],
            ["date", "number"],
        )

    def test_a_plain_string_field_becomes_one_opaque_text_literal(self) -> None:
        detached = detach_claim_citations(WITH_TEXT)

        kinds = [literal.kind for literal in detached.protection.literals]
        self.assertEqual(kinds, ["date", "text"])
        text_literal = detached.protection.literals[1]
        self.assertEqual(text_literal.text, "감소")
        self.assertIn("__FESTIVAL_TEXT_A__", detached.protection.masked)

    def test_citation_metadata_is_kept_per_event(self) -> None:
        detached = detach_claim_citations(TWO_EVENTS)

        self.assertEqual(
            [attachment.markers for attachment in detached.attachments],
            [("[1]",), ("[2]",)],
        )
        self.assertEqual(detached.expected_citation_sequence, ("[1]", "[2]"))


class AttachmentTests(unittest.TestCase):
    def test_a_faithful_echo_round_trips(self) -> None:
        detached = detach_claim_citations(SINGLE)

        result = attach_detached_citations(detached.protection.masked, detached)

        self.assertTrue(result.valid)
        self.assertIn("2023-06-30", result.final_answer)
        self.assertIn("1,000주", result.final_answer)
        self.assertIn("[1]", result.final_answer)

    def test_each_event_keeps_its_own_citation(self) -> None:
        detached = detach_claim_citations(TWO_EVENTS)

        result = attach_detached_citations(detached.protection.masked, detached)

        self.assertTrue(result.valid)
        self.assertEqual(result.attached_citation_count, 2)
        self.assertLess(
            result.final_answer.index("[1]"), result.final_answer.index("[2]")
        )

    def test_a_broken_placeholder_run_refuses_to_attach(self) -> None:
        detached = detach_claim_citations(SINGLE)
        broken = detached.protection.masked.replace(
            detached.protection.placeholders[0], ""
        )

        result = attach_detached_citations(broken, detached)

        self.assertFalse(result.valid)
        self.assertIsNone(result.final_answer)

    def test_the_reference_answer_is_reproducible(self) -> None:
        detached = detach_claim_citations(SINGLE)

        self.assertEqual(
            expected_attached_answer(detached),
            attach_detached_citations(
                detached.protection.masked, detached
            ).final_answer,
        )


class LeakageDetectionTests(unittest.TestCase):
    def test_numbers_outside_a_placeholder_are_found(self) -> None:
        detached = detach_claim_citations(SINGLE)

        self.assertEqual(
            unprotected_numeric_tokens(
                detached.protection.masked + " 2,000", detached.protection
            ),
            ["2,000"],
        )

    def test_a_clean_reply_reports_no_unprotected_number(self) -> None:
        detached = detach_claim_citations(SINGLE)

        self.assertEqual(
            unprotected_numeric_tokens(detached.protection.masked, detached.protection),
            [],
        )

    def test_a_regenerated_text_value_is_found(self) -> None:
        detached = detach_claim_citations(WITH_TEXT)

        self.assertEqual(
            unprotected_text_literals(
                "감소하여 " + detached.protection.masked, detached.protection
            ),
            ["감소"],
        )

    def test_a_clean_reply_reports_no_text_leakage(self) -> None:
        detached = detach_claim_citations(WITH_TEXT)

        self.assertEqual(
            unprotected_text_literals(detached.protection.masked, detached.protection),
            [],
        )


class VerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detached = detach_claim_citations(SINGLE)
        self.masked = self.detached.protection.masked

    def _verify(self, raw: str):
        return verify_lossless_candidate(raw, claim=SINGLE, detached=self.detached)

    def test_a_faithful_echo_passes(self) -> None:
        result = self._verify(self.masked)

        self.assertTrue(result.valid)
        self.assertEqual(result.final_answer, expected_attached_answer(self.detached))
        self.assertEqual(result.attached_citation_count, 1)

    def test_a_missing_placeholder_fails(self) -> None:
        result = self._verify(
            self.masked.replace(self.detached.protection.placeholders[0], "")
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "placeholder_missing")

    def test_a_generated_citation_fails(self) -> None:
        result = self._verify(self.masked + "[1]")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, GENERATED_CITATION)

    def test_an_invented_number_fails(self) -> None:
        result = self._verify("이전에는 2,000주였고 " + self.masked)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, UNPROTECTED_NUMERIC)

    def test_leaked_structured_text_fails(self) -> None:
        detached = detach_claim_citations(WITH_TEXT)

        result = verify_lossless_candidate(
            "감소하여 " + detached.protection.masked,
            claim=WITH_TEXT,
            detached=detached,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, STRUCTURED_TEXT_LEAKAGE)

    def test_investment_language_fails(self) -> None:
        result = self._verify(self.masked + " 매수를 추천합니다")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, FORBIDDEN_LANGUAGE)

    def test_an_added_conclusion_fails(self) -> None:
        result = self._verify(self.masked + " 이를 통해 추이를 알 수 있습니다")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, INFERENCE_MARKER)

    def test_a_dropped_entity_fails(self) -> None:
        date, number = self.detached.protection.placeholders

        result = self._verify(f"변동일 {date}, 변동 후 주식수 {number}")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "entity_missing")

    def test_a_failed_verification_never_returns_an_answer(self) -> None:
        for raw in (self.masked + "[1]", "이전 2,000주 " + self.masked, "   "):
            with self.subTest(raw=raw[:20]):
                result = self._verify(raw)

                self.assertFalse(result.valid)
                self.assertIsNone(result.final_answer)

    def test_the_served_answer_carries_no_placeholder(self) -> None:
        result = self._verify(self.masked)

        self.assertIsNone(PLACEHOLDER_PATTERN.search(result.final_answer))

    def test_reasons_are_stable_names(self) -> None:
        self.assertEqual(GENERATED_CITATION, "generated_citation")
        self.assertEqual(UNPROTECTED_NUMERIC, "unprotected_numeric_generation")
        self.assertEqual(STRUCTURED_TEXT_LEAKAGE, "unprotected_structured_text_leakage")
        self.assertEqual(FORBIDDEN_LANGUAGE, "forbidden_investment_language")
        self.assertEqual(INFERENCE_MARKER, "inference_marker_added")
        self.assertEqual(CITATION_ATTACHMENT_FAILED, "citation_attachment_failed")


if __name__ == "__main__":
    unittest.main()
