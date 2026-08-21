from __future__ import annotations

import unittest

from app.generation.answer_validator import (
    ValidationPolicy,
    extract_citation_markers,
    extract_numeric_tokens,
    validate_verbalized_answer,
)


REFERENCE = (
    "국민연금기금의 효성중공업 보유주식수는 2023년 03월 07일 기준 "
    "655,490주입니다.[1] 보유비율은 7.02%입니다.[2]"
)


def _validate(candidate: str, **kwargs) -> object:
    return validate_verbalized_answer(candidate, reference=REFERENCE, **kwargs)


class NumericPreservationTests(unittest.TestCase):
    def test_faithful_rewording_passes(self) -> None:
        candidate = (
            "효성중공업에 대한 국민연금기금의 보유주식수는 2023년 03월 07일 기준으로 "
            "655,490주입니다.[1] 이때 보유비율은 7.02%입니다.[2]"
        )

        result = _validate(candidate)

        self.assertTrue(result.valid)
        self.assertIsNone(result.reason)

    def test_hallucinated_number_is_rejected(self) -> None:
        candidate = REFERENCE + " 전분기 대비 12,345주 늘었습니다."

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "numeric_token_changed")
        self.assertIn("12,345", result.changed_tokens)

    def test_dropped_number_is_rejected(self) -> None:
        candidate = (
            "국민연금기금의 효성중공업 보유주식수는 2023년 03월 07일 기준 "
            "655,490주입니다.[1] 보유비율도 공시되어 있습니다.[2]"
        )

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "numeric_token_changed")

    def test_rounded_number_is_rejected(self) -> None:
        candidate = REFERENCE.replace("655,490주", "약 65만 주")

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "numeric_token_changed")

    def test_separator_removal_is_rejected(self) -> None:
        candidate = REFERENCE.replace("655,490", "655490")

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "numeric_token_changed")

    def test_reformatted_date_is_rejected(self) -> None:
        candidate = REFERENCE.replace("2023년 03월 07일", "2023-03-07")

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "numeric_token_changed")

    def test_dropped_percent_sign_is_rejected(self) -> None:
        candidate = REFERENCE.replace("7.02%", "7.02")

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "numeric_token_changed")

    def test_repeating_a_number_is_allowed(self) -> None:
        candidate = REFERENCE + " 다시 정리하면 655,490주입니다."

        result = _validate(candidate)

        self.assertTrue(result.valid)


class CitationPreservationTests(unittest.TestCase):
    def test_removed_citation_is_rejected(self) -> None:
        candidate = REFERENCE.replace("[2]", "")

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "citation_marker_changed")
        self.assertIn("2", result.changed_tokens)

    def test_added_citation_is_rejected(self) -> None:
        candidate = REFERENCE + " 추가 근거입니다.[3]"

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "citation_marker_changed")
        self.assertIn("3", result.changed_tokens)

    def test_citation_digits_are_not_read_as_facts(self) -> None:
        self.assertEqual(extract_citation_markers("값 5건입니다.[1][2]"), {"1", "2"})
        self.assertEqual(extract_numeric_tokens("값 5건입니다.[1][2]"), {"5"})


class LanguageAndEntityTests(unittest.TestCase):
    def test_introduced_investment_language_is_rejected(self) -> None:
        candidate = REFERENCE + " 이에 따라 매수 의견을 제시합니다."

        result = _validate(candidate)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "forbidden_investment_language")
        self.assertIn("매수", result.changed_tokens)

    def test_forecast_language_already_in_the_reference_is_kept(self) -> None:
        reference = "회사는 성장할 계획을 공시했습니다.[1]"

        result = validate_verbalized_answer(
            "공시에 따르면 회사는 성장할 계획입니다.[1]", reference=reference
        )

        self.assertTrue(result.valid)

    def test_missing_required_entity_is_rejected(self) -> None:
        candidate = REFERENCE.replace("효성중공업", "해당 회사")

        result = _validate(candidate, required_terms=["효성중공업"])

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "entity_missing")
        self.assertEqual(result.changed_tokens, ("효성중공업",))

    def test_required_entity_absent_from_reference_is_not_required(self) -> None:
        result = _validate(REFERENCE, required_terms=["삼성전자"])

        self.assertTrue(result.valid)


class BoundsTests(unittest.TestCase):
    def test_empty_candidate_is_rejected(self) -> None:
        result = _validate("   ")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "empty_output")

    def test_overlong_candidate_is_rejected(self) -> None:
        result = validate_verbalized_answer(
            "가" * 100, reference="짧은 답변", policy=ValidationPolicy(max_length_ratio=2.0)
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "length_exceeded")

    def test_policy_rejects_a_non_positive_ratio(self) -> None:
        with self.assertRaises(ValueError):
            ValidationPolicy(max_length_ratio=0.0)


if __name__ == "__main__":
    unittest.main()
