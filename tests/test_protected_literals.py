from __future__ import annotations

import unittest

from app.generation.answer_validator import (
    extract_citation_markers,
    extract_numeric_tokens,
)
from app.generation.protected_literals import (
    PLACEHOLDER_DUPLICATED,
    PLACEHOLDER_MISSING,
    PLACEHOLDER_REORDERED,
    PLACEHOLDER_UNEXPECTED,
    check_placeholder_integrity,
    contains_placeholder_syntax,
    protect_literals,
    restore_literals,
)


class ProtectionTests(unittest.TestCase):
    def test_masks_citations_dates_and_numbers(self) -> None:
        protection = protect_literals(
            "국민연금기금 보유주식수는 2023년 03월 07일 기준 655,490주입니다.[1]"
        )

        self.assertEqual(
            [literal.kind for literal in protection.literals],
            ["date", "number", "citation"],
        )
        self.assertEqual(
            [literal.text for literal in protection.literals],
            ["2023년 03월 07일", "655,490", "[1]"],
        )

    def test_labels_are_deterministic_and_ordered(self) -> None:
        protection = protect_literals("2023년 03월 07일 655,490주[1]")

        self.assertEqual(
            protection.placeholders,
            (
                "__FESTIVAL_DATE_A__",
                "__FESTIVAL_NUMBER_B__",
                "__FESTIVAL_CITATION_C__",
            ),
        )

    def test_protection_is_repeatable(self) -> None:
        text = "2023년 03월 07일 기준 655,490주입니다.[1]"

        self.assertEqual(protect_literals(text), protect_literals(text))

    def test_full_date_wins_over_its_component_numbers(self) -> None:
        protection = protect_literals("2023년 03월 07일")

        self.assertEqual(len(protection.literals), 1)
        self.assertEqual(protection.literals[0].text, "2023년 03월 07일")

    def test_hyphen_and_dot_dates_are_protected_as_one_span(self) -> None:
        for text in ("2023-03-07", "2023.03.07", "2023/03/07"):
            with self.subTest(text=text):
                protection = protect_literals(text)

                self.assertEqual(len(protection.literals), 1)
                self.assertEqual(protection.literals[0].kind, "date")
                self.assertEqual(protection.literals[0].text, text)

    def test_citation_digits_are_not_protected_separately(self) -> None:
        protection = protect_literals("근거입니다.[12]")

        self.assertEqual(len(protection.literals), 1)
        self.assertEqual(protection.literals[0].text, "[12]")

    def test_percent_and_negative_numbers_keep_their_form(self) -> None:
        protection = protect_literals("비율은 12.45%이고 변동은 -283,151주입니다.")

        self.assertEqual(
            [literal.text for literal in protection.literals], ["12.45%", "-283,151"]
        )

    def test_text_without_literals_is_unchanged(self) -> None:
        protection = protect_literals("보유 현황을 공시했습니다.")

        self.assertEqual(protection.masked, protection.original)
        self.assertEqual(protection.literals, ())

    def test_more_than_twenty_six_literals_keep_digit_free_labels(self) -> None:
        protection = protect_literals(" ".join(str(index) for index in range(30)))

        self.assertEqual(len(protection.literals), 30)
        self.assertTrue(protection.placeholders[26].endswith("_AA__"))
        for placeholder in protection.placeholders:
            self.assertFalse(any(character.isdigit() for character in placeholder))


class PlaceholderInvisibilityTests(unittest.TestCase):
    """Placeholders must not look like facts to the validator's tokenizers."""

    def setUp(self) -> None:
        self.masked = protect_literals(
            "2023년 03월 07일 기준 655,490주입니다.[1]"
        ).masked

    def test_validator_sees_no_numbers_in_masked_text(self) -> None:
        self.assertEqual(extract_numeric_tokens(self.masked), set())

    def test_validator_sees_no_citations_in_masked_text(self) -> None:
        self.assertEqual(extract_citation_markers(self.masked), set())


class RoundTripTests(unittest.TestCase):
    def test_masking_then_restoring_is_the_identity(self) -> None:
        text = "국민연금기금은 2023년 03월 07일 기준 655,490주(12.45%)를 보유합니다.[1][2]"
        protection = protect_literals(text)

        self.assertEqual(restore_literals(protection.masked, protection), text)

    def test_restores_into_reworded_prose(self) -> None:
        protection = protect_literals("2023년 03월 07일 기준 655,490주입니다.[1]")
        date, number, citation = protection.placeholders

        restored = restore_literals(
            f"기준일은 {date}이고 수량은 {number}주입니다.{citation}", protection
        )

        self.assertEqual(restored, "기준일은 2023년 03월 07일이고 수량은 655,490주입니다.[1]")

    def test_unknown_placeholder_raises_rather_than_guessing(self) -> None:
        protection = protect_literals("655,490주")

        with self.assertRaises(ValueError):
            restore_literals("__FESTIVAL_NUMBER_Z__", protection)


class IntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protection = protect_literals(
            "2023년 03월 07일 기준 655,490주입니다.[1]"
        )
        self.date, self.number, self.citation = self.protection.placeholders

    def _check(self, candidate: str):
        return check_placeholder_integrity(candidate, self.protection)

    def test_exact_survival_passes(self) -> None:
        self.assertTrue(self._check(self.protection.masked).valid)

    def test_reworded_but_intact_passes(self) -> None:
        candidate = f"{self.date}에는 {self.number}주였습니다.{self.citation}"

        self.assertTrue(self._check(candidate).valid)

    def test_missing_placeholder_is_named(self) -> None:
        result = self._check(f"{self.date} {self.number}")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, PLACEHOLDER_MISSING)
        self.assertIn(self.citation, result.detail)

    def test_unexpected_placeholder_is_named(self) -> None:
        result = self._check(
            f"{self.date} {self.number} {self.citation} __FESTIVAL_NUMBER_Z__"
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, PLACEHOLDER_UNEXPECTED)

    def test_duplicated_placeholder_is_named(self) -> None:
        result = self._check(
            f"{self.date} {self.number} {self.citation} {self.number}"
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, PLACEHOLDER_DUPLICATED)

    def test_reordered_placeholders_are_named(self) -> None:
        result = self._check(f"{self.number} {self.date} {self.citation}")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, PLACEHOLDER_REORDERED)

    def test_empty_protection_accepts_plain_text(self) -> None:
        protection = protect_literals("보유 현황입니다.")

        self.assertTrue(check_placeholder_integrity("현황을 정리했습니다.", protection).valid)


class PlaceholderSyntaxDetectionTests(unittest.TestCase):
    def test_detects_existing_placeholder_syntax(self) -> None:
        self.assertTrue(contains_placeholder_syntax("값은 __FESTIVAL_NUMBER_A__입니다."))

    def test_ordinary_text_is_not_flagged(self) -> None:
        self.assertFalse(contains_placeholder_syntax("보유주식수는 655,490주입니다.[1]"))

    def test_similar_looking_text_is_not_flagged(self) -> None:
        self.assertFalse(contains_placeholder_syntax("__FESTIVAL__ 및 __A_B__"))


if __name__ == "__main__":
    unittest.main()
