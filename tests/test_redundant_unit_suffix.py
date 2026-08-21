from __future__ import annotations

import unittest

from app.generation.compact_claim import ClaimCitation, ClaimField, CompactClaim, _render
from app.generation.lossless_verbalization import (
    REDUNDANT_UNIT_SUFFIX,
    UNIT_SUFFIXES,
    citation_adjacent_units,
    detach_claim_citations,
    redundant_unit_suffixes,
    verify_lossless_candidate,
)
from app.generation.protected_literals import ProtectedLiteral, ProtectedText


def _claim(value: str) -> CompactClaim:
    field = ClaimField(
        name="after_value",
        label="변동 후",
        value=value,
        marker="[1]",
        chunk_id="c1",
    )
    citation = ClaimCitation(
        marker="[1]",
        chunk_id="c1",
        doc_id="d1",
        source_refs=({"table_id": "t1", "row_start": 1, "row_end": 1},),
    )
    return CompactClaim(
        question="테스트 질문",
        company="파마리서치",
        reporter="국민연금공단",
        fields=(field,),
        citations=(citation,),
        deterministic_text=_render("파마리서치", "국민연금공단", [field]),
    )


def _verify(value: str, tail: str):
    """Verify a reply that renders the claim with ``tail`` after the placeholder."""

    claim = _claim(value)
    detached = detach_claim_citations(claim)
    placeholder = detached.protection.literals[0].placeholder
    candidate = detached.protection.masked.replace(
        placeholder, placeholder + tail, 1
    )
    return verify_lossless_candidate(candidate, claim=claim, detached=detached)


class ReportedBugTests(unittest.TestCase):
    """The exact production defect: 7.12% served as "7.12% [1]%입니다."."""

    def test_a_percent_appended_to_a_percent_value_is_rejected(self) -> None:
        result = _verify("7.12%", "%입니다.")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, REDUNDANT_UNIT_SUFFIX)
        self.assertIsNone(result.final_answer)

    def test_a_spaced_percent_is_rejected(self) -> None:
        result = _verify("7.12%", " %입니다.")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, REDUNDANT_UNIT_SUFFIX)

    def test_a_negative_percentage_is_covered_too(self) -> None:
        result = _verify("-3.07%", "%입니다.")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, REDUNDANT_UNIT_SUFFIX)

    def test_the_malformed_answer_can_no_longer_be_produced(self) -> None:
        result = _verify("7.12%", "%입니다.")

        self.assertIsNone(result.final_answer)
        # The shapes the bug produced, now unreachable.
        for shape in ("%%", "% [1]%"):
            with self.subTest(shape=shape):
                self.assertNotIn(shape, result.final_answer or "")


class DuplicatedTemplateUnitTests(unittest.TestCase):
    """The same defect in its other form: the unit sits in the claim template."""

    def test_a_doubled_share_unit_is_rejected(self) -> None:
        result = _verify("655,490주", "주주가 되었습니다.")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, REDUNDANT_UNIT_SUFFIX)

    def test_a_spaced_doubled_share_unit_is_rejected(self) -> None:
        result = _verify("655,490주", " 주주입니다.")

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, REDUNDANT_UNIT_SUFFIX)


class ControlCaseTests(unittest.TestCase):
    """The wording live runs produced must keep passing."""

    def test_the_template_share_unit_is_not_a_duplicate(self) -> None:
        result = _verify("655,490주", "주가 되었습니다.")

        self.assertTrue(result.valid)
        self.assertIn("655,490주", result.final_answer)
        self.assertIn("[1]", result.final_answer)

    def test_a_percent_value_without_an_appended_unit_passes(self) -> None:
        result = _verify("7.12%", "입니다.")

        self.assertTrue(result.valid)
        self.assertIn("7.12%", result.final_answer)
        self.assertNotIn("%%", result.final_answer)

    def test_a_unitless_number_passes(self) -> None:
        result = _verify("2,000", "입니다.")

        self.assertTrue(result.valid)
        self.assertIn("2,000", result.final_answer)

    def test_a_faithful_echo_passes(self) -> None:
        result = _verify("655,490주", "")

        self.assertTrue(result.valid)


class DetectionHelperTests(unittest.TestCase):
    def _protection(self, literal_text: str, template_tail: str) -> ProtectedText:
        placeholder = "__FESTIVAL_NUMBER_A__"
        return ProtectedText(
            original=literal_text + template_tail,
            masked=placeholder + template_tail,
            literals=(
                ProtectedLiteral(
                    placeholder=placeholder, text=literal_text, kind="number"
                ),
            ),
        )

    def test_a_unit_inside_the_literal_may_not_be_repeated(self) -> None:
        protection = self._protection("7.12%", "")

        self.assertTrue(
            redundant_unit_suffixes("__FESTIVAL_NUMBER_A__%", protection)
        )

    def test_a_literal_that_already_ends_in_the_template_unit_is_flagged(self) -> None:
        """The spec's case: the protected value itself carries the unit."""

        protection = self._protection("655,490주", "")

        self.assertTrue(
            redundant_unit_suffixes("__FESTIVAL_NUMBER_A__주", protection)
        )

    def test_the_template_unit_alone_is_not_flagged(self) -> None:
        protection = self._protection("655,490", "주")

        self.assertEqual(
            redundant_unit_suffixes("__FESTIVAL_NUMBER_A__주", protection), []
        )

    def test_a_missing_placeholder_is_left_to_the_integrity_check(self) -> None:
        protection = self._protection("7.12%", "")

        self.assertEqual(redundant_unit_suffixes("값이 없습니다", protection), [])

    def test_every_declared_unit_is_detected(self) -> None:
        for unit in UNIT_SUFFIXES:
            with self.subTest(unit=unit):
                protection = self._protection(f"12{unit}", "")

                self.assertTrue(
                    redundant_unit_suffixes(
                        f"__FESTIVAL_NUMBER_A__{unit}", protection
                    )
                )


class CitationAdjacentUnitTests(unittest.TestCase):
    """The second net, on the answer that would have been served."""

    def test_the_reported_malformed_answer_is_detected(self) -> None:
        self.assertEqual(
            citation_adjacent_units("보유 비율이 변동 후 7.12% [1]%입니다."), ["[1]%"]
        )

    def test_a_correct_answer_is_not_flagged(self) -> None:
        for answer in (
            "에스엠 변동 후 주식 수는 2,967,759주 [1]입니다.",
            "효성중공업 주식은 변동 후 655,490주 [1]가 되었습니다.",
            "보유 비율은 7.12% [1]입니다.",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(citation_adjacent_units(answer), [])

    def test_detection_only_never_rewrites(self) -> None:
        answer = "값은 7.12% [1]%입니다."

        citation_adjacent_units(answer)

        self.assertEqual(answer, "값은 7.12% [1]%입니다.")


class PromptRuleTests(unittest.TestCase):
    def test_the_prompt_forbids_appending_a_unit(self) -> None:
        from app.generation.lossless_verbalization import (
            LOSSLESS_VERBALIZER_SYSTEM_PROMPT as prompt,
        )

        self.assertIn("보호된 값 전체를 나타낸다", prompt)
        self.assertIn("어떤 단위나 기호도 덧붙이지 않는다", prompt)

    def test_the_prompt_names_no_real_case(self) -> None:
        from app.generation.lossless_verbalization import (
            LOSSLESS_VERBALIZER_SYSTEM_PROMPT as prompt,
        )

        for leaked in ("7.12", "655,490", "2,967,759", "파마리서치", "HX0"):
            with self.subTest(leaked=leaked):
                self.assertNotIn(leaked, prompt)


if __name__ == "__main__":
    unittest.main()
