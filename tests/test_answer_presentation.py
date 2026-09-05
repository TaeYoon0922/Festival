"""Presentation must be provably lossless.

The rewrite runs on a finished, already-guarded answer, so the only way it can
do harm is by losing something: a citation marker, a table row, a figure. Each
test below pins one of those, and the last two pin the property that matters
most -- an answer this does not recognise is returned exactly as it arrived.
"""

from __future__ import annotations

import unittest

from app.generation.answer_presentation import citation_markers, readable_answer


GENERAL = """General evidence
1. [기업명] LG에너지솔루션
[공시명] 반기보고서 (2025.06)
[Section Path] II. 사업의 내용 > 3. 원재료 및 생산설비
[Table] - 당사는 2025년 당기 중 신·증설 투자 등에 총 5.8조원을 사용하였습니다.
[단위] 백만원
[기간표현] 2025년, 당기, 2025.01, 2025.06, 누적

| 사업부문 | 투자목적 | 당기 투자액 |
| --- | --- | --- |
| 2차전지 | 신ㆍ증설 보완 | 57,694 |
| 합 계 |  | 57,694 | [1]
2. [기업명] 삼성SDI
[공시명] 반기보고서 (2025.06)
[Section Path] II. 사업의 내용 > 3. 원재료 및 생산설비

(4) 설비 등 투자현황

당사는 2025년 상반기 중 1조 8,430억원을 시설 투자에 사용하였습니다. [2]

인용
[1]
doc_id: periodic_20250814001590
chunk_id: periodic_20250814001590:ch_4330cbaf92fcb9128605
[2]
doc_id: periodic_20250814003107
chunk_id: periodic_20250814003107:ch_0871f8ec3cc1c1dcda2b"""


class LosslessTests(unittest.TestCase):
    def test_every_citation_marker_survives_in_order(self) -> None:
        self.assertEqual(
            citation_markers(readable_answer(GENERAL)), citation_markers(GENERAL)
        )

    def test_every_figure_survives(self) -> None:
        for figure in ("57,694", "5.8조원", "1조 8,430억원", "2025.06"):
            with self.subTest(figure=figure):
                self.assertIn(figure, readable_answer(GENERAL))

    def test_table_rows_survive_unindented(self) -> None:
        # Indenting a markdown table by four spaces turns it into a code block,
        # so body lines must not be touched at all.
        output = readable_answer(GENERAL)
        for row in ("| 2차전지 | 신ㆍ증설 보완 | 57,694 |", "| --- | --- | --- |"):
            with self.subTest(row=row):
                self.assertIn(f"\n{row}", output)

    def test_the_citation_block_is_untouched(self) -> None:
        output = readable_answer(GENERAL)
        self.assertIn("doc_id: periodic_20250814001590", output)
        self.assertIn(
            "chunk_id: periodic_20250814003107:ch_0871f8ec3cc1c1dcda2b", output
        )

    def test_no_body_line_is_lost(self) -> None:
        tags = ("[기업명]", "[공시명]", "[Section Path]", "[Table]", "[단위]", "[기간표현]")
        body = [
            line
            for line in GENERAL.split("\n")
            if not line.strip().startswith(tags) and not line.strip().startswith("1. [")
            and not line.strip().startswith("2. [")
        ]
        output = readable_answer(GENERAL)
        for line in body:
            with self.subTest(line=line):
                self.assertIn(line, output)


class RewriteTests(unittest.TestCase):
    def test_the_filing_becomes_one_heading(self) -> None:
        output = readable_answer(GENERAL)
        self.assertIn("1. LG에너지솔루션 · 반기보고서 (2025.06)", output)
        self.assertIn("2. 삼성SDI · 반기보고서 (2025.06)", output)

    def test_the_section_path_is_kept_without_its_tag(self) -> None:
        output = readable_answer(GENERAL)
        self.assertIn("   II. 사업의 내용 > 3. 원재료 및 생산설비", output)
        self.assertNotIn("[Section Path]", output)

    def test_table_context_is_relabelled_not_dropped(self) -> None:
        output = readable_answer(GENERAL)
        self.assertIn("   단위 백만원", output)
        self.assertIn("   기간 2025년, 당기, 2025.01, 2025.06, 누적", output)
        self.assertNotIn("[단위]", output)
        self.assertNotIn("[기간표현]", output)

    def test_it_is_idempotent(self) -> None:
        once = readable_answer(GENERAL)
        self.assertEqual(readable_answer(once), once)


class UnrecognisedInputTests(unittest.TestCase):
    """What this does not recognise, it must not touch."""

    def test_a_refusal_is_returned_unchanged(self) -> None:
        for text in (
            "현재 확보된 공시 근거만으로는 해당 내용을 확인하기 어렵습니다.",
            "어느 회사에 대한 공시를 확인할까요?",
            "제공된 공시 기준으로 확인한 이후 해지 계약은 없습니다.",
            "",
        ):
            with self.subTest(text=text):
                self.assertEqual(readable_answer(text), text)

    def test_a_holding_timeline_is_returned_unchanged(self) -> None:
        holding = (
            "보유 변동 내역\n파마리서치에 대해 확인된 보유 변동은 다음과 같습니다.\n"
            "2022-12-05 | 국민연금공단 | 613,758주 → 720,039주 | 6.07% → 7.12% [1]"
        )
        self.assertEqual(readable_answer(holding), holding)

    def test_a_correction_prefix_is_not_mistaken_for_a_tag(self) -> None:
        # "[기재정정]사업보고서" opens with a bracket but is a report name, not
        # one of the retrieval tags.
        periodic = (
            "Periodic fact 1\n근거 1 보고 기간: 2025년\n"
            "근거 1 보고서: [기재정정]사업보고서 (2023.12)\n내용: | 변경일 | [5]"
        )
        self.assertEqual(readable_answer(periodic), periodic)

    def test_a_lone_tag_without_an_item_number_is_left_alone(self) -> None:
        text = "[공시명] 반기보고서 (2025.06)\n본문"
        self.assertEqual(readable_answer(text), text)


if __name__ == "__main__":
    unittest.main()
