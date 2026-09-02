"""A reprint's own label, and the one table it heads.

A 정정신고 reprints a corrected region under a label the filer wrote as its own
source block -- ``(주3) 정정 전``, then the region's heading, then the table.
Neither the reprinted table nor the projection built from it carries that label,
so a projection read on its own looks unmarked while the filing has in fact said
which side of the correction it states.

The association is read from the parser's own ``content_order`` and is bounded
at both ends: a label heads the table that comes next -- optionally across the
region's own reprinted heading and nothing else -- and stops there.  These tests
pin those bounds, because every looser reading of "which label belongs to this
table" was measured against the corpus and found to contradict filings that
label their reprints in a table caption instead.

The other half is inheritance.  A projection takes the state of the table it was
*built from* and of no other table it happens to cite: a filing's projections
routinely also reference a shared metadata table, and inheriting through those
refs marks a report body as its own superseded text.
"""

import unittest

from app.reasoning.holding_correction_state import (
    ASSOCIATION_CONFLICT,
    ASSOCIATION_RULE,
    CORRECTED_STATE,
    PRIOR_STATE,
    document_correction_state_details,
    document_correction_states,
    document_table_correction_states,
)


HEADING = "3. 보유주식등의 수 및 보유비율"


def _section(*blocks_and_tables, section_id="s0001"):
    """A section built from an interleaved run of text blocks and table ids.

    ``str`` entries are source text blocks, ``("table", id)`` entries are
    tables, in the order the filing states them.
    """

    blocks: list[str] = []
    order: list[dict] = []
    for entry in blocks_and_tables:
        if isinstance(entry, tuple):
            order.append({"kind": "table", "table_id": entry[1]})
            continue
        order.append({"kind": "text", "block_index": len(blocks)})
        blocks.append(entry)
    return {
        "section_id": section_id,
        "text": "\n\n".join(blocks),
        "content_order": order,
    }


def _projection(chunk_id, table_id, *, section_path=("정 정 신 고 (보고)",),
                table_title=HEADING, refs=None, shares="1,000"):
    return {
        "chunk_id": chunk_id,
        "doc_id": "holding_fixture",
        "projection_type": "holding_report",
        "table_id": table_id,
        "source_table_id": table_id,
        "table_title": table_title,
        "section_title": section_path[-1],
        "section_path": list(section_path),
        "projection_fields": {"보유주식수": shares},
        "source_refs": [
            {"table_id": value, "row_start": 3, "row_end": 3}
            for value in (refs if refs is not None else [table_id])
        ],
    }


def _table_chunk(table_id, title=HEADING):
    return {"chunk_id": "chunk_" + table_id, "table_id": table_id,
            "table_title": title, "section_path": ["정 정 신 고 (보고)"]}


BODY_PATH = ("제1부 보고의 개요", "3. 보유주식등의 수 및 보유비율")


class LabelHeadedTableTests(unittest.TestCase):
    """C: zero gap.  D: one reprinted-heading gap.  E/F/G/H: the bounds."""

    def test_a_label_heads_the_table_that_immediately_follows(self) -> None:
        section = _section("(주3) 정정 전", ("table", "t0009"))
        states = document_table_correction_states([section])
        self.assertEqual(states["t0009"]["correction_state"], PRIOR_STATE)
        provenance = states["t0009"]["provenance"]
        self.assertEqual(provenance["rule"], ASSOCIATION_RULE)
        self.assertEqual(provenance["label_text"], "(주3) 정정 전")
        self.assertEqual(provenance["table_id"], "t0009")
        self.assertIsNone(provenance["intervening_heading"])

    def test_the_regions_own_reprinted_heading_may_stand_between(self) -> None:
        section = _section("(주4) 정정 후", HEADING, ("table", "t0010"))
        states = document_table_correction_states(
            [section], table_titles={"t0010": HEADING}
        )
        self.assertEqual(states["t0010"]["correction_state"], CORRECTED_STATE)
        self.assertEqual(states["t0010"]["provenance"]["intervening_heading"], HEADING)

    def test_unrelated_prose_between_label_and_table_breaks_it(self) -> None:
        section = _section(
            "(주3) 정정 전",
            "단순 오기재를 바로잡기 위하여 아래와 같이 정정합니다.",
            ("table", "t0009"),
        )
        self.assertEqual(document_table_correction_states([section]), {})

    def test_two_blocks_between_label_and_table_break_it(self) -> None:
        section = _section("(주3) 정정 전", HEADING, HEADING, ("table", "t0009"))
        self.assertEqual(
            document_table_correction_states(
                [section], table_titles={"t0009": HEADING}
            ),
            {},
        )

    def test_the_state_stops_at_the_table_it_heads(self) -> None:
        section = _section("(주3) 정정 전", ("table", "t0009"), ("table", "t0010"))
        states = document_table_correction_states([section])
        self.assertEqual(set(states), {"t0009"})

    def test_a_label_with_no_table_after_it_heads_nothing(self) -> None:
        section = _section("(주3) 정정 전", HEADING)
        self.assertEqual(document_table_correction_states([section]), {})

    def test_two_labels_disagreeing_about_one_table_fail_closed(self) -> None:
        first = _section("(주3) 정정 전", ("table", "t0009"), section_id="s0001")
        second = _section("(주4) 정정 후", ("table", "t0009"), section_id="s0002")
        states = document_table_correction_states([first, second])
        self.assertIsNone(states["t0009"]["correction_state"])
        self.assertEqual(states["t0009"]["provenance"]["rule"], ASSOCIATION_CONFLICT)

    def test_a_caption_is_not_a_standalone_label(self) -> None:
        """``주3) 정정 전 3. 보유…`` is decided on the projection, not here."""

        section = _section("주3) 정정 전 " + HEADING, ("table", "t0009"))
        self.assertEqual(document_table_correction_states([section]), {})

    def test_the_frozen_refusals_survive(self) -> None:
        for label in ("정정 전후", "정정 전과 정정 후", "정정 전과 동일"):
            with self.subTest(label=label):
                section = _section(label, ("table", "t0009"))
                self.assertEqual(document_table_correction_states([section]), {})


class ProjectionInheritanceTests(unittest.TestCase):
    """A/B: the C010 and C011 shape.  J: the C012 contamination regression."""

    def _document(self, refs_for_body=None):
        section = _section(
            "(주3) 정정 전", HEADING, ("table", "t0009"),
            "(주4) 정정 후", HEADING, ("table", "t0010"),
        )
        chunks = [
            _table_chunk("t0009"),
            _table_chunk("t0010"),
            _projection("prior_projection", "t0009", shares="15,565,026"),
            _projection("corrected_projection", "t0010", shares="15,601,252"),
            _projection(
                "body_projection", "t0043",
                section_path=BODY_PATH,
                refs=refs_for_body if refs_for_body is not None else ["t0043"],
                shares="15,601,252",
            ),
        ]
        return chunks, [section]

    def test_each_reprint_inherits_its_own_tables_state(self) -> None:
        chunks, sections = self._document()
        states = document_correction_states(chunks)
        self.assertEqual(states, {})          # nothing is provable without sections

        details = document_correction_state_details(chunks, sections)
        self.assertEqual(details["prior_projection"]["state"], PRIOR_STATE)
        self.assertEqual(details["corrected_projection"]["state"], CORRECTED_STATE)
        self.assertEqual(details["prior_projection"]["source"], ASSOCIATION_RULE)
        self.assertNotIn("body_projection", details)

    def test_a_body_citing_a_labelled_metadata_table_stays_unlabelled(self) -> None:
        """The C012 shape: the body also references the prior region's table."""

        chunks, sections = self._document(refs_for_body=["t0009", "t0043"])
        details = document_correction_state_details(chunks, sections)
        self.assertNotIn("body_projection", details)

    def test_without_sections_the_frozen_readers_decide_alone(self) -> None:
        chunks, _sections = self._document()
        self.assertEqual(document_correction_state_details(chunks), {})


class ExistingAuthorityTests(unittest.TestCase):
    """K: a table label never overrides a state the projection itself proves."""

    def _document(self, caption):
        section = _section("(주4) 정정 후", HEADING, ("table", "t0009"))
        chunks = [
            _table_chunk("t0009", title=caption),
            _projection("projection", "t0009", table_title=caption),
        ]
        return chunks, [section]

    def test_a_declared_state_stands(self) -> None:
        chunks, sections = self._document("주3) 정정 전")
        details = document_correction_state_details(chunks, sections)
        entry = details["projection"]
        self.assertEqual(entry["state"], PRIOR_STATE)
        self.assertEqual(entry["source"], "declared")

    def test_a_disagreement_is_recorded_rather_than_resolved(self) -> None:
        chunks, sections = self._document("주3) 정정 전")
        entry = document_correction_state_details(chunks, sections)["projection"]
        self.assertEqual(entry["conflict"]["table_authority"], CORRECTED_STATE)
        self.assertEqual(entry["conflict"]["provenance"]["rule"], ASSOCIATION_RULE)

    def test_agreement_records_no_conflict(self) -> None:
        chunks, sections = self._document("주4) 정정 후")
        entry = document_correction_state_details(chunks, sections)["projection"]
        self.assertEqual(entry["state"], CORRECTED_STATE)
        self.assertNotIn("conflict", entry)


if __name__ == "__main__":
    unittest.main()
