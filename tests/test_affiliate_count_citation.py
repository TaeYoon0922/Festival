"""IEV2-C086: the affiliate count is cited where its row sits.

The question -- "계열회사는 상장과 비상장을 합쳐 몇 개인가" -- asks one 기업집단
for one count.  Every periodic report of the group restates that row for its
own period, so retrieval serves the 2023.03, 2023.06 and 2023.09 rows beside
the 사업보고서 row the answer is read from.  Composing three of them together
cited filings the stated total was never read from and left the annual report
-- served, and the row the counts come from -- uncited.

The fixtures below are the four 계열회사 현황 rows as the corpus persists them:
the same four columns, one countable row each, and 2023.03 counting one more
unlisted affiliate than the rest.  ``source_refs`` is absent from every one of
them because the chunk metadata does not carry it, so the row a citation
belongs to is the one ``affiliate_group`` reads out of the chunk itself.
"""

from __future__ import annotations

import unittest

from app.generation.answer_generator import generate_answer
from app.reasoning.affiliate_group import affiliate_counts
from app.reasoning.answer_composer import AnswerComposer
from app.reasoning.evidence_builder import EvidenceItem, EvidenceSet
from app.reasoning.periodic_evidence_selector import PeriodicEvidenceSelector
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from app.reasoning.periodic_metric_view import source_chunk_view
from tests.test_periodic_fact_resolver import _group


QUESTION = "세아베스틸지주가 속한 기업집단의 계열회사는 상장과 비상장을 합쳐 몇 개인가?"

#: The gold document, chunk and table row for IEV2-C086.
GOLD_DOC = "periodic_20240312000629"
GOLD_CHUNK = "periodic_20240312000629:ch_878421d46c72c812591b"
GOLD_SOURCE_REF = {"table_id": "t0645", "row_start": 2, "row_end": 2}

SECTION_PATH = ("IX. 계열회사 등에 관한 사항",)
HEADERS = [
    "기업집단의 명칭",
    "계열회사의 수 / 상장",
    "계열회사의 수 / 비상장",
    "계열회사의 수 / 계",
]


def _table_text(listed: int, unlisted: int, total: int) -> str:
    return "\n".join(
        [
            "| " + " | ".join(HEADERS) + " |",
            "| --- | --- | --- | --- |",
            f"| 세아 | {listed} | {unlisted} | {total} |",
        ]
    )


def _cell(text: str) -> dict[str, object]:
    return {
        "colspan": 1,
        "is_header": False,
        "rowspan": 1,
        "source_tag": "te",
        "text": text,
    }


def _affiliate_item(
    *,
    chunk_id: str,
    doc_id: str,
    rank: int,
    report_nm: str,
    rcept_dt: str,
    table_id: str,
    base_month: int,
    listed: int = 5,
    unlisted: int = 22,
    readable: bool = True,
) -> EvidenceItem:
    """One 계열회사 현황 row as the chunker persists and the backend serves it.

    ``readable`` false keeps the section and the wording but drops the persisted
    table, which is how a prose chunk from the same section arrives.
    """

    total = listed + unlisted
    if readable:
        content = _table_text(listed, unlisted, total)
    else:
        content = (
            "당사가 속한 기업집단의 계열회사 현황은 아래 표를 참조하시기 바랍니다."
        )
    period = {
        "base_year": 2023,
        "base_month": base_month,
        "basis_period": f"기준일 현재 계열회사 현황 ({rcept_dt})",
    }
    source_chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "periodic",
        "chunk_type": "table" if readable else "text",
        "section_path": list(SECTION_PATH),
        "content": content,
        "retrieval_text": content,
        "report_nm": report_nm,
        "table_id": table_id,
        "corp_code": "00106669",
        "corp_name": "세아베스틸지주",
    }
    if readable:
        source_chunk.update(
            {
                "row_start": 2,
                "row_end": 2,
                "column_headers": list(HEADERS),
                "table_rows": [
                    [
                        _cell("세아"),
                        _cell(str(listed)),
                        _cell(str(unlisted)),
                        _cell(str(total)),
                    ]
                ],
            }
        )
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        company_id="00106669",
        corp_code="00106669",
        corp_name="세아베스틸지주",
        doc_group="periodic",
        chunk_type="table" if readable else "text",
        section_path=SECTION_PATH,
        evidence_text=content,
        retrieval_rank=rank,
        retrieval_score=1.0 - rank / 100,
        rcept_dt=rcept_dt,
        report_nm=report_nm,
        period=period,
        # The chunk metadata carries no source_refs, so neither does the item.
        source_refs=(),
        provenance={
            "source_chunk_id": chunk_id,
            "source_doc_id": doc_id,
            "source_chunk": source_chunk,
        },
        holding={},
        temporal_match=None,
    )


def _q1(rank: int = 1, **kwargs) -> EvidenceItem:
    return _affiliate_item(
        chunk_id="periodic_20230515002568:ch_f21aaa5df26be5a7b104",
        doc_id="periodic_20230515002568",
        rank=rank,
        report_nm="분기보고서 (2023.03)",
        rcept_dt="20230515",
        table_id="t0427",
        base_month=3,
        unlisted=23,
        **kwargs,
    )


def _h1(rank: int = 2, **kwargs) -> EvidenceItem:
    return _affiliate_item(
        chunk_id="periodic_20230814002199:ch_d826d1997084410f6fc7",
        doc_id="periodic_20230814002199",
        rank=rank,
        report_nm="반기보고서 (2023.06)",
        rcept_dt="20230814",
        table_id="t0441",
        base_month=6,
        **kwargs,
    )


def _q3(rank: int = 3, **kwargs) -> EvidenceItem:
    return _affiliate_item(
        chunk_id="periodic_20231114000256:ch_ee3286af6455252df783",
        doc_id="periodic_20231114000256",
        rank=rank,
        report_nm="분기보고서 (2023.09)",
        rcept_dt="20231114",
        table_id="t0439",
        base_month=9,
        **kwargs,
    )


def _annual_2023(rank: int = 4, **kwargs) -> EvidenceItem:
    return _affiliate_item(
        chunk_id=GOLD_CHUNK,
        doc_id=GOLD_DOC,
        rank=rank,
        report_nm="사업보고서 (2023.12)",
        rcept_dt="20240312",
        table_id="t0645",
        base_month=12,
        **kwargs,
    )


def _annual_2024(rank: int, **kwargs) -> EvidenceItem:
    return _affiliate_item(
        chunk_id="periodic_20250318000465:ch_6869416e6a1017f5515d",
        doc_id="periodic_20250318000465",
        rank=rank,
        report_nm="사업보고서 (2024.12)",
        rcept_dt="20250318",
        table_id="t1880",
        base_month=12,
        unlisted=19,
        **kwargs,
    )


def _evidence_set(
    items: tuple[EvidenceItem, ...], *, periodic_intent: str
) -> EvidenceSet:
    groups = [_group(f"g{index}", item) for index, item in enumerate(items, start=1)]
    return EvidenceSet(
        question=QUESTION,
        query_plan={
            "raw_query": QUESTION,
            "task_type": "periodic_fact",
            "company": "세아베스틸지주",
            "metric": None,
            "period": {
                "year": None,
                "quarter": None,
                "from": None,
                "to": None,
                "period_type": None,
            },
            "evidence": {
                "periodic_intent": periodic_intent,
                "periodic_intent_evidence": "계열회사",
            },
        },
        task_type="periodic_fact",
        evidence_groups=tuple(groups),
        retrieval_order=tuple(item.chunk_id for item in items),
        raw_candidate_count=len(items),
        selected_evidence_count=len(items),
        warnings=(),
        ambiguity={
            "temporal_ambiguity": False,
            "temporal_constraint": {
                "explicit": False,
                "year": None,
                "quarter": None,
                "from_date": None,
                "to_date": None,
                "period_type": None,
            },
        },
    )


def _served(*items: EvidenceItem) -> tuple[EvidenceItem, ...]:
    return items or (_q1(), _h1(), _q3(), _annual_2023())


def _select(*items: EvidenceItem, periodic_intent: str = "affiliate_count"):
    evidence = _evidence_set(_served(*items), periodic_intent=periodic_intent)
    plan = evidence.query_plan
    resolution = resolve_periodic_facts(evidence, query_plan=plan)
    selection = PeriodicEvidenceSelector().select(resolution, query_plan=plan)
    return evidence, selection


class AffiliateCountCitationTests(unittest.TestCase):
    def test_the_row_that_states_the_counts_is_the_only_selected_source(self):
        _evidence, selection = _select()

        self.assertEqual(selection.selected_chunk_ids, (GOLD_CHUNK,))
        self.assertIn("affiliate_count_row_preferred", selection.warnings)

    def test_the_answer_source_and_the_citation_are_the_same_row(self):
        evidence, selection = _select()

        draft = AnswerComposer().compose(
            evidence, periodic_resolution=selection.resolution
        )

        self.assertEqual(len(draft.citations), 1)
        citation = draft.citations[0]
        source = draft.answer_sections[0].content["fact"]["sources"][0]
        self.assertEqual(citation.chunk_id, source["chunk_id"])
        self.assertEqual(citation.doc_id, source["doc_id"])
        self.assertEqual((citation.doc_id, citation.chunk_id), (GOLD_DOC, GOLD_CHUNK))

    def test_listed_unlisted_and_total_are_read_from_the_cited_row(self):
        _evidence, selection = _select()
        source = selection.resolution.facts[0].sources[0]

        counts = affiliate_counts(source_chunk_view(source))

        self.assertEqual((counts.listed, counts.unlisted, counts.total), (5, 22, 27))
        self.assertEqual(counts.stated_total, 27)
        self.assertEqual(dict(counts.source_ref), GOLD_SOURCE_REF)

    def test_the_persisted_row_provenance_survives_resolver_and_composer(self):
        evidence, selection = _select()
        served = {
            item.chunk_id: item
            for group in evidence.evidence_groups
            for item in group.items
        }
        source = selection.resolution.facts[0].sources[0]
        item = served[source.chunk_id]
        chunk = source.provenance["source_chunk"]

        self.assertEqual(source.provenance["source_chunk_id"], item.chunk_id)
        self.assertEqual(source.provenance["source_doc_id"], item.doc_id)
        self.assertEqual(chunk["column_headers"], HEADERS)
        self.assertEqual(
            (chunk["table_id"], chunk["row_start"], chunk["row_end"]),
            (GOLD_SOURCE_REF["table_id"], 2, 2),
        )

    def test_no_other_period_is_substituted_as_the_citation(self):
        evidence, selection = _select()
        other_docs = {
            item.doc_id
            for group in evidence.evidence_groups
            for item in group.items
            if item.doc_id != GOLD_DOC
        }

        generated = generate_answer(
            AnswerComposer().compose(evidence, periodic_resolution=selection.resolution)
        )

        self.assertEqual(
            {citation.doc_id for citation in generated.citations}, {GOLD_DOC}
        )
        self.assertEqual(
            {citation.citation_id for citation in generated.citations}, {"[1]"}
        )
        for doc_id in other_docs:
            self.assertNotIn(doc_id, generated.answer_text)
        # 2023.03 counts one more unlisted affiliate; the cited row never says 28.
        self.assertNotIn("| 세아 | 5 | 23 | 28 |", generated.answer_text)
        self.assertIn("| 세아 | 5 | 22 | 27 |", generated.answer_text)

    def test_the_selected_row_is_answerable_without_a_period_notice(self):
        evidence, selection = _select()

        draft = AnswerComposer().compose(
            evidence, periodic_resolution=selection.resolution
        )

        self.assertTrue(draft.answerable)
        self.assertFalse(draft.ambiguity["temporal_ambiguity"])


class AffiliateRowSelectionTests(unittest.TestCase):
    def test_the_served_ranking_decides_between_two_annual_rows(self):
        """No year is named, so the ranking decides -- not the row's own date."""

        _evidence, gold_first = _select(
            _q1(), _h1(), _q3(), _annual_2023(rank=4), _annual_2024(rank=5)
        )
        _evidence, other_first = _select(
            _q1(), _h1(), _q3(), _annual_2024(rank=4), _annual_2023(rank=5)
        )

        self.assertEqual(gold_first.selected_chunk_ids, (GOLD_CHUNK,))
        self.assertEqual(
            other_first.selected_chunk_ids,
            ("periodic_20250318000465:ch_6869416e6a1017f5515d",),
        )

    def test_without_an_annual_report_the_readable_rows_stand_on_their_own(self):
        _evidence, selection = _select(_q1(), _h1(), _q3())

        self.assertEqual(
            selection.selected_chunk_ids,
            ("periodic_20230515002568:ch_f21aaa5df26be5a7b104",),
        )
        self.assertIn("affiliate_count_row_preferred", selection.warnings)

    def test_a_section_that_states_no_countable_row_narrows_nothing(self):
        _evidence, selection = _select(
            _q1(readable=False),
            _h1(readable=False),
            _q3(readable=False),
            _annual_2023(readable=False),
        )

        self.assertEqual(len(selection.selected_chunk_ids), 3)
        self.assertNotIn("affiliate_count_row_preferred", selection.warnings)


class OtherPeriodicIntentsAreUnchangedTests(unittest.TestCase):
    def test_an_ordinary_periodic_question_still_composes_up_to_three_sources(self):
        _evidence, selection = _select(periodic_intent="business_product")

        self.assertEqual(len(selection.selected_chunk_ids), 3)
        self.assertNotIn("affiliate_count_row_preferred", selection.warnings)

    def test_a_listing_history_question_is_untouched(self):
        _evidence, selection = _select(periodic_intent="listing_history")

        self.assertEqual(len(selection.selected_chunk_ids), 3)
        self.assertNotIn("affiliate_count_row_preferred", selection.warnings)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
