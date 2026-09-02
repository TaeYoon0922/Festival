import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

from app.reasoning.correction_graph import (
    AMBIGUOUS,
    CorrectionGraphUnavailable,
    CORRECTION_RELATION_TYPE,
    RESOLVED,
    SOURCE_CORRECTION_NOTICE,
    SOURCE_CYCLE_DETECTED,
    SOURCE_EVENT_TITLE_KEY,
    SOURCE_MULTIPLE_CANDIDATES,
    SOURCE_NOTICE_TARGET_MISSING,
    SOURCE_PERIODIC_PERIOD_KEY,
    SOURCE_SELF_REFERENCE,
    UNRESOLVED,
    CorrectionNotice,
    CorrectionRelation,
    DisclosureRecord,
    assemble_correction_groups,
    NOTICE_SOURCE_TABLE,
    NOTICE_SOURCE_TEXT,
    build_correction_graph,
    extract_correction_notice,
    extract_correction_notice_from_text,
    normalize_report_title,
    relation_id,
)
from app.reasoning.correction_policy import (
    POLICY_CORRECTED_ONLY,
    document_states,
    prefers_document,
    POLICY_LATEST_PREFERRED,
    POLICY_ORIGINAL_ONLY,
    correction_summary,
    correction_timeline,
)
from app.reasoning.query_plan import QueryExecutor, QueryPeriod, QueryPlan
from app.reasoning.router import QueryRouter
from app.retrieval.correction_repository import PostgresCorrectionRepository
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _record(
    doc_id: str,
    *,
    corp_code: str = "00126380",
    doc_group: str = "exchange",
    report_nm: str = "단일판매ㆍ공급계약체결",
    rcept_dt: str = "2023-01-10",
    rcept_no: str | None = None,
    doc_subtype: str | None = "단일판매공급계약체결",
    base_year: int | None = None,
    base_month: int | None = None,
    is_correction: bool = False,
) -> DisclosureRecord:
    return DisclosureRecord(
        doc_id=doc_id,
        corp_code=corp_code,
        doc_group=doc_group,
        report_nm=report_nm,
        rcept_no=rcept_no or doc_id.rsplit("_", 1)[-1],
        rcept_dt=rcept_dt,
        doc_subtype=doc_subtype,
        base_year=base_year,
        base_month=base_month,
        is_correction=is_correction,
    )


def _notice(doc_id: str, submitted_on: str, name: str = "단일판매ㆍ공급계약 체결") -> CorrectionNotice:
    return CorrectionNotice(
        doc_id=doc_id,
        target_submitted_on=submitted_on,
        target_report_nm=name,
        corrected_on=None,
        source_table_id="t0002",
        source_label="정정관련공시서류제출일",
    )


def _notice_tables(correction_date: str, target_name: str, submitted_on: str) -> list[dict]:
    """The frozen table shape a correcting disclosure actually carries."""

    def row(*cells: str) -> list[dict]:
        return [{"text": cell, "is_header": False} for cell in cells]

    return [
        {"table_id": "t0001", "rows": [row("정정일자", correction_date)]},
        {
            "table_id": "t0002",
            "rows": [
                row("1. 정정관련 공시서류", target_name),
                row("2. 정정관련 공시서류제출일", submitted_on),
                row("3. 정정사유", "계약금액 변경"),
            ],
        },
    ]


def _notice_sections(correction_date: str, target_name: str, submitted_on: str) -> list[dict]:
    """The same notice as ``_notice_tables``, written as prose instead.

    Some filers' HTML emits the header lines as paragraphs, so the frozen
    section text carries exactly what the one-row tables carry elsewhere.
    """

    return [
        {
            "section_id": "s0001",
            "text": "\n\n".join(
                (
                    "정정일자 : " + correction_date,
                    "1. 정정관련 공시서류 : " + target_name,
                    "2. 정정관련 공시서류제출일 : " + submitted_on,
                    "3. 정정사유 : 계약금액 변경",
                )
            ),
        }
    ]


def _chain_fixture() -> tuple[list[DisclosureRecord], dict[str, CorrectionNotice]]:
    """Original -> correction -> re-correction of the previous correction."""

    records = [
        _record("exchange_1", rcept_dt="2023-01-10", rcept_no="20230110800001"),
        _record(
            "exchange_2",
            rcept_dt="2023-02-01",
            rcept_no="20230201800001",
            report_nm="[기재정정]단일판매ㆍ공급계약체결",
            is_correction=True,
        ),
        _record(
            "exchange_3",
            rcept_dt="2023-03-01",
            rcept_no="20230301800001",
            report_nm="[기재정정]단일판매ㆍ공급계약체결",
            is_correction=True,
        ),
    ]
    notices = {
        "exchange_2": _notice("exchange_2", "2023-01-10"),
        "exchange_3": _notice("exchange_3", "2023-02-01"),
    }
    return records, notices


class CorrectionNoticeExtractionTests(unittest.TestCase):
    def test_notice_is_read_from_the_frozen_table_payload(self) -> None:
        notice = extract_correction_notice(
            "exchange_2",
            _notice_tables("2023-02-01", "단일판매ㆍ공급계약 체결", "2020-12-22"),
        )
        self.assertIsNotNone(notice)
        self.assertEqual(notice.target_submitted_on, "2020-12-22")
        self.assertEqual(notice.target_report_nm, "단일판매ㆍ공급계약 체결")
        self.assertEqual(notice.corrected_on, "2023-02-01")

    def test_dart_numbered_labels_and_date_formats_are_read(self) -> None:
        tables = [
            {
                "table_id": "t0001",
                "table_rows": [
                    [{"text": "1. 정정대상 공시서류 :"}, {"text": "주요사항보고서(유상증자결정)"}],
                    [{"text": "2. 정정대상 공시서류의 최초제출일 :"}, {"text": "2025년 2월 4일"}],
                ],
            }
        ]
        notice = extract_correction_notice("major_1", tables)
        self.assertEqual(notice.target_submitted_on, "2025-02-04")
        self.assertEqual(notice.target_report_nm, "주요사항보고서(유상증자결정)")

    def test_prose_notice_reads_the_same_fields_as_the_table_form(self) -> None:
        """A filing that wrote the notice as prose states the same thing."""

        table = extract_correction_notice(
            "exchange_2",
            _notice_tables("2023-02-01", "단일판매ㆍ공급계약 체결", "2020-12-22"),
        )
        prose = extract_correction_notice_from_text(
            "exchange_2",
            _notice_sections("2023-02-01", "단일판매ㆍ공급계약 체결", "2020-12-22"),
        )
        self.assertIsNotNone(prose)
        self.assertEqual(prose.target_submitted_on, table.target_submitted_on)
        self.assertEqual(prose.target_report_nm, table.target_report_nm)
        self.assertEqual(prose.corrected_on, table.corrected_on)
        self.assertEqual(table.source_kind, NOTICE_SOURCE_TABLE)
        self.assertEqual(prose.source_kind, NOTICE_SOURCE_TEXT)

    def test_prose_notice_reads_the_korean_spelled_date(self) -> None:
        """``2024년 9월 2일`` is the frozen date reader's, not a new rule."""

        prose = extract_correction_notice_from_text(
            "holding_x",
            [
                {
                    "section_id": "s0001",
                    "text": (
                        "1. 정정대상 공시서류 : 주식등의 대량보유상황보고서\n\n"
                        "2. 정정대상 공시서류의 최초제출일 : 2024년 9월 2일"
                    ),
                }
            ],
        )
        self.assertEqual(prose.target_submitted_on, "2024-09-02")
        self.assertEqual(prose.target_report_nm, "주식등의 대량보유상황보고서")

    def test_prose_without_a_labelled_line_is_not_a_notice(self) -> None:
        """Free text that merely mentions a correction proves nothing."""

        self.assertIsNone(
            extract_correction_notice_from_text(
                "exchange_9",
                [
                    {
                        "section_id": "s0001",
                        "text": (
                            "이 보고서는 기재사항을 정정하기 위하여 제출되었으며 "
                            "정정대상 공시서류를 참고하시기 바랍니다."
                        ),
                    }
                ],
            )
        )

    def test_prose_reader_ignores_a_line_whose_date_is_unreadable(self) -> None:
        self.assertIsNone(
            extract_correction_notice_from_text(
                "exchange_9",
                [{"section_id": "s0001", "text": "2. 정정관련 공시서류제출일 : 해당사항 없음"}],
            )
        )

    def test_document_without_a_notice_returns_none(self) -> None:
        tables = [{"table_id": "t0001", "rows": [[{"text": "계약금액"}, {"text": "100"}]]}]
        self.assertIsNone(extract_correction_notice("exchange_9", tables))

    def test_correction_marker_is_stripped_but_qualifier_is_kept(self) -> None:
        self.assertEqual(
            normalize_report_title("[기재정정]단일판매ㆍ공급계약체결"),
            normalize_report_title("단일판매·공급계약 체결"),
        )
        self.assertNotEqual(
            normalize_report_title("단일판매ㆍ공급계약체결(자율공시)"),
            normalize_report_title("단일판매ㆍ공급계약체결"),
        )


class CorrectionEdgeResolutionTests(unittest.TestCase):
    def test_single_correction_forms_a_two_document_group(self) -> None:
        records = [
            _record("exchange_1", rcept_dt="2023-01-10", rcept_no="20230110800001"),
            _record(
                "exchange_2",
                rcept_dt="2023-02-01",
                rcept_no="20230201800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(records, {"exchange_2": _notice("exchange_2", "2023-01-10")})

        chain = graph.get_correction_chain("exchange_2")
        self.assertEqual([member.doc_id for member in chain], ["exchange_1", "exchange_2"])
        self.assertEqual([member.correction_order for member in chain], [0, 1])
        self.assertIsNone(chain[0].parent_doc_id)
        self.assertEqual(chain[1].parent_doc_id, "exchange_1")
        self.assertEqual(chain[1].resolution_source, SOURCE_CORRECTION_NOTICE)
        self.assertEqual(graph.get_latest_report("exchange_1"), "exchange_2")
        self.assertEqual(
            graph.get_latest_report(correction_group_id="exchange_1"), "exchange_2"
        )

    def test_original_correction_recorrection_chain(self) -> None:
        graph = build_correction_graph(*_chain_fixture())
        chain = graph.get_correction_chain("exchange_1")
        self.assertEqual(
            [member.doc_id for member in chain],
            ["exchange_1", "exchange_2", "exchange_3"],
        )
        self.assertEqual([member.is_latest for member in chain], [False, False, True])
        self.assertEqual(graph.get_latest_report("exchange_1"), "exchange_3")
        self.assertEqual(graph.diagnostics()["multi_hop_chain_count"], 1)

    def test_recorrection_targets_the_previous_correction_not_the_original(self) -> None:
        graph = build_correction_graph(*_chain_fixture())
        chain = {member.doc_id: member for member in graph.get_correction_chain("exchange_3")}
        self.assertEqual(chain["exchange_3"].parent_doc_id, "exchange_2")
        self.assertEqual(chain["exchange_2"].parent_doc_id, "exchange_1")
        self.assertEqual(chain["exchange_3"].root_doc_id, "exchange_1")
        self.assertEqual(graph.diagnostics()["max_chain_depth"], 2)

    def test_similar_titles_in_one_company_need_the_stated_target_date(self) -> None:
        records = [
            _record("exchange_1", rcept_dt="2023-01-05", rcept_no="20230105800001"),
            _record("exchange_2", rcept_dt="2023-02-05", rcept_no="20230205800001"),
            _record(
                "exchange_3",
                rcept_dt="2023-03-01",
                rcept_no="20230301800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
            _record(
                "exchange_4",
                rcept_dt="2023-03-02",
                rcept_no="20230302800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(
            records, {"exchange_4": _notice("exchange_4", "2023-02-05")}
        )
        relations = {relation.source_doc_id: relation for relation in graph.relations}

        # Two equally plausible originals and no stated date: left ambiguous.
        self.assertEqual(relations["exchange_3"].resolution_status, AMBIGUOUS)
        self.assertEqual(
            relations["exchange_3"].resolution_source, SOURCE_MULTIPLE_CANDIDATES
        )
        self.assertIsNone(relations["exchange_3"].target_doc_id)
        self.assertEqual(
            relations["exchange_3"].evidence["event_candidates"],
            ["exchange_1", "exchange_2"],
        )
        # The stated submission date separates them.
        self.assertEqual(relations["exchange_4"].target_doc_id, "exchange_2")
        self.assertEqual(graph.get_latest_report("exchange_2"), "exchange_4")

    def test_identical_titles_in_other_companies_are_never_linked(self) -> None:
        records = [
            _record("exchange_a1", corp_code="A", rcept_dt="2023-01-10", rcept_no="1"),
            _record("exchange_b1", corp_code="B", rcept_dt="2023-01-10", rcept_no="2"),
            _record(
                "exchange_b2",
                corp_code="B",
                rcept_dt="2023-02-10",
                rcept_no="3",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(
            records, {"exchange_b2": _notice("exchange_b2", "2023-01-10")}
        )
        chain = graph.get_correction_chain("exchange_b2")
        self.assertEqual([member.doc_id for member in chain], ["exchange_b1", "exchange_b2"])
        self.assertIsNone(graph.get_correction_group("exchange_a1"))
        self.assertEqual(graph.get_latest_report("exchange_a1"), "exchange_a1")

    def test_periodic_corrections_chain_on_the_reporting_period_key(self) -> None:
        records = [
            _record(
                "periodic_1",
                doc_group="periodic",
                doc_subtype="annual",
                report_nm="사업보고서 (2024.12)",
                rcept_dt="2025-03-06",
                rcept_no="20250306000001",
                base_year=2024,
                base_month=12,
            ),
            _record(
                "periodic_2",
                doc_group="periodic",
                doc_subtype="annual",
                report_nm="[기재정정]사업보고서 (2024.12)",
                rcept_dt="2025-03-12",
                rcept_no="20250312000001",
                base_year=2024,
                base_month=12,
                is_correction=True,
            ),
            _record(
                "periodic_3",
                doc_group="periodic",
                doc_subtype="annual",
                report_nm="[기재정정]사업보고서 (2024.12)",
                rcept_dt="2025-03-18",
                rcept_no="20250318000001",
                base_year=2024,
                base_month=12,
                is_correction=True,
            ),
            # Same company, different period: must stay out of the chain.
            _record(
                "periodic_other",
                doc_group="periodic",
                doc_subtype="annual",
                report_nm="사업보고서 (2023.12)",
                rcept_dt="2024-03-06",
                rcept_no="20240306000001",
                base_year=2023,
                base_month=12,
            ),
        ]
        graph = build_correction_graph(records)
        chain = graph.get_correction_chain("periodic_2")
        self.assertEqual(
            [member.doc_id for member in chain],
            ["periodic_1", "periodic_2", "periodic_3"],
        )
        self.assertEqual(chain[2].parent_doc_id, "periodic_2")
        self.assertEqual(chain[1].resolution_source, SOURCE_PERIODIC_PERIOD_KEY)
        self.assertIsNone(graph.get_correction_group("periodic_other"))

    def test_unresolved_correction_names_a_target_outside_the_corpus(self) -> None:
        records = [
            _record("exchange_1", rcept_dt="2023-01-10", rcept_no="20230110800001"),
            _record(
                "exchange_2",
                rcept_dt="2023-02-01",
                rcept_no="20230201800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(
            records, {"exchange_2": _notice("exchange_2", "2020-12-22")}
        )
        relation = graph.relations[0]
        self.assertEqual(relation.resolution_status, UNRESOLVED)
        self.assertEqual(relation.resolution_source, SOURCE_NOTICE_TARGET_MISSING)
        self.assertIsNone(relation.target_doc_id)

        # The unresolved correction never joins the untouched original.
        member = graph.get_correction_group("exchange_2").members[0]
        self.assertEqual(member.correction_group_id, "exchange_2")
        self.assertIsNone(member.parent_doc_id)
        self.assertIsNone(graph.get_correction_group("exchange_1"))
        self.assertEqual(graph.get_latest_report("exchange_1"), "exchange_1")

    def test_correction_with_no_candidate_at_all_is_unresolved(self) -> None:
        records = [
            _record(
                "major_1",
                doc_group="major",
                doc_subtype=None,
                report_nm="[기재정정]주요사항보고서(유상증자결정)",
                rcept_dt="2023-01-04",
                rcept_no="20230104000001",
                is_correction=True,
            )
        ]
        graph = build_correction_graph(records)
        self.assertEqual(graph.relations[0].resolution_status, UNRESOLVED)
        self.assertEqual(graph.diagnostics()["unresolved_relations"], 1)

    def test_ambiguous_correction_does_not_disturb_a_resolved_group(self) -> None:
        records = [
            _record("exchange_1", rcept_dt="2023-01-05", rcept_no="20230105800001"),
            _record("exchange_2", rcept_dt="2023-02-05", rcept_no="20230205800001"),
            _record(
                "exchange_3",
                rcept_dt="2023-03-01",
                rcept_no="20230301800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
            _record(
                "exchange_4",
                rcept_dt="2023-03-02",
                rcept_no="20230302800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(
            records, {"exchange_4": _notice("exchange_4", "2023-02-05")}
        )
        ambiguous = graph.get_correction_group("exchange_3")
        self.assertEqual(ambiguous.resolution_status, AMBIGUOUS)
        self.assertEqual(ambiguous.doc_ids, ("exchange_3",))
        self.assertTrue(ambiguous.members[0].is_latest)

        resolved = graph.get_correction_group("exchange_2")
        self.assertEqual(resolved.doc_ids, ("exchange_2", "exchange_4"))
        self.assertEqual(resolved.latest_doc_id, "exchange_4")
        self.assertNotIn("exchange_3", resolved.doc_ids)


class CorrectionGraphGuardTests(unittest.TestCase):
    def test_self_reference_is_rejected(self) -> None:
        records = [_record("exchange_1", is_correction=True)]
        relation = CorrectionRelation(
            relation_id=relation_id("exchange_1", CORRECTION_RELATION_TYPE, "exchange_1"),
            source_doc_id="exchange_1",
            target_doc_id="exchange_1",
            resolution_status=RESOLVED,
            resolution_source=SOURCE_CORRECTION_NOTICE,
            confidence=0.95,
        )
        members, relations = assemble_correction_groups(records, [relation])
        self.assertEqual(relations[0].resolution_status, AMBIGUOUS)
        self.assertEqual(relations[0].resolution_source, SOURCE_SELF_REFERENCE)
        self.assertIsNone(relations[0].target_doc_id)
        self.assertEqual([member.parent_doc_id for member in members], [None])
        self.assertEqual(members[0].correction_group_id, "exchange_1")

    def test_a_cycle_is_detected_and_no_group_is_formed(self) -> None:
        records = [
            _record("exchange_1", rcept_no="1", is_correction=True),
            _record("exchange_2", rcept_no="2", is_correction=True),
        ]
        edges = [
            CorrectionRelation(
                relation_id=relation_id("exchange_1", CORRECTION_RELATION_TYPE, "exchange_2"),
                source_doc_id="exchange_1",
                target_doc_id="exchange_2",
                resolution_status=RESOLVED,
                resolution_source=SOURCE_CORRECTION_NOTICE,
                confidence=0.95,
            ),
            CorrectionRelation(
                relation_id=relation_id("exchange_2", CORRECTION_RELATION_TYPE, "exchange_1"),
                source_doc_id="exchange_2",
                target_doc_id="exchange_1",
                resolution_status=RESOLVED,
                resolution_source=SOURCE_CORRECTION_NOTICE,
                confidence=0.95,
            ),
        ]
        members, relations = assemble_correction_groups(records, edges)
        self.assertEqual(
            {relation.resolution_source for relation in relations},
            {SOURCE_CYCLE_DETECTED},
        )
        self.assertEqual({relation.resolution_status for relation in relations}, {AMBIGUOUS})
        self.assertEqual({member.parent_doc_id for member in members}, {None})
        self.assertEqual(
            sorted(member.correction_group_id for member in members),
            ["exchange_1", "exchange_2"],
        )

    def test_duplicate_edges_for_one_source_produce_one_parent(self) -> None:
        records = [
            _record("exchange_1", rcept_no="1"),
            _record("exchange_2", rcept_no="2"),
            _record("exchange_3", rcept_no="3", is_correction=True),
        ]
        edge = CorrectionRelation(
            relation_id=relation_id("exchange_3", CORRECTION_RELATION_TYPE, "exchange_1"),
            source_doc_id="exchange_3",
            target_doc_id="exchange_1",
            resolution_status=RESOLVED,
            resolution_source=SOURCE_CORRECTION_NOTICE,
            confidence=0.95,
        )
        duplicate = CorrectionRelation(
            relation_id=relation_id("exchange_3", CORRECTION_RELATION_TYPE, "exchange_2"),
            source_doc_id="exchange_3",
            target_doc_id="exchange_2",
            resolution_status=RESOLVED,
            resolution_source=SOURCE_EVENT_TITLE_KEY,
            confidence=0.70,
        )
        members, _relations = assemble_correction_groups(records, [edge, duplicate])
        parents = {member.doc_id: member.parent_doc_id for member in members}
        self.assertEqual(parents["exchange_3"], "exchange_1")

    def test_every_resolved_group_has_exactly_one_latest(self) -> None:
        records, notices = _chain_fixture()
        records += [
            _record("exchange_x1", corp_code="B", rcept_dt="2023-01-02", rcept_no="10"),
            _record(
                "exchange_x2",
                corp_code="B",
                rcept_dt="2023-04-02",
                rcept_no="11",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
            _record(
                "exchange_x3",
                corp_code="C",
                rcept_dt="2023-04-03",
                rcept_no="12",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(records, notices)
        by_group: dict[str, list] = {}
        for member in graph.members:
            by_group.setdefault(member.correction_group_id, []).append(member)
        for group_id, members in by_group.items():
            with self.subTest(group=group_id):
                self.assertEqual(sum(1 for m in members if m.is_latest), 1)
        diagnostics = graph.diagnostics()
        self.assertEqual(diagnostics["invalid_latest_group_count"], 0)
        self.assertEqual(diagnostics["cycle_count"], 0)
        self.assertEqual(diagnostics["self_reference_count"], 0)

    def test_building_twice_is_identical_and_free_of_duplicates(self) -> None:
        records, notices = _chain_fixture()
        first = build_correction_graph(records, notices)
        second = build_correction_graph(records, notices)
        self.assertEqual(
            [relation.to_dict() for relation in first.relations],
            [relation.to_dict() for relation in second.relations],
        )
        self.assertEqual(
            [member.to_dict() for member in first.members],
            [member.to_dict() for member in second.members],
        )
        self.assertEqual(second.diagnostics()["duplicate_relation_count"], 0)
        self.assertEqual(
            len({relation.relation_id for relation in second.relations}),
            len(second.relations),
        )


class FakeCorrectionCursor:
    def __init__(self, store: "FakeCorrectionStore") -> None:
        self.store = store
        self.rowcount = 0
        self.rows: list[dict] = []

    def __enter__(self) -> "FakeCorrectionCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: tuple = ()) -> None:
        self.rowcount = self.store.execute(query, params)
        self.rows = []

    def executemany(self, query: str, params: list[tuple]) -> None:
        for row in params:
            self.store.insert(query, row)
        self.rowcount = len(params)

    def fetchall(self) -> list[dict]:
        return self.rows

    def fetchone(self) -> dict | None:
        return self.rows[0] if self.rows else None


class FakeCorrectionConnection:
    def __init__(self, store: "FakeCorrectionStore") -> None:
        self.store = store

    def cursor(self, **_kwargs: object) -> FakeCorrectionCursor:
        return FakeCorrectionCursor(self.store)

    def commit(self) -> None:
        self.store.commits += 1

    def close(self) -> None:
        return None


class FakeCorrectionStore:
    """Enough of the two correction tables to prove the upsert converges."""

    def __init__(self) -> None:
        self.relations: dict[str, tuple] = {}
        self.members: dict[str, tuple] = {}
        self.commits = 0

    def connection(self):
        store = self

        class _Context:
            def __enter__(self) -> FakeCorrectionConnection:
                return FakeCorrectionConnection(store)

            def __exit__(self, *_args: object) -> None:
                return None

        return _Context()

    def execute(self, query: str, params: tuple = ()) -> int:
        scoped = "source_doc_id = ANY(%s)" in query or "doc_id = ANY(%s) AND NOT" in query
        if "DELETE FROM correction_relations" in query:
            scope = set(params[0]) if scoped else None
            keep = set(params[1] if scoped else params[0])
            removed = [
                key
                for key, row in self.relations.items()
                if key not in keep and (scope is None or row[1] in scope)
            ]
            for key in removed:
                del self.relations[key]
            return len(removed)
        if "DELETE FROM correction_group_members" in query:
            scope = set(params[0]) if scoped else None
            keep = set(params[1] if scoped else params[0])
            removed = [
                key
                for key in self.members
                if key not in keep and (scope is None or key in scope)
            ]
            for key in removed:
                del self.members[key]
            return len(removed)
        if "SET is_latest = false" in query:
            member_ids, group_ids = set(params[0]), set(params[1])
            cleared = 0
            for doc_id, row in list(self.members.items()):
                if row[5] and (doc_id in member_ids or row[1] in group_ids):
                    self.members[doc_id] = (*row[:5], False, *row[6:])
                    cleared += 1
            return cleared
        raise AssertionError(f"unexpected statement: {query}")

    def insert(self, query: str, row: tuple) -> None:
        if "INSERT INTO correction_relations" in query:
            self.relations[row[0]] = row
            return
        if "INSERT INTO correction_group_members" in query:
            if row[5]:
                clash = [
                    doc_id
                    for doc_id, other in self.members.items()
                    if other[1] == row[1] and other[5] and doc_id != row[0]
                ]
                if clash:
                    raise AssertionError(
                        f"group {row[1]} would hold two latest documents"
                    )
            self.members[row[0]] = row
            return
        raise AssertionError(f"unexpected statement: {query}")


class CorrectionRepositoryPersistenceTests(unittest.TestCase):
    def test_backfill_run_twice_leaves_one_row_per_relation_and_member(self) -> None:
        graph = build_correction_graph(*_chain_fixture())
        store = FakeCorrectionStore()
        repository = PostgresCorrectionRepository(store)

        first = repository.persist_graph(graph)
        relations_after_first = dict(store.relations)
        members_after_first = dict(store.members)
        second = repository.persist_graph(graph)

        self.assertEqual(first["relations_written"], second["relations_written"])
        self.assertEqual(second["relations_removed"], 0)
        self.assertEqual(second["members_removed"], 0)
        self.assertEqual(store.relations, relations_after_first)
        self.assertEqual(store.members, members_after_first)
        self.assertEqual(len(store.relations), len(graph.relations))
        self.assertEqual(len(store.members), len(graph.members))
        self.assertEqual(store.commits, 2)

    def test_a_scoped_write_never_deletes_rows_outside_its_scope(self) -> None:
        """A build over part of the corpus must not touch what it never saw."""

        store = FakeCorrectionStore()
        repository = PostgresCorrectionRepository(store)
        repository.persist_graph(build_correction_graph(*_chain_fixture()))
        self.assertEqual(sorted(store.members), ["exchange_1", "exchange_2", "exchange_3"])

        # A separate company's disclosures, built and written on their own.
        other = [
            _record("other_1", corp_code="B", rcept_dt="2023-05-01", rcept_no="90"),
            _record(
                "other_2",
                corp_code="B",
                rcept_dt="2023-06-01",
                rcept_no="91",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        stats = repository.persist_graph(
            build_correction_graph(other),
            scope_doc_ids={record.doc_id for record in other},
        )

        self.assertFalse(stats["full_rebuild"])
        self.assertEqual(stats["scope_doc_count"], 2)
        self.assertEqual(stats["members_removed"], 0)
        self.assertEqual(stats["relations_removed"], 0)
        self.assertEqual(
            sorted(store.members),
            ["exchange_1", "exchange_2", "exchange_3", "other_1", "other_2"],
        )
        # The untouched chain keeps its final document.
        self.assertTrue(store.members["exchange_3"][5])
        self.assertFalse(store.members["exchange_1"][5])

    def test_a_full_rebuild_from_a_partial_graph_would_delete_everything_else(self) -> None:
        """The destructive case the build script refuses to run unattended."""

        store = FakeCorrectionStore()
        repository = PostgresCorrectionRepository(store)
        repository.persist_graph(build_correction_graph(*_chain_fixture()))

        other = [
            _record("other_1", corp_code="B", rcept_dt="2023-05-01", rcept_no="90"),
            _record(
                "other_2",
                corp_code="B",
                rcept_dt="2023-06-01",
                rcept_no="91",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        stats = repository.persist_graph(build_correction_graph(other))

        self.assertTrue(stats["full_rebuild"])
        self.assertEqual(stats["members_removed"], 3)
        self.assertEqual(sorted(store.members), ["other_1", "other_2"])

    def test_rebuild_removes_rows_the_new_graph_no_longer_holds(self) -> None:
        graph = build_correction_graph(*_chain_fixture())
        store = FakeCorrectionStore()
        repository = PostgresCorrectionRepository(store)
        repository.persist_graph(graph)

        records, notices = _chain_fixture()
        smaller = build_correction_graph(records[:2], {"exchange_2": notices["exchange_2"]})
        stats = repository.persist_graph(smaller)

        self.assertEqual(stats["relations_removed"], 1)
        self.assertEqual(stats["members_removed"], 1)
        self.assertNotIn("exchange_3", store.members)
        latest = [doc_id for doc_id, row in store.members.items() if row[5]]
        self.assertEqual(latest, ["exchange_2"])


class ResolvedLatestSemanticsTests(unittest.TestCase):
    """`is_latest` alone must never be read as "verified final version"."""

    def setUp(self) -> None:
        records = [
            _record("exchange_1", rcept_dt="2023-01-05", rcept_no="1"),
            _record("exchange_2", rcept_dt="2023-02-05", rcept_no="2"),
            _record(
                "exchange_3",
                rcept_dt="2023-03-01",
                rcept_no="3",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        # exchange_3 has two equally plausible originals, so it stays ambiguous
        # and is stored as a group of one -- carrying is_latest.
        self.graph = build_correction_graph(records)
        self.states = self.graph.document_states(["exchange_3"])

    def test_ambiguous_singleton_carries_is_latest_but_is_not_resolved_latest(self) -> None:
        state = self.states["exchange_3"]
        self.assertTrue(state.is_latest)
        self.assertFalse(state.is_resolved)
        self.assertFalse(state.is_resolved_latest)
        self.assertFalse(state.is_superseded)
        # It looks like a root too, which is why is_group_root needs the guard.
        self.assertTrue(state.is_group_root)
        self.assertFalse(state.is_resolved_root)

    def test_ambiguous_group_reports_no_resolved_latest(self) -> None:
        group = self.graph.get_correction_group("exchange_3")
        self.assertEqual(group.resolution_status, AMBIGUOUS)
        self.assertFalse(group.is_resolved)
        self.assertEqual(group.latest_doc_id, "exchange_3")
        self.assertIsNone(group.resolved_latest_doc_id)

    def test_an_ambiguous_correction_is_ranked_exactly_as_before_the_graph(self) -> None:
        """Its boost comes from is_correction, never from its singleton is_latest.

        Leaving it at the pre-graph value is deliberate: the document really is
        a correction, so demoting it would lose recall, and promoting it as a
        group's verified final version would be a claim the evidence does not
        support.  What must not happen is the boost coming from ``is_latest``.
        """

        state = self.states["exchange_3"]
        self.assertTrue(state.is_latest)
        self.assertFalse(state.is_resolved_latest)
        with_graph = prefers_document(
            POLICY_LATEST_PREFERRED, is_correction=True, state=state
        )
        without_graph = prefers_document(
            POLICY_LATEST_PREFERRED, is_correction=True, state=None
        )
        self.assertEqual(with_graph, without_graph)

    def test_only_a_resolved_final_document_is_promoted_over_its_predecessor(self) -> None:
        resolved = build_correction_graph(*_chain_fixture())
        states = resolved.document_states(["exchange_2", "exchange_3"])
        superseded, final = states["exchange_2"], states["exchange_3"]

        self.assertTrue(final.is_resolved_latest)
        self.assertTrue(superseded.is_superseded)
        self.assertTrue(
            prefers_document(POLICY_LATEST_PREFERRED, is_correction=True, state=final)
        )
        # The behaviour change the graph buys: a superseded correction stops
        # being preferred just because it is a correction.
        self.assertFalse(
            prefers_document(
                POLICY_LATEST_PREFERRED, is_correction=True, state=superseded
            )
        )

    def test_an_ambiguous_correction_never_displaces_a_resolved_latest(self) -> None:
        records, notices = _chain_fixture()
        records.append(
            _record(
                "exchange_9",
                rcept_dt="2023-04-01",
                rcept_no="20230401800001",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            )
        )
        graph = build_correction_graph(records, notices)
        # exchange_9 has several plausible originals, so it stands alone.
        self.assertEqual(graph.get_correction_group("exchange_9").doc_ids, ("exchange_9",))
        chain = graph.get_correction_group("exchange_1")
        self.assertEqual(chain.doc_ids, ("exchange_1", "exchange_2", "exchange_3"))
        self.assertEqual(chain.resolved_latest_doc_id, "exchange_3")

    def test_summary_hides_latest_for_an_unresolved_group(self) -> None:
        summary = correction_summary(self.graph, ["exchange_3"])
        group = summary["groups"]["exchange_3"]
        self.assertFalse(group["is_resolved"])
        self.assertIsNone(group["latest_doc_id"])
        self.assertEqual(group["resolution_status"], AMBIGUOUS)


class UnavailableGraphTests(unittest.TestCase):
    """Only "the graph is not there" degrades; a real defect must surface."""

    class _Unavailable:
        def document_states(self, doc_ids):
            raise CorrectionGraphUnavailable("db/006 not applied")

        def get_correction_chain(self, doc_id):
            raise CorrectionGraphUnavailable("db/006 not applied")

    class _Broken:
        def document_states(self, doc_ids):
            raise RuntimeError("undefined column correction_group_members.oops")

    def test_missing_migration_falls_back_and_logs(self) -> None:
        import app.reasoning.correction_policy as policy_module

        policy_module._REPORTED_UNAVAILABLE.clear()
        with self.assertLogs(policy_module.logger, level="WARNING") as logs:
            self.assertEqual(document_states(self._Unavailable(), ["d1"]), {})
        self.assertIn("006_correction_graph.sql", "".join(logs.output))

    def test_repeated_unavailability_warns_once(self) -> None:
        import app.reasoning.correction_policy as policy_module

        policy_module._REPORTED_UNAVAILABLE.clear()
        graph = self._Unavailable()
        with self.assertLogs(policy_module.logger, level="DEBUG") as logs:
            document_states(graph, ["d1"])
            document_states(graph, ["d2"])
        warnings = [line for line in logs.output if line.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)

    def test_a_programming_error_is_not_swallowed(self) -> None:
        with self.assertRaises(RuntimeError):
            document_states(self._Broken(), ["d1"])

    def test_router_degrades_to_pre_graph_behaviour_when_unavailable(self) -> None:
        plan = _plan("삼성전자 정정 계약금액")
        chunks = [
            _chunk("exchange_2", is_correction=True),
            _chunk("exchange_1", is_correction=False),
        ]
        import app.reasoning.correction_policy as policy_module

        policy_module._REPORTED_UNAVAILABLE.clear()
        router = QueryRouter(correction_graph=self._Unavailable())
        with self.assertLogs(policy_module.logger, level="WARNING"):
            prepared = router.prepare_chunks(chunks, router.route(plan))
        boosts = {
            candidate.doc_id: candidate.metadata_match.soft_boosts["is_correction"]
            for candidate in prepared
        }
        self.assertEqual(boosts, {"exchange_1": False, "exchange_2": True})


def _plan(query: str, *, correction_policy: str | None = None) -> QueryPlan:
    from app.reasoning.query_understanding import QueryUnderstanding

    return QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
        query, correction_policy=correction_policy
    )


def _document(doc_id: str, *, is_correction: bool) -> CandidateDocument:
    return CandidateDocument(
        doc_id=doc_id,
        metadata={
            "doc_id": doc_id,
            "doc_group": "exchange",
            "is_correction": is_correction,
            "report_nm": "단일판매ㆍ공급계약체결",
        },
        metadata_match=MetadataMatch(),
    )


def _chunk(doc_id: str, *, is_correction: bool) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=f"{doc_id}:c1",
        doc_id=doc_id,
        chunk={
            "doc_id": doc_id,
            "doc_group": "exchange",
            "is_correction": is_correction,
            "content": "계약금액 100원",
            "retrieval_text": "계약금액 100원",
        },
        metadata_match=MetadataMatch(),
    )


class CorrectionPolicyRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_correction_graph(*_chain_fixture())
        self.documents = [
            _document("exchange_1", is_correction=False),
            _document("exchange_2", is_correction=True),
            _document("exchange_3", is_correction=True),
        ]
        self.chunks = [
            _chunk("exchange_1", is_correction=False),
            _chunk("exchange_2", is_correction=True),
            _chunk("exchange_3", is_correction=True),
        ]

    def test_latest_preferred_boosts_the_final_document_not_every_correction(self) -> None:
        plan = _plan("삼성전자 정정 계약금액")
        self.assertEqual(plan.correction_policy, POLICY_LATEST_PREFERRED)
        router = QueryRouter(correction_graph=self.graph)
        route = router.route(plan)
        boosts = {
            candidate.doc_id: candidate.metadata_match.soft_boosts["is_correction"]
            for candidate in router.prepare_chunks(self.chunks, route)
        }
        self.assertEqual(
            boosts,
            {"exchange_1": False, "exchange_2": False, "exchange_3": True},
        )

        # Without the graph the pre-existing behaviour is unchanged.
        plain = QueryRouter()
        plain_boosts = {
            candidate.doc_id: candidate.metadata_match.soft_boosts["is_correction"]
            for candidate in plain.prepare_chunks(self.chunks, plain.route(plan))
        }
        self.assertEqual(
            plain_boosts,
            {"exchange_1": False, "exchange_2": True, "exchange_3": True},
        )

    def test_latest_preferred_ranks_the_final_document_above_a_superseded_one(self) -> None:
        from app.retrieval.interfaces import RetrievalResult

        plan = _plan("삼성전자 정정 계약금액")
        router = QueryRouter(correction_graph=self.graph)
        route = router.route(plan)
        prepared = router.prepare_chunks(self.chunks, route)
        results = [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                bm25_score=1.0,
                rank=index,
                metadata_match=candidate.metadata_match.to_dict(),
            )
            # The superseded correction starts ahead of the final document.
            for index, candidate in enumerate(
                [prepared[1], prepared[2], prepared[0]], start=1
            )
        ]
        reranked = router.rerank(results, route, chunks=prepared, top_k=3)
        self.assertEqual(reranked[0].doc_id, "exchange_3")

    def test_original_only_keeps_the_document_each_chain_starts_from(self) -> None:
        plan = _plan("삼성전자 정정 제외 계약금액")
        self.assertEqual(plan.correction_policy, POLICY_ORIGINAL_ONLY)
        router = QueryRouter(correction_graph=self.graph)
        route = router.route(plan)
        kept = router.filter_documents(self.documents, route)
        self.assertEqual([document.doc_id for document in kept], ["exchange_1"])

    def test_corrected_only_keeps_correcting_documents(self) -> None:
        plan = _plan("삼성전자 정정 공시만 계약금액")
        self.assertEqual(plan.correction_policy, POLICY_CORRECTED_ONLY)
        router = QueryRouter(correction_graph=self.graph)
        route = router.route(plan)
        kept = router.filter_documents(self.documents, route)
        self.assertEqual(
            [document.doc_id for document in kept], ["exchange_2", "exchange_3"]
        )

    def test_ambiguous_correction_still_answers_original_and_corrected_policies(self) -> None:
        records = [
            _record("exchange_1", rcept_dt="2023-01-05", rcept_no="1"),
            _record("exchange_2", rcept_dt="2023-02-05", rcept_no="2"),
            _record(
                "exchange_3",
                rcept_dt="2023-03-01",
                rcept_no="3",
                report_nm="[기재정정]단일판매ㆍ공급계약체결",
                is_correction=True,
            ),
        ]
        graph = build_correction_graph(records)
        documents = [
            _document("exchange_1", is_correction=False),
            _document("exchange_2", is_correction=False),
            _document("exchange_3", is_correction=True),
        ]
        router = QueryRouter(correction_graph=graph)
        original = router.route(_plan("삼성전자 정정 제외 계약금액"))
        corrected = router.route(_plan("삼성전자 정정 공시만 계약금액"))
        self.assertEqual(
            [item.doc_id for item in router.filter_documents(documents, original)],
            ["exchange_1", "exchange_2"],
        )
        self.assertEqual(
            [item.doc_id for item in router.filter_documents(documents, corrected)],
            ["exchange_3"],
        )

    def test_before_after_comparison_reads_a_time_ordered_chain(self) -> None:
        timeline = correction_timeline(self.graph.get_correction_chain("exchange_3"))
        self.assertEqual(
            [(item["doc_id"], item["role"]) for item in timeline],
            [
                ("exchange_1", "original"),
                ("exchange_2", "correction"),
                ("exchange_3", "correction"),
            ],
        )
        self.assertEqual([item["is_latest"] for item in timeline], [False, False, True])

        summary = correction_summary(
            self.graph,
            ["exchange_1", "exchange_2", "exchange_3"],
            policy=POLICY_LATEST_PREFERRED,
        )
        self.assertEqual(summary["group_count"], 1)
        self.assertEqual(summary["superseded_doc_ids"], ["exchange_1", "exchange_2"])
        group = summary["groups"]["exchange_1"]
        self.assertEqual(group["latest_doc_id"], "exchange_3")
        self.assertEqual(
            [item["doc_id"] for item in group["timeline"]],
            ["exchange_1", "exchange_2", "exchange_3"],
        )

    def test_router_surfaces_the_correction_chain_in_its_routing_summary(self) -> None:
        router = QueryRouter(correction_graph=self.graph)
        route = router.route(_plan("삼성전자 정정 계약금액"))
        summary = router.correction_summary(self.documents, route)
        self.assertEqual(summary["policy"], POLICY_LATEST_PREFERRED)
        self.assertEqual(summary["groups"]["exchange_1"]["latest_doc_id"], "exchange_3")
        self.assertEqual(QueryRouter().correction_summary(self.documents, route), {})


class _CorrectionBackend:
    """A metadata/chunk/retrieval backend over one correction chain."""

    def __init__(self, documents, chunks) -> None:
        self._documents = list(documents)
        self._chunks = {chunk.doc_id: chunk for chunk in chunks}

    def get_candidate_documents(self, **_filters):
        return list(self._documents)

    def get_candidate_chunks(self, documents):
        return [
            self._chunks[document.doc_id]
            for document in documents
            if document.doc_id in self._chunks
        ]

    def retrieve(self, _query, candidates, *, top_k=None):
        ranked = list(candidates)[: top_k or len(candidates)]
        return [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                bm25_score=1.0,
                rank=rank,
                metadata_match=candidate.metadata_match.to_dict(),
            )
            for rank, candidate in enumerate(ranked, start=1)
        ]


class CorrectionExecutionIntegrationTests(unittest.TestCase):
    """The whole plan -> route -> retrieve path, with and without the graph."""

    def setUp(self) -> None:
        self.graph = build_correction_graph(*_chain_fixture())
        self.documents = [
            _document("exchange_1", is_correction=False),
            _document("exchange_2", is_correction=True),
            _document("exchange_3", is_correction=True),
        ]
        self.chunks = [
            _chunk("exchange_1", is_correction=False),
            _chunk("exchange_2", is_correction=True),
            _chunk("exchange_3", is_correction=True),
        ]
        self.backend = _CorrectionBackend(self.documents, self.chunks)

    def _execute(self, plan: QueryPlan, *, graph) -> object:
        return QueryExecutor(
            self.backend, router=QueryRouter(correction_graph=graph)
        ).execute(plan)

    def test_execution_reports_the_chain_and_the_superseded_documents(self) -> None:
        execution = self._execute(_plan("삼성전자 정정 계약금액"), graph=self.graph)
        correction = execution.routing["correction"]
        self.assertEqual(correction["policy"], POLICY_LATEST_PREFERRED)
        self.assertEqual(
            correction["groups"]["exchange_1"]["latest_doc_id"], "exchange_3"
        )
        self.assertEqual(
            [item["doc_id"] for item in correction["groups"]["exchange_1"]["timeline"]],
            ["exchange_1", "exchange_2", "exchange_3"],
        )
        self.assertEqual(
            correction["superseded_doc_ids"], ["exchange_1", "exchange_2"]
        )

    def test_execution_without_a_graph_keeps_its_previous_routing_shape(self) -> None:
        execution = self._execute(_plan("삼성전자 정정 계약금액"), graph=None)
        self.assertNotIn("correction", execution.routing)
        self.assertEqual(len(execution.results), 3)

    def test_original_only_execution_returns_only_the_original_document(self) -> None:
        execution = self._execute(
            _plan("삼성전자 정정 제외 계약금액"), graph=self.graph
        )
        self.assertEqual(
            [document.doc_id for document in execution.documents], ["exchange_1"]
        )
        self.assertEqual(
            [result.doc_id for result in execution.results], ["exchange_1"]
        )


class OrdinaryDisclosureRegressionTests(unittest.TestCase):
    """Documents outside any correction group must retrieve exactly as before."""

    def setUp(self) -> None:
        self.graph = build_correction_graph(*_chain_fixture())
        self.documents = [
            _document("plain_1", is_correction=False),
            _document("plain_2", is_correction=False),
        ]
        self.chunks = [
            _chunk("plain_1", is_correction=False),
            _chunk("plain_2", is_correction=False),
        ]

    def _routed(self, router: QueryRouter, plan: QueryPlan) -> tuple:
        route = router.route(plan)
        documents = router.filter_documents(self.documents, route)
        chunks = router.prepare_chunks(self.chunks, route)
        return (
            [document.doc_id for document in documents],
            [candidate.metadata_match.to_dict() for candidate in chunks],
        )

    def _compare(self, plan: QueryPlan) -> None:
        self.assertEqual(
            self._routed(QueryRouter(correction_graph=self.graph), plan),
            self._routed(QueryRouter(), plan),
        )

    def test_plain_question_is_unchanged_by_the_graph(self) -> None:
        self._compare(_plan("삼성전자 2024년 매출액"))

    def test_correction_question_over_uncorrected_documents_is_unchanged(self) -> None:
        self._compare(_plan("삼성전자 정정 매출액"))

    def test_correction_summary_is_empty_for_documents_without_corrections(self) -> None:
        router = QueryRouter(correction_graph=self.graph)
        route = router.route(_plan("삼성전자 정정 매출액"))
        self.assertEqual(router.correction_summary(self.documents, route), {})


class RouteRegressionTests(unittest.TestCase):
    """Every disclosure route retrieves identically with the graph switched on.

    The graph in these tests is populated -- it just holds no row for any of the
    documents under test, which is the situation for every disclosure in the
    corpus that was never corrected.
    """

    ROUTES = {
        "periodic": ("사업보고서 (2024.12)", "annual"),
        "holding": ("주식등의대량보유상황보고서(일반)", "대량보유상황보고서"),
        "major": ("주요사항보고서(유상증자결정)", None),
        "exchange": ("단일판매ㆍ공급계약체결", "단일판매공급계약체결"),
    }
    QUERIES = (
        "삼성전자 2024년 매출액",
        "삼성전자 정정 매출액",
        "삼성전자 정정 공시만 매출액",
        "삼성전자 정정 제외 매출액",
    )

    def setUp(self) -> None:
        # A populated graph that simply holds no row for these documents.
        self.graph = build_correction_graph(*_chain_fixture())

    def _documents(self, doc_group: str) -> list[CandidateDocument]:
        report_nm, subtype = self.ROUTES[doc_group]
        return [
            CandidateDocument(
                doc_id=f"{doc_group}_plain_{index}",
                metadata={
                    "doc_id": f"{doc_group}_plain_{index}",
                    "doc_group": doc_group,
                    "doc_subtype": subtype,
                    "report_nm": report_nm,
                    "is_correction": False,
                    "rcept_dt": "2024-03-01",
                },
                metadata_match=MetadataMatch(),
            )
            for index in (1, 2)
        ]

    def _chunks(self, documents) -> list[CandidateChunk]:
        return [
            CandidateChunk(
                chunk_id=f"{document.doc_id}:c1",
                doc_id=document.doc_id,
                chunk={
                    **document.metadata,
                    "content": "매출액 100원",
                    "retrieval_text": "매출액 100원",
                    "section_path": ["III. 재무에 관한 사항"],
                },
                metadata_match=MetadataMatch(),
            )
            for document in documents
        ]

    def _run(self, router: QueryRouter, plan: QueryPlan, doc_group: str) -> tuple:
        documents = self._documents(doc_group)
        route = router.route(plan)
        kept = router.filter_documents(documents, route)
        prepared = router.prepare_chunks(self._chunks(kept), route)
        results = [
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                bm25_score=1.0 - index * 0.1,
                rank=index + 1,
                metadata_match=candidate.metadata_match.to_dict(),
            )
            for index, candidate in enumerate(prepared)
        ]
        reranked = router.rerank(results, route, chunks=prepared, top_k=10)
        return (
            [document.doc_id for document in kept],
            [candidate.chunk_id for candidate in prepared],
            [candidate.metadata_match.to_dict() for candidate in prepared],
            [(result.rank, result.chunk_id, result.doc_id) for result in reranked],
            router.correction_summary(kept, route),
        )

    def test_uncorrected_documents_are_identical_on_every_route(self) -> None:
        with_graph = QueryRouter(correction_graph=self.graph)
        without = QueryRouter()
        for doc_group in self.ROUTES:
            for query in self.QUERIES:
                with self.subTest(route=doc_group, query=query):
                    plan = _plan(query)
                    self.assertEqual(
                        self._run(with_graph, plan, doc_group),
                        self._run(without, plan, doc_group),
                    )

    def test_no_correction_block_is_reported_for_uncorrected_routes(self) -> None:
        router = QueryRouter(correction_graph=self.graph)
        for doc_group in self.ROUTES:
            with self.subTest(route=doc_group):
                route = router.route(_plan("삼성전자 정정 매출액"))
                self.assertEqual(
                    router.correction_summary(self._documents(doc_group), route), {}
                )


class BuildScriptSafetyTests(unittest.TestCase):
    """``--write`` must not rewrite the whole graph from a partial corpus."""

    class _Repository:
        def __init__(self, stored: set[str]) -> None:
            self._stored = stored
            self.persisted: list[dict] = []

        def disclosure_doc_ids(self) -> set[str]:
            return set(self._stored)

        def persist_graph(self, graph, *, scope_doc_ids=None):
            self.persisted.append(
                {"members": len(graph.members), "scope": scope_doc_ids}
            )
            return {"relations_written": len(graph.relations)}

    def _main(self, repository, records, argv: list[str]) -> None:
        import scripts.build_correction_graph as script

        manifest = [{"doc_id": record.doc_id} for record in records]
        by_doc = {record.doc_id: record for record in records}
        patches = [
            mock.patch.object(sys, "argv", ["build_correction_graph.py", *argv]),
            mock.patch.object(script, "load_manifest", return_value=manifest),
            mock.patch.object(script, "collect_notices", return_value={}),
            # The prose pass reads the same corpus the table pass does; this
            # harness has no corpus, so both readers are stubbed together.
            mock.patch.object(script, "collect_prose_notices", return_value={}),
            mock.patch.object(
                script.DisclosureRecord,
                "from_mapping",
                staticmethod(lambda row: by_doc[row["doc_id"]]),
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "app.retrieval.correction_repository": mock.Mock(
                        PostgresCorrectionRepository=lambda _backend: repository
                    ),
                    "app.retrieval.postgres_backend": mock.Mock(
                        PostgresBackend=lambda: object()
                    ),
                },
            ),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(
                contextlib.redirect_stdout(io.StringIO())
            )
            script.main()

    def test_write_is_refused_when_the_build_misses_disclosures(self) -> None:
        records, _notices = _chain_fixture()
        repository = self._Repository(
            {"exchange_1", "exchange_2", "exchange_3", "absent_1"}
        )
        with self.assertRaises(SystemExit) as raised:
            self._main(repository, records, ["--write"])
        self.assertIn("refusing to write", str(raised.exception))
        self.assertIn("absent_1", str(raised.exception))
        self.assertEqual(repository.persisted, [])

    def test_a_full_build_writes_without_a_scope(self) -> None:
        records, _notices = _chain_fixture()
        repository = self._Repository({"exchange_1", "exchange_2", "exchange_3"})
        self._main(repository, records, ["--write"])
        self.assertEqual(len(repository.persisted), 1)
        self.assertIsNone(repository.persisted[0]["scope"])

    def test_an_allowed_partial_write_is_scoped_to_what_it_built(self) -> None:
        records, _notices = _chain_fixture()
        repository = self._Repository(
            {"exchange_1", "exchange_2", "exchange_3", "absent_1"}
        )
        self._main(repository, records, ["--write", "--allow-partial-write"])
        self.assertEqual(len(repository.persisted), 1)
        self.assertEqual(
            repository.persisted[0]["scope"],
            {"exchange_1", "exchange_2", "exchange_3"},
        )

    def test_sample_only_prints_and_never_changes_what_is_written(self) -> None:
        records, _notices = _chain_fixture()
        plain = self._Repository({"exchange_1", "exchange_2", "exchange_3"})
        sampled = self._Repository({"exchange_1", "exchange_2", "exchange_3"})
        self._main(plain, records, ["--write"])
        self._main(sampled, records, ["--write", "--sample", "5"])
        self.assertEqual(plain.persisted, sampled.persisted)


class PointInTimeScopeTests(unittest.TestCase):
    """What a date-bounded question does with corrections filed later.

    P0-A has no point-in-time snapshot feature.  What it does have is the
    existing ``rcept_dt`` hard filter, and these tests pin down exactly how far
    that gets: a correction filed after the window is never retrieved, while the
    reported chain stays the document's full history.
    """

    def setUp(self) -> None:
        self.graph = build_correction_graph(*_chain_fixture())
        self.documents = []
        for doc_id, rcept_dt, corrected in (
            ("exchange_1", "2023-01-10", False),
            ("exchange_2", "2023-02-01", True),
            ("exchange_3", "2023-03-01", True),
        ):
            document = _document(doc_id, is_correction=corrected)
            document.metadata["rcept_dt"] = rcept_dt
            self.documents.append(document)
        self.router = QueryRouter(correction_graph=self.graph)

    def _windowed_plan(self, to_date: str) -> QueryPlan:
        return QueryPlan(
            query="계약금액",
            raw_query=f"2023-01-10부터 {to_date}까지 공시된 계약금액",
            period=QueryPeriod(
                from_date="2023-01-10", to_date=to_date, period_type="receipt_date"
            ),
            correction_policy=POLICY_LATEST_PREFERRED,
        )

    def test_a_correction_filed_after_the_window_is_not_retrieved(self) -> None:
        route = self.router.route(self._windowed_plan("2023-02-10"))
        self.assertEqual(route.date_range, ("2023-01-10", "2023-02-10"))
        kept = [d.doc_id for d in self.router.filter_documents(self.documents, route)]
        self.assertEqual(kept, ["exchange_1", "exchange_2"])
        self.assertNotIn("exchange_3", kept)

    def test_widening_the_window_admits_the_later_correction(self) -> None:
        route = self.router.route(self._windowed_plan("2023-03-31"))
        kept = [d.doc_id for d in self.router.filter_documents(self.documents, route)]
        self.assertEqual(kept, ["exchange_1", "exchange_2", "exchange_3"])

    def test_the_reported_latest_is_the_graph_wide_one_not_an_as_of_date_one(self) -> None:
        """Documents a known P0-A limitation rather than asserting a feature.

        ``correction_summary`` answers "what is the final version of this
        report", which is graph-wide.  It is not an as-of-date snapshot, so a
        consumer that needs "latest as of the window" must derive it from the
        timeline and the candidates it actually retrieved.
        """

        route = self.router.route(self._windowed_plan("2023-02-10"))
        kept = self.router.filter_documents(self.documents, route)
        summary = self.router.correction_summary(kept, route)
        group = summary["groups"]["exchange_1"]

        self.assertEqual(group["latest_doc_id"], "exchange_3")
        self.assertNotIn("exchange_3", [d.doc_id for d in kept])
        # The full history is present, so the caller can pick the last entry
        # that is actually inside its own candidate set.
        retrieved = {d.doc_id for d in kept}
        in_window = [i for i in group["timeline"] if i["doc_id"] in retrieved]
        self.assertEqual(in_window[-1]["doc_id"], "exchange_2")


class CorrectionMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sql = (PROJECT_ROOT / "db" / "006_correction_graph.sql").read_text(
            encoding="utf-8"
        )

    def test_migration_is_additive_and_touches_no_frozen_table(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS correction_relations", self.sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS correction_group_members", self.sql)
        for statement in ("DROP TABLE", "DROP COLUMN", "TRUNCATE"):
            self.assertNotIn(statement, self.sql)
        for table in ("chunks", "disclosures", "sections", "disclosure_tables"):
            self.assertNotIn(f"ALTER TABLE {table}", self.sql)
        self.assertNotIn("chunk_embeddings", self.sql)

    def test_migration_enforces_the_graph_invariants(self) -> None:
        self.assertIn("REFERENCES disclosures(doc_id)", self.sql)
        self.assertIn("uq_correction_relations_edge", self.sql)
        self.assertIn("coalesce(target_doc_id, '')", self.sql)
        self.assertIn("uq_correction_group_members_latest", self.sql)
        self.assertIn("WHERE is_latest", self.sql)
        self.assertIn("correction_relations_no_self_reference", self.sql)
        self.assertIn("correction_group_members_no_self_parent", self.sql)
        self.assertIn("resolution_status IN ('resolved', 'ambiguous', 'unresolved')", self.sql)


class ResolverRegressionTests(unittest.TestCase):
    """Representative holding and periodic resolver behaviour is untouched."""

    def test_holding_event_resolver_still_reconstructs_core_fields(self) -> None:
        from app.reasoning.holding_event_resolver import resolve_holding_events
        from tests.test_holding_event_resolver import _evidence_set, _group, _item

        item = _item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2024년 03월 07일",
                "after_shares": "1,234,567",
            },
        )
        event = resolve_holding_events(_evidence_set([_group("g1", item)])).events[0]
        self.assertEqual(event.reporter, "국민연금기금")
        self.assertEqual(event.reference_date, "2024-03-07")
        self.assertEqual(event.after_shares.normalized, 1234567)

    def test_periodic_fact_resolver_still_merges_one_repeated_fact(self) -> None:
        from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
        from tests.test_periodic_fact_resolver import _evidence, _group, _item

        older = _item("p23:c1", "p23", rank=1, text="HUBO 로봇 사업.", year=2023)
        newer = _item("p24:c1", "p24", rank=2, text="HUBO 로봇 사업.", year=2024)
        resolution = resolve_periodic_facts(
            _evidence(
                [_group("g-hubo", older, newer, group_type="periodic_repeated_fact")],
                question="테스트회사 HUBO 로봇 사업 설명",
            )
        )
        self.assertEqual(len(resolution.facts), 1)
        self.assertEqual(
            sorted(source.chunk_id for source in resolution.facts[0].sources),
            ["p23:c1", "p24:c1"],
        )


if __name__ == "__main__":
    unittest.main()
