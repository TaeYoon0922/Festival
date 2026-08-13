import gzip
import json
import tempfile
import unittest
from pathlib import Path

from app.parsing.chunking import REQUIRED_CHUNK_FIELDS, build_chunks
from app.parsing.dart_xml import parse_dart_text
from app.parsing.full_pipeline import _process_source


DOCUMENT_METADATA = {
    "doc_id": "periodic_20250101000001",
    "corp_code": "00123456",
    "corp_name": "테스트전자",
    "stock_code": "123456",
    "doc_group": "periodic",
    "doc_subtype": "annual",
    "report_nm": "사업보고서 (2024.12)",
    "rcept_no": "20250101000001",
    "rcept_dt": "20250101",
    "is_correction": False,
    "base_year": 2024,
    "base_month": 12,
}


def build_from_xml(source: str, metadata: dict | None = None, **kwargs):
    document = dict(DOCUMENT_METADATA)
    if metadata:
        document.update(metadata)
    parsed = parse_dart_text(source, fallback_title=str(document["report_nm"]))
    chunks = build_chunks(
        str(document["doc_id"]),
        parsed,
        document_metadata=document,
        source_file="raw/test.xml",
        **kwargs,
    )
    return parsed, chunks


class StructuralChunkingTests(unittest.TestCase):
    def test_chunks_never_cross_section_boundaries(self) -> None:
        source = """
        <DOCUMENT><BODY>
          <SECTION-1><TITLE>첫 섹션</TITLE><P>첫 섹션의 고유 본문입니다.</P></SECTION-1>
          <SECTION-1><TITLE>둘째 섹션</TITLE><P>둘째 섹션의 고유 본문입니다.</P></SECTION-1>
        </BODY></DOCUMENT>
        """
        parsed, chunks = build_from_xml(source, min_chars=10, max_chars=1500)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            {chunk["section_id"] for chunk in chunks},
            {section.section_id for section in parsed.sections},
        )
        self.assertFalse(
            any(
                "첫 섹션의 고유" in chunk["content"]
                and "둘째 섹션의 고유" in chunk["content"]
                for chunk in chunks
            )
        )

    def test_long_paragraph_splits_only_at_sentence_boundaries(self) -> None:
        sentences = [f"문장 {index}의 의미를 온전히 보존합니다." for index in range(90)]
        source = (
            "<DOCUMENT><BODY><SECTION-1><TITLE>긴 본문</TITLE><P>"
            + " ".join(sentences)
            + "</P></SECTION-1></BODY></DOCUMENT>"
        )
        _, chunks = build_from_xml(
            source, min_chars=200, target_chars=400, max_chars=500, overlap=80
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["content"].endswith(".") for chunk in chunks))
        self.assertTrue(all("문장 " in chunk["content"] for chunk in chunks))

    def test_small_table_is_one_chunk(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>매출</TITLE>
          <TABLE><TR><TH>구분</TH><TH>금액</TH></TR>
          <TR><TD>제품</TD><TD>100</TD></TR>
          <TR><TD>서비스</TD><TD>50</TD></TR></TABLE>
        </SECTION-1></BODY></DOCUMENT>
        """
        _, chunks = build_from_xml(source)
        table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0]["column_headers"], ["구분", "금액"])
        self.assertIn("제품", table_chunks[0]["content"])
        self.assertIn("서비스", table_chunks[0]["content"])

    def test_large_table_repeats_headers_without_row_overlap(self) -> None:
        rows = "".join(
            f"<TR><TD>항목 {index}</TD><TD>{index * 10}</TD></TR>"
            for index in range(45)
        )
        source = (
            "<DOCUMENT><BODY><SECTION-1><TITLE>재무 표</TITLE>"
            "<TABLE><TR><TH>구분</TH><TH>2025</TH></TR>"
            f"{rows}</TABLE></SECTION-1></BODY></DOCUMENT>"
        )
        _, chunks = build_from_xml(source)
        table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
        self.assertEqual(len(table_chunks), 3)
        self.assertTrue(
            all(chunk["content"].startswith("| 구분 | 2025 |") for chunk in table_chunks)
        )
        self.assertEqual(
            [(chunk["row_start"], chunk["row_end"]) for chunk in table_chunks],
            [(1, 20), (21, 40), (41, 45)],
        )

    def test_rowspan_colspan_are_preserved_in_logical_headers_and_rows(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>손익</TITLE>
          <TABLE>
            <TR><TH rowspan="2">구분</TH><TH colspan="2">금액</TH></TR>
            <TR><TH>2025</TH><TH>2024</TH></TR>
            <TR><TD rowspan="2">매출</TD><TD>300</TD><TD>280</TD></TR>
            <TR><TD>310</TD><TD>290</TD></TR>
          </TABLE>
        </SECTION-1></BODY></DOCUMENT>
        """
        _, chunks = build_from_xml(source)
        table = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
        self.assertEqual(
            table["column_headers"],
            ["구분", "금액 / 2025", "금액 / 2024"],
        )
        self.assertEqual(table["table_rows"][0][0]["rowspan"], 2)
        self.assertIn("| 매출 | 310 | 290 |", table["content"])

    def test_oversized_table_row_is_never_split_mid_cell(self) -> None:
        long_value = "긴셀내용" * 600
        source = (
            "<DOCUMENT><BODY><SECTION-1><TITLE>주석</TITLE><TABLE>"
            f"<TR><TD>설명</TD><TD>{long_value}</TD></TR>"
            "</TABLE></SECTION-1></BODY></DOCUMENT>"
        )
        _, chunks = build_from_xml(source, max_chars=300)
        table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
        self.assertEqual(len(table_chunks), 1)
        self.assertIn(long_value, table_chunks[0]["content"])
        self.assertGreater(table_chunks[0]["char_count"], 300)

    def test_extreme_table_keeps_source_row_and_adds_traceable_projections(self) -> None:
        long_value = "극도로긴검색텍스트" * 1_500
        source = (
            "<DOCUMENT><BODY><SECTION-1><TITLE>연구개발</TITLE><TABLE>"
            "<TR><TH>항목</TH><TH>설명</TH></TR>"
            f"<TR><TD>과제</TD><TD>{long_value}</TD></TR>"
            "</TABLE></SECTION-1></BODY></DOCUMENT>"
        )
        parsed, chunks = build_from_xml(source)
        original = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
        projections = [
            chunk
            for chunk in chunks
            if chunk["chunk_type"] == "table_projection"
        ]
        self.assertIn(long_value, original["content"])
        self.assertGreater(len(projections), 1)
        self.assertTrue(all(chunk["source_table_id"] == "t0001" for chunk in projections))
        self.assertTrue(all(chunk["source_row_start"] == 1 for chunk in projections))
        self.assertTrue(all(chunk["source_row_end"] == 1 for chunk in projections))
        self.assertTrue(all(chunk["char_count"] <= 1_500 for chunk in projections))
        self.assertEqual(parsed.tables[0].rows[1][1].text, long_value)

    def test_extreme_projection_removes_duplicate_content_lines(self) -> None:
        repeated = "동일한 계약 설명 " * 500
        source = (
            "<DOCUMENT><BODY><SECTION-1><TITLE>계약</TITLE><TABLE>"
            "<TR><TH>구분</TH><TH>설명</TH><TH>설명</TH></TR>"
            f"<TR><TD>A</TD><TD>{repeated}</TD><TD>{repeated}</TD></TR>"
            "</TABLE></SECTION-1></BODY></DOCUMENT>"
        )
        _, chunks = build_from_xml(source)
        projections = [
            chunk
            for chunk in chunks
            if chunk.get("projection_type") == "extreme_table_row"
        ]
        self.assertTrue(projections)
        for projection in projections:
            lines = projection["content"].splitlines()
            self.assertEqual(len(lines), len(set(lines)))

    def test_context_only_short_text_merges_but_short_fact_survives(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>계약 현황</TITLE>
          <P>② 전기말</P>
          <P>계약에 관한 상세 설명이 이어집니다.</P>
          <P>계약금액: 500억원</P>
        </SECTION-1></BODY></DOCUMENT>
        """
        _, chunks = build_from_xml(source, min_chars=10)
        texts = [chunk for chunk in chunks if chunk["chunk_type"] == "text"]
        self.assertFalse(any(chunk["content"] == "② 전기말" for chunk in texts))
        merged = next(chunk for chunk in texts if "상세 설명" in chunk["content"])
        self.assertIn("② 전기말", merged["content"])
        self.assertIn("② 전기말", merged["context_labels"])
        self.assertTrue(any("계약금액: 500억원" in chunk["content"] for chunk in texts))

    def test_evidence_note_table_is_indexable(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>변동 사항</TITLE>
          <TABLE><TR><TD>※ 기준일 환율을 적용한 실질 증거입니다.</TD></TR></TABLE>
        </SECTION-1></BODY></DOCUMENT>
        """
        _, chunks = build_from_xml(source)
        table = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
        self.assertTrue(table["is_indexable"])
        self.assertEqual(table["former_exclusion_reason"], "basis_period_note")

    def test_holding_report_projection_preserves_source_traceability(self) -> None:
        source = """
        <DOCUMENT><BODY>
          <SECTION-1><TITLE>주식등의 대량보유상황보고서</TITLE>
            <TABLE>
              <TR><TD>보고사유</TD><TD>단순추가취득</TD></TR>
              <TR><TD>보유목적</TD><TD>단순투자</TD></TR>
            </TABLE>
          </SECTION-1>
          <SECTION-1><TITLE>3. 보유주식등의 수 및 보유비율</TITLE>
            <TABLE>
              <TR><TH rowspan="2"></TH><TH rowspan="2">보고서작성기준일</TH><TH colspan="2">보고자</TH><TH colspan="2">주식등</TH></TR>
              <TR><TH>본인 성명</TH><TH>특별관계자수</TH><TH>주식등의 수(주)</TH><TH>비율(%)</TH></TR>
              <TR><TD>직전보고서</TD><TD>2024년 01월 01일</TD><TD>국민연금공단</TD><TD>1</TD><TD>100</TD><TD>5.0</TD></TR>
              <TR><TD>이번보고서</TD><TD>2025년 01월 01일</TD><TD>국민연금공단</TD><TD>1</TD><TD>120</TD><TD>6.0</TD></TR>
              <TR><TD colspan="4">증 감</TD><TD>20</TD><TD>1.0</TD></TR>
            </TABLE>
          </SECTION-1>
        </BODY></DOCUMENT>
        """
        metadata = {"doc_group": "holding"}
        parsed, chunks = build_from_xml(source, metadata=metadata)
        projection = next(
            chunk
            for chunk in chunks
            if chunk.get("projection_type") == "holding_report"
        )
        self.assertIn("[보고자/보유자] 국민연금공단", projection["retrieval_text"])
        self.assertIn("[보유주식수] 120", projection["retrieval_text"])
        self.assertIn("[증감주식수] 20", projection["retrieval_text"])
        self.assertIn("[보유 목적] 단순투자", projection["retrieval_text"])
        self.assertIn("[변동 사유] 단순추가취득", projection["retrieval_text"])
        self.assertEqual(projection["source_table_id"], parsed.tables[1].table_id)
        self.assertEqual(
            projection["source_table_ids"],
            [parsed.tables[1].table_id, parsed.tables[0].table_id],
        )
        self.assertEqual(projection["source_row_start"], 2)
        self.assertEqual(projection["source_row_end"], 4)
        self.assertEqual(
            projection["projection_field_refs"]["보유 목적"],
            [{"table_id": parsed.tables[0].table_id, "row_start": 1, "row_end": 1}],
        )
        self.assertEqual(
            projection["projection_field_refs"]["변동 사유"],
            [{"table_id": parsed.tables[0].table_id, "row_start": 0, "row_end": 0}],
        )
        original = next(
            chunk
            for chunk in chunks
            if chunk["chunk_type"] == "table"
            and chunk["table_id"] == parsed.tables[1].table_id
        )
        self.assertIn("직전보고서", original["content"])
        self.assertIn("이번보고서", original["content"])

    def test_holding_placeholder_is_explicit_and_source_backed(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>정정 신고</TITLE>
          <TABLE>
            <TR><TH rowspan="2">성명(명칭)</TH><TH rowspan="2">변동일*</TH><TH colspan="3">변동 내역</TH></TR>
            <TR><TH>변동전</TH><TH>증감</TH><TH>변동후</TH></TR>
            <TR><TD colspan="5">정정 전과 동일</TD></TR>
          </TABLE>
        </SECTION-1></BODY></DOCUMENT>
        """
        _, chunks = build_from_xml(source, metadata={"doc_group": "holding"})
        projection = next(
            chunk
            for chunk in chunks
            if chunk.get("projection_type") == "holding_detail_row"
        )
        self.assertEqual(projection["projection_state"], "explicit_placeholder")
        self.assertIn("explicit_placeholder", projection["quality_flags"])
        self.assertEqual(
            projection["projection_field_refs"]["보고자/보유자"],
            projection["source_refs"],
        )

    def test_required_metadata_links_and_deterministic_ids(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>사업</TITLE>
          <P>첫 번째 의미 있는 설명입니다.</P>
          <TABLE><TR><TH>구분</TH><TH>금액</TH></TR><TR><TD>매출</TD><TD>100</TD></TR></TABLE>
          <P>두 번째 의미 있는 설명입니다.</P>
        </SECTION-1></BODY></DOCUMENT>
        """
        parsed, first = build_from_xml(source, min_chars=10)
        _, second = build_from_xml(source, min_chars=10)
        self.assertEqual(
            [chunk["chunk_id"] for chunk in first],
            [chunk["chunk_id"] for chunk in second],
        )
        for index, chunk in enumerate(first):
            self.assertTrue(all(field in chunk for field in REQUIRED_CHUNK_FIELDS))
            self.assertTrue(all(chunk[field] is not None for field in DOCUMENT_METADATA))
            self.assertEqual(chunk["doc_id"], DOCUMENT_METADATA["doc_id"])
            self.assertIn(chunk["section_id"], parsed.section_map())
            self.assertEqual(
                chunk["prev_chunk_id"], first[index - 1]["chunk_id"] if index else None
            )
            self.assertEqual(
                chunk["next_chunk_id"],
                first[index + 1]["chunk_id"] if index + 1 < len(first) else None,
            )
            if chunk["chunk_type"] == "table":
                self.assertEqual(chunk["table_id"], parsed.tables[0].table_id)

    def test_correction_metadata_is_not_interpreted_or_dropped(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>정정 내용</TITLE>
          <P>정정 전후의 원문 증거입니다.</P>
        </SECTION-1></BODY></DOCUMENT>
        """
        metadata = {
            "doc_id": "periodic_20250102000002",
            "rcept_no": "20250102000002",
            "rcept_dt": "20250102",
            "is_correction": True,
        }
        _, chunks = build_from_xml(source, metadata=metadata, min_chars=10)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk["is_correction"] is True for chunk in chunks))
        self.assertTrue(
            all(chunk["rcept_no"] == "20250102000002" for chunk in chunks)
        )

    def test_restart_reuses_one_output_without_duplicate_chunks(self) -> None:
        source = """
        <DOCUMENT><BODY><SECTION-1><TITLE>재시작</TITLE>
          <P>재시작 후에도 한 번만 존재해야 하는 본문입니다.</P>
        </SECTION-1></BODY></DOCUMENT>
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.xml"
            source_path.write_text(source, encoding="utf-8")
            output_path = root / "output.json.gz"
            row = dict(DOCUMENT_METADATA)
            row.update(
                {
                    "listed_name": "테스트전자",
                    "industry": "IT",
                    "sector": "전자",
                    "flr_nm": "테스트전자",
                    "file_path": "raw/test",
                    "file_format": "xml",
                    "n_files": 1,
                }
            )
            task = {
                "part_id": f"{row['doc_id']}:source",
                "row": row,
                "source_path": str(source_path),
                "source_relative": "raw/test/source.xml",
                "source_format": "xml",
                "is_primary": True,
                "output_path": str(output_path),
                "output_relative": "documents/test/output.json.gz",
                "target_chars": 1200,
                "min_chars": 700,
                "max_chars": 1500,
                "overlap": 120,
                "resume": True,
            }
            first = _process_source(task)
            second = _process_source(task)
            self.assertEqual(first["status"], "parsed")
            self.assertEqual(second["status"], "resumed")
            with gzip.open(output_path, "rt", encoding="utf-8") as saved:
                payload = json.load(saved)
            chunk_ids = [chunk["chunk_id"] for chunk in payload["chunks"]]
            self.assertEqual(len(chunk_ids), len(set(chunk_ids)))


if __name__ == "__main__":
    unittest.main()
