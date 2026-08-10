import unittest

from app.parsing import build_chunks, parse_dart_text


class DartParserTests(unittest.TestCase):
    def test_structured_xml_extracts_hierarchy_text_and_table(self) -> None:
        source = """
        <DOCUMENT><BODY>
          <SECTION-1><TITLE>I. 사업의 내용</TITLE>
            <P>첫 번째 사업 설명입니다.</P>
            <SECTION-2><TITLE>1. 매출</TITLE>
              <P>매출 현황입니다.</P>
              <TABLE border="1"><TR><TH>구분</TH><TH>금액</TH></TR>
                <TR><TD rowspan="2">제품</TD><TD>100</TD></TR></TABLE>
            </SECTION-2>
          </SECTION-1>
        </BODY></DOCUMENT>
        """
        parsed = parse_dart_text(source, fallback_title="테스트 보고서")

        self.assertEqual([section.title for section in parsed.sections], ["I. 사업의 내용", "1. 매출"])
        self.assertEqual(parsed.sections[1].path, ["I. 사업의 내용", "1. 매출"])
        self.assertEqual(parsed.sections[1].text, "매출 현황입니다.")
        self.assertEqual(len(parsed.tables), 1)
        self.assertEqual(parsed.tables[0].rows[0][0].text, "구분")
        self.assertTrue(parsed.tables[0].rows[0][0].is_header)
        self.assertEqual(parsed.tables[0].rows[1][0].rowspan, 2)

        chunks = build_chunks("test_doc", parsed, max_chars=200, overlap=20)
        self.assertTrue(any(chunk["kind"] == "text" for chunk in chunks))
        self.assertTrue(any(chunk["kind"] == "table" for chunk in chunks))
        second_section_chunks = [
            chunk for chunk in chunks if chunk["section_id"] == parsed.sections[1].section_id
        ]
        self.assertEqual(
            [chunk["kind"] for chunk in second_section_chunks], ["text", "table"]
        )

    def test_exchange_html_uses_document_title_as_synthetic_section(self) -> None:
        source = """
        <html><head><title>단일판매 공급계약</title></head><body>
          <div><span>계약 공시</span></div>
          <table><tr><td>계약금액</td><td>500억원</td></tr></table>
        </body></html>
        """
        parsed = parse_dart_text(source)

        self.assertEqual(parsed.document_title, "단일판매 공급계약")
        self.assertEqual(parsed.sections[0].title, "단일판매 공급계약")
        self.assertEqual(parsed.tables[0].rows[0][1].text, "500억원")


if __name__ == "__main__":
    unittest.main()
