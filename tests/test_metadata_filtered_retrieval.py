import unittest

from app.parsing.metadata_filtered_retrieval import extract_metadata_filters


class MetadataFilterExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.companies = {
            "삼성전자": {"삼성전자"},
            "에스엠": {"에스엠"},
            "하이브": {"하이브"},
            "LS ELECTRIC": {"엘에스일렉트릭"},
            "엘에스일렉트릭": {"엘에스일렉트릭"},
            "한미반도체": {"한미반도체"},
        }

    def test_holding_reference_year_does_not_filter_filing_year(self) -> None:
        result = extract_metadata_filters(
            "에스엠 국민연금 2023년 보유주식 수와 비율", self.companies
        )
        self.assertEqual(result["companies"], ["에스엠"])
        self.assertEqual(result["doc_group"], "holding")
        self.assertFalse(result["year_filter_applied"])

    def test_clear_exchange_subtype_is_extracted(self) -> None:
        result = extract_metadata_filters(
            "LS ELECTRIC 신규 시설투자 종료일", self.companies
        )
        self.assertEqual(result["companies"], ["엘에스일렉트릭"])
        self.assertEqual(result["doc_group"], "exchange")
        self.assertEqual(result["doc_subtype"], "신규시설투자등")

    def test_facility_investment_with_cash_on_hand_is_not_holding(self) -> None:
        companies = {
            **self.companies,
            "고려아연": {"고려아연"},
        }
        queries = (
            "고려아연이 최근 공시한 신규시설투자 금액은 자기자본 대비 어느 정도 수준이며, "
            "현재 보유 중인 현금성 자산으로 자체 조달이 가능한가요?",
            "LS ELECTRIC 최근 공시한 시설투자 금액은 자기자본 대비 비율이며 "
            "현재 보유 중인 현금성 자산으로 자체 조달이 가능한가",
            "삼성전자 최근 공시 신규시설투자 금액과 자기자본 대비",
        )
        for query in queries:
            with self.subTest(query=query):
                result = extract_metadata_filters(query, companies)
                self.assertEqual(result["doc_group"], "exchange")
                self.assertEqual(result["doc_subtype"], "신규시설투자등")

    def test_current_share_holding_is_still_holding(self) -> None:
        result = extract_metadata_filters(
            "파마리서치 국민연금 2022년 12월 5일 현재 보유 비율",
            {**self.companies, "파마리서치": {"파마리서치"}},
        )
        self.assertEqual(result["doc_group"], "holding")
        self.assertIsNone(result["doc_subtype"])

    def test_cash_on_hand_alone_is_not_a_holding_disclosure(self) -> None:
        result = extract_metadata_filters(
            "삼성전자가 현재 보유 중인 현금성 자산", self.companies
        )
        self.assertNotEqual(result["doc_group"], "holding")
        self.assertIsNone(result["doc_subtype"])

    def test_all_explicit_company_mentions_are_or_candidates(self) -> None:
        result = extract_metadata_filters(
            "에스엠 하이브 보유주식 수와 비율", self.companies
        )
        self.assertEqual(result["companies"], ["에스엠", "하이브"])
        self.assertEqual(result["doc_group"], "holding")

    def test_ambiguous_merger_does_not_force_a_group(self) -> None:
        result = extract_metadata_filters(
            "삼성전자 흡수합병 합병기일", self.companies
        )
        self.assertIsNone(result["doc_group"])

    def test_holding_compound_language_is_extracted(self) -> None:
        result = extract_metadata_filters(
            "에스엠 풋옵션 행사 주식 취득일과 취득 수량", self.companies
        )
        self.assertEqual(result["doc_group"], "holding")

    def test_executive_ownership_report_title_is_holding(self) -> None:
        result = extract_metadata_filters(
            "기아의 가장 최근 임원ㆍ주요주주특정증권등소유상황보고서에서 "
            "특수관계자가 한 명 더 늘어난 이유는?",
            {**self.companies, "기아": {"기아"}},
        )
        self.assertEqual(result["companies"], ["기아"])
        self.assertEqual(result["doc_group"], "holding")
        self.assertEqual(result["doc_group_evidence"], "소유상황보고서")

    def test_clear_major_transaction_language_is_extracted(self) -> None:
        result = extract_metadata_filters(
            "삼성전자 합병 목적과 분할 대상", self.companies
        )
        self.assertEqual(result["doc_group"], "major")

    def test_clear_periodic_disclosure_language_is_extracted(self) -> None:
        result = extract_metadata_filters(
            "삼성전자 2023년 연결대상회사와 주요종속회사 수", self.companies
        )
        self.assertEqual(result["doc_group"], "periodic")

    def test_quarter_subtype_implies_periodic_group(self) -> None:
        result = extract_metadata_filters(
            "한미반도체 1분기 매출액", self.companies
        )
        self.assertEqual(result["doc_group"], "periodic")
        self.assertEqual(result["doc_subtype"], "quarter")


if __name__ == "__main__":
    unittest.main()
