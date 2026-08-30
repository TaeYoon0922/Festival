import unittest

from app.reasoning.holding_company_role_resolution import (
    BIDIRECTIONAL,
    INCOMPLETE_INDEX,
    NO_DIRECTION,
    RESOLVED,
    STALE_INDEX,
    HoldingCompanyRoleResolver,
)
from app.reasoning.holding_report_index import HoldingReportIndex, HoldingReportRecord
from app.reasoning.holding_reporter import canonical_reporter_key


A = "가상발행사"
B = "가상투자사"
C = "가상투자사정밀"
A_CODE = "00000001"
B_CODE = "00000002"
C_CODE = "00000003"
IDENTITY = {"corpus_manifest_sha256": "fixture"}


def record(
    issuer: str,
    reporter: str,
    suffix: str,
    *,
    raw_reporter: str | None = None,
) -> HoldingReportRecord:
    return HoldingReportRecord(
        issuer_corp_code=issuer,
        reporter_key=canonical_reporter_key(reporter),
        raw_reporter=raw_reporter or reporter,
        doc_id=f"holding_{suffix}",
        projection_chunk_id=f"holding_{suffix}:projection",
        reference_date="20240101",
        after_shares="100",
        after_ratio="1.00",
    )


def index_of(
    *records: HoldingReportRecord,
    complete: bool = True,
    identity=None,
) -> HoldingReportIndex:
    return HoldingReportIndex(
        records,
        complete=complete,
        correction_finality_available=True,
        identity=IDENTITY if identity is None else identity,
    )


def resolver(*records: HoldingReportRecord) -> HoldingCompanyRoleResolver:
    return HoldingCompanyRoleResolver(
        index_of(*records), active_corpus_identity=IDENTITY
    )


class DirectionResolutionTests(unittest.TestCase):
    def test_unique_first_direction_resolves(self) -> None:
        result = resolver(record(A_CODE, B, "a_b")).resolve(
            A, A_CODE, B, B_CODE
        )

        self.assertEqual(result.status, RESOLVED)
        self.assertTrue(result.resolved)
        self.assertEqual(result.issuer, A)
        self.assertEqual(result.issuer_corp_code, A_CODE)
        self.assertEqual(result.reporter, B)
        self.assertEqual(result.reporter_key, canonical_reporter_key(B))

    def test_unique_reverse_direction_resolves(self) -> None:
        result = resolver(record(B_CODE, A, "b_a")).resolve(
            A, A_CODE, B, B_CODE
        )

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.issuer, B)
        self.assertEqual(result.issuer_corp_code, B_CODE)
        self.assertEqual(result.reporter, A)

    def test_reversing_input_order_keeps_the_same_roles(self) -> None:
        value = resolver(record(A_CODE, B, "a_b"))

        forward = value.resolve(A, A_CODE, B, B_CODE)
        reverse = value.resolve(B, B_CODE, A, A_CODE)

        self.assertEqual(forward.issuer, reverse.issuer)
        self.assertEqual(forward.issuer_corp_code, reverse.issuer_corp_code)
        self.assertEqual(forward.reporter, reverse.reporter)
        self.assertEqual(forward.reporter_key, reverse.reporter_key)

    def test_one_supporting_report_is_sufficient(self) -> None:
        result = resolver(record(A_CODE, B, "only")).resolve(
            A, A_CODE, B, B_CODE
        )

        self.assertTrue(result.resolved)
        self.assertEqual(result.direction_1_report_count, 1)
        self.assertEqual(result.direction_2_report_count, 0)

    def test_multiple_reports_in_one_direction_are_not_ambiguous(self) -> None:
        result = resolver(
            record(A_CODE, B, "old"),
            record(A_CODE, B, "new"),
        ).resolve(A, A_CODE, B, B_CODE)

        self.assertTrue(result.resolved)
        self.assertEqual(result.direction_1_report_count, 2)
        self.assertEqual(result.direction_2_report_count, 0)

    def test_zero_direction_fails_closed(self) -> None:
        result = resolver().resolve(A, A_CODE, B, B_CODE)

        self.assertEqual(result.status, NO_DIRECTION)
        self.assertFalse(result.resolved)

    def test_bidirectional_relation_fails_closed(self) -> None:
        result = resolver(
            record(A_CODE, B, "a_b"),
            record(B_CODE, A, "b_a"),
        ).resolve(A, A_CODE, B, B_CODE)

        self.assertEqual(result.status, BIDIRECTIONAL)
        self.assertFalse(result.resolved)
        self.assertEqual(result.direction_1_report_count, 1)
        self.assertEqual(result.direction_2_report_count, 1)


class ReporterIdentityTests(unittest.TestCase):
    def test_legal_form_normalization_is_reused(self) -> None:
        result = resolver(
            record(A_CODE, "(주)가상투자사", "legal_form")
        ).resolve(A, A_CODE, "가상투자사(주)", B_CODE)

        self.assertTrue(result.resolved)
        self.assertEqual(result.reporter_key, canonical_reporter_key(B))

    def test_exact_canonical_reporter_key_resolves(self) -> None:
        result = resolver(record(A_CODE, B, "exact")).resolve(
            A, A_CODE, B, B_CODE
        )

        self.assertTrue(result.resolved)
        self.assertEqual(result.reporter_key, canonical_reporter_key(B))

    def test_partial_reporter_substring_does_not_resolve(self) -> None:
        result = resolver(record(A_CODE, C, "longer")).resolve(
            A, A_CODE, B, B_CODE
        )

        self.assertEqual(result.status, NO_DIRECTION)
        self.assertFalse(result.resolved)

    def test_longer_company_name_collision_does_not_resolve_in_reverse(self) -> None:
        result = resolver(record(A_CODE, B, "shorter")).resolve(
            A, A_CODE, C, C_CODE
        )

        self.assertEqual(result.status, NO_DIRECTION)
        self.assertFalse(result.resolved)


class IndexAvailabilityTests(unittest.TestCase):
    def test_incomplete_index_fails_closed(self) -> None:
        value = HoldingCompanyRoleResolver(
            index_of(record(A_CODE, B, "one"), complete=False),
            active_corpus_identity=IDENTITY,
        )

        self.assertEqual(
            value.resolve(A, A_CODE, B, B_CODE).status,
            INCOMPLETE_INDEX,
        )

    def test_stale_index_fails_closed(self) -> None:
        value = HoldingCompanyRoleResolver(
            index_of(record(A_CODE, B, "one")),
            active_corpus_identity={"corpus_manifest_sha256": "different"},
        )

        self.assertEqual(
            value.resolve(A, A_CODE, B, B_CODE).status,
            STALE_INDEX,
        )


if __name__ == "__main__":
    unittest.main()
