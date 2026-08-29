import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning import holding_report_relative as report_relative
from app.reasoning.holding_report_index import (
    AMBIGUOUS,
    ARTIFACT_SCHEMA_VERSION,
    CHANGE_UNAVAILABLE,
    CORRECTION_AMBIGUOUS,
    CURRENT_UNAVAILABLE,
    INCOMPLETE_CORPUS,
    NO_INDEX,
    NO_MATCH,
    PREVIOUS_INCONSISTENT,
    PREVIOUS_UNAVAILABLE,
    PROJECTION_RESOLVED,
    RESOLVED,
    ROLE_CHANGE,
    ROLE_CURRENT,
    ROLE_PREVIOUS,
    SELECTOR_EXACT_RECEIPT_DATE,
    SELECTOR_EXACT_REFERENCE_DATE,
    SELECTOR_LATEST,
    SELECTOR_SELECTED_CONTEXT,
    STALE_INDEX,
    UNSUPPORTED_ROLE,
    UNSUPPORTED_SELECTOR,
    HoldingReportIndex,
    HoldingReportRecord,
    execute_report_relative,
    load_index,
    project_role,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/corpus/holding_report_index.json"

ISSUER = "00000001"
OTHER_ISSUER = "00000002"


def record(**over) -> HoldingReportRecord:
    """A report with plausible values; every test overrides what it is about."""

    base = dict(
        issuer_corp_code=ISSUER,
        reporter_key="holder",
        raw_reporter="Holder",
        doc_id="holding_1",
        projection_chunk_id="holding_1:ch_1",
        reference_date="20240101",
        receipt_date="20240105",
        previous_date="20230101",
        before_shares="100",
        before_ratio="1.00",
        change_shares="50",
        change_ratio="0.50",
        change_direction="increase",
        after_shares="150",
        after_ratio="1.50",
    )
    base.update(over)
    return HoldingReportRecord(**base)


def index_of(*records, **over) -> HoldingReportIndex:
    settings = dict(complete=True, correction_finality_available=True,
                    identity={"corpus_manifest_sha256": "abc"})
    settings.update(over)
    return HoldingReportIndex(records, **settings)


class LatestSelectorTests(unittest.TestCase):
    """"Latest" is the newest report by the date it is *about*."""

    def test_latest_is_ordered_by_reference_date_not_receipt_date(self) -> None:
        """The ordering that 62.5% of this corpus would otherwise get wrong.

        The report about the newer holding state arrived *first*; the older
        state was filed later.  Ordering by arrival returns the stale position.
        """

        older_state_filed_later = record(
            doc_id="holding_late", projection_chunk_id="holding_late:ch",
            reference_date="20240101", receipt_date="20240901", after_shares="100")
        newer_state_filed_earlier = record(
            doc_id="holding_early", projection_chunk_id="holding_early:ch",
            reference_date="20240601", receipt_date="20240605", after_shares="900")

        result = index_of(older_state_filed_later, newer_state_filed_earlier)\
            .select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.selected.doc_id, "holding_early")
        self.assertEqual(result.selected.reference_date, "20240601")

    def test_latest_is_not_ordered_by_document_id(self) -> None:
        zzz_but_older = record(doc_id="holding_zzz",
                               projection_chunk_id="holding_zzz:ch",
                               reference_date="20200101")
        aaa_but_newer = record(doc_id="holding_aaa",
                               projection_chunk_id="holding_aaa:ch",
                               reference_date="20250101")

        result = index_of(zzz_but_older, aaa_but_newer)\
            .select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.selected.doc_id, "holding_aaa")

    def test_another_holders_newer_report_never_wins(self) -> None:
        """The whole point of keying on the holder.

        A newer filing about the same issuer by a different holder describes a
        different position, and answering with it would attribute one holder's
        stake to another.
        """

        ours = record(reporter_key="holder", reference_date="20240101")
        theirs = record(reporter_key="other", raw_reporter="Other",
                        doc_id="holding_2", projection_chunk_id="holding_2:ch",
                        reference_date="20250101")

        result = index_of(ours, theirs).select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.selected.reference_date, "20240101")
        self.assertEqual(result.selected.reporter_key, "holder")

    def test_another_issuers_report_never_wins(self) -> None:
        ours = record(issuer_corp_code=ISSUER, reference_date="20240101")
        theirs = record(issuer_corp_code=OTHER_ISSUER, doc_id="holding_2",
                        projection_chunk_id="holding_2:ch",
                        reference_date="20250101")

        result = index_of(ours, theirs).select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.selected.reference_date, "20240101")

    def test_a_tied_latest_date_declines(self) -> None:
        """Two final reports about the same day, and nothing to choose between.

        Receipt date, receipt number and document id could all order these, and
        each would be inventing a reason to prefer one filing's numbers.
        """

        one = record(doc_id="holding_a", projection_chunk_id="holding_a:ch",
                     reference_date="20240601", receipt_date="20240602")
        two = record(doc_id="holding_b", projection_chunk_id="holding_b:ch",
                     reference_date="20240601", receipt_date="20240603")

        result = index_of(one, two).select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, AMBIGUOUS)
        self.assertIsNone(result.selected)
        self.assertEqual(sorted(result.detail["doc_ids"]),
                         ["holding_a", "holding_b"])


class CorrectionSafetyTests(unittest.TestCase):
    """An unproven correction is not an absent correction."""

    def test_a_correction_declines_when_finality_is_unprovable(self) -> None:
        plain = record(reference_date="20240101")
        corrected = record(doc_id="holding_2", projection_chunk_id="holding_2:ch",
                           reference_date="20240201", is_correction=True)

        result = index_of(plain, corrected, correction_finality_available=False)\
            .select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)
        self.assertIsNone(result.selected)

    def test_the_correction_gate_runs_before_any_selector(self) -> None:
        """Ordering first and collapsing later can select a superseded member."""

        plain = record(reference_date="20240101")
        corrected = record(doc_id="holding_2", projection_chunk_id="holding_2:ch",
                           reference_date="20240201", is_correction=True)
        index = index_of(plain, corrected, correction_finality_available=False)

        for selector, kwargs in (
            (SELECTOR_LATEST, {}),
            (SELECTOR_EXACT_REFERENCE_DATE, {"reference_date": "20240101"}),
            (SELECTOR_EXACT_RECEIPT_DATE, {"receipt_date": "20240105"}),
        ):
            with self.subTest(selector=selector):
                result = index.select_report(ISSUER, "Holder", selector, **kwargs)
                self.assertEqual(result.status, CORRECTION_AMBIGUOUS)

    def test_a_timeline_without_corrections_is_unaffected(self) -> None:
        result = index_of(record(), correction_finality_available=False)\
            .select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, RESOLVED)


class ExactDateSelectorTests(unittest.TestCase):
    def test_reference_and_receipt_are_different_axes(self) -> None:
        """A filing's arrival day is not the day its holdings are stated for."""

        only = record(reference_date="20240601", receipt_date="20240901")
        index = index_of(only)

        by_reference = index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20240601")
        crossed = index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_RECEIPT_DATE,
            receipt_date="20240601")
        by_receipt = index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_RECEIPT_DATE,
            receipt_date="20240901")

        self.assertEqual(by_reference.status, RESOLVED)
        self.assertEqual(crossed.status, NO_MATCH)
        self.assertEqual(by_receipt.status, RESOLVED)

    def test_an_exact_date_matching_nothing_declines(self) -> None:
        result = index_of(record()).select_report(
            ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="19000101")
        self.assertEqual(result.status, NO_MATCH)

    def test_a_tied_exact_date_declines(self) -> None:
        one = record(doc_id="holding_a", projection_chunk_id="holding_a:ch")
        two = record(doc_id="holding_b", projection_chunk_id="holding_b:ch")

        result = index_of(one, two).select_report(
            ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20240101")

        self.assertEqual(result.status, AMBIGUOUS)

    def test_an_exact_date_is_stable_when_a_newer_report_arrives(self) -> None:
        before = index_of(record(reference_date="20240314"))
        after = index_of(
            record(reference_date="20240314"),
            record(doc_id="holding_new", projection_chunk_id="holding_new:ch",
                   reference_date="20250605"))

        for index in (before, after):
            result = index.select_report(
                ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
                reference_date="20240314")
            self.assertEqual(result.selected.reference_date, "20240314")

    def test_latest_moves_when_a_newer_report_arrives(self) -> None:
        snapshot_a = index_of(record(reference_date="20240604"))
        snapshot_b = index_of(
            record(reference_date="20240604"),
            record(doc_id="holding_new", projection_chunk_id="holding_new:ch",
                   reference_date="20250605"))

        self.assertEqual(
            snapshot_a.select_report(ISSUER, "Holder", SELECTOR_LATEST)
            .selected.reference_date, "20240604")
        self.assertEqual(
            snapshot_b.select_report(ISSUER, "Holder", SELECTOR_LATEST)
            .selected.reference_date, "20250605")


class PreviousStateTests(unittest.TestCase):
    """"Previous" is a field of the selected report, never another report."""

    def test_previous_values_come_from_the_selected_report_itself(self) -> None:
        older = record(doc_id="holding_old", projection_chunk_id="holding_old:ch",
                       reference_date="20230101", after_shares="777")
        newest = record(doc_id="holding_new", projection_chunk_id="holding_new:ch",
                        reference_date="20240101", before_shares="100",
                        before_ratio="1.00")

        selected = index_of(older, newest)\
            .select_report(ISSUER, "Holder", SELECTOR_LATEST).selected

        self.assertEqual(selected.before_shares, "100")
        self.assertNotEqual(selected.before_shares, older.after_shares)

    def test_a_first_report_has_no_previous_state(self) -> None:
        first = record(previous_date=None, before_shares=None, before_ratio=None)

        self.assertFalse(first.has_previous_state)
        self.assertIsNone(first.before_shares)

    def test_a_missing_previous_value_is_never_zero(self) -> None:
        for placeholder in ("-", "", "—", "…"):
            with self.subTest(placeholder=placeholder):
                row = record().to_dict()
                row.update(previous_date=placeholder, before_shares=placeholder,
                           before_ratio=placeholder)
                parsed = HoldingReportRecord.from_dict(row)
                self.assertIsNone(parsed.before_shares)
                self.assertNotEqual(parsed.before_shares, "0")
                self.assertFalse(parsed.has_previous_state)

    def test_a_previous_state_missing_its_date_is_inconsistent(self) -> None:
        """Observed in the corpus: no previous report named, yet numbers present.

        Reporting those as a previous state would name a report the filing does
        not identify.
        """

        odd = record(previous_date=None, before_shares="0", before_ratio="0.00")

        self.assertTrue(odd.previous_state_is_inconsistent)
        self.assertFalse(odd.has_previous_state)


class ReporterScopeTests(unittest.TestCase):
    def test_the_frozen_canonical_key_is_reused(self) -> None:
        stored = record(reporter_key="하이브", raw_reporter="(주)하이브")
        index = index_of(stored)

        self.assertEqual(len(index.enumerate_reports(ISSUER, "하이브")), 1)
        self.assertEqual(len(index.enumerate_reports(ISSUER, "(주)하이브")), 1)
        self.assertEqual(len(index.enumerate_reports(ISSUER, "주식회사 하이브")), 1)

    def test_a_holder_is_never_matched_by_containment(self) -> None:
        stored = record(reporter_key="영풍", raw_reporter="영풍")

        self.assertEqual(
            len(index_of(stored).enumerate_reports(ISSUER, "영풍정밀")), 0)

    def test_an_empty_holder_indexes_and_matches_nothing(self) -> None:
        index = index_of(record(reporter_key="", raw_reporter=""))

        self.assertEqual(index.record_count, 0)
        self.assertEqual(index.enumerate_reports(ISSUER, ""), ())
        self.assertEqual(
            index_of(record()).select_report(ISSUER, "-", SELECTOR_LATEST).status,
            NO_MATCH)


class IndexIntegrityTests(unittest.TestCase):
    def test_a_stale_index_refuses_to_answer(self) -> None:
        index = index_of(record())

        result = index.select_report(
            ISSUER, "Holder", SELECTOR_LATEST,
            active_corpus_identity={"corpus_manifest_sha256": "a-different-corpus"})

        self.assertEqual(result.status, STALE_INDEX)
        self.assertIsNone(result.selected)

    def test_a_matching_identity_is_usable(self) -> None:
        result = index_of(record()).select_report(
            ISSUER, "Holder", SELECTOR_LATEST,
            active_corpus_identity={"corpus_manifest_sha256": "abc"})

        self.assertEqual(result.status, RESOLVED)

    def test_an_incomplete_source_cannot_claim_latest(self) -> None:
        """The report a partial index never saw is the one that changes latest."""

        result = index_of(record(), complete=False)\
            .select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, INCOMPLETE_CORPUS)

    def test_a_duplicate_record_is_not_counted_twice(self) -> None:
        duplicated = record()
        index = index_of(duplicated, record())

        self.assertEqual(index.record_count, 1)
        self.assertEqual(len(index.duplicate_chunk_ids), 1)

    def test_a_selector_naming_no_report_is_refused(self) -> None:
        """``selected_context`` points at a report the question never named."""

        result = index_of(record()).select_report(
            ISSUER, "Holder", "selected_context")

        self.assertEqual(result.status, UNSUPPORTED_SELECTOR)
        self.assertIsNone(result.selected)

    def test_a_malformed_artifact_yields_no_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (
                ("not_json.json", "{{{"),
                ("no_records.json", json.dumps({"identity": {}})),
                ("wrong_version.json", json.dumps(
                    {"identity": {"artifact_schema_version": "0.0"},
                     "records": []})),
                ("bad_record.json", json.dumps(
                    {"identity": {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION},
                     "records": [{"doc_id": "only"}]})),
            ):
                with self.subTest(name=name):
                    path = Path(tmp) / name
                    path.write_text(text, encoding="utf-8")
                    self.assertIsNone(load_index(path))

    def test_a_missing_artifact_yields_no_index(self) -> None:
        self.assertIsNone(load_index(Path("does/not/exist.json")))


@unittest.skipUnless(ARTIFACT.is_file(), "generated index not present")
class GeneratedArtifactTests(unittest.TestCase):
    """The artifact actually shipped, over the whole holding corpus."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = load_index(ARTIFACT)

    def test_the_artifact_loads_and_claims_the_whole_holding_corpus(self) -> None:
        identity = self.index.identity

        self.assertTrue(self.index.complete)
        self.assertEqual(identity["artifact_schema_version"],
                         ARTIFACT_SCHEMA_VERSION)
        self.assertTrue(identity["corpus_manifest_sha256"])
        self.assertEqual(identity["source_holding_disclosure_count"], 1083)

    def test_correction_finality_is_not_claimed(self) -> None:
        """No frozen source can prove it for this corpus, so it is not claimed."""

        self.assertFalse(self.index.correction_finality_available)

    def test_every_pair_either_resolves_or_declines_explicitly(self) -> None:
        allowed = {RESOLVED, AMBIGUOUS, CORRECTION_AMBIGUOUS, NO_MATCH}
        for (issuer, key) in self.index._by_pair:
            with self.subTest(pair=(issuer, key)):
                result = self.index.select_report(issuer, key, SELECTOR_LATEST)
                self.assertIn(result.status, allowed)
                if result.status != RESOLVED:
                    self.assertIsNone(result.selected)

    def test_correction_bearing_timelines_decline(self) -> None:
        declined = 0
        for (issuer, key), records in self.index._by_pair.items():
            if any(r.is_correction for r in records):
                result = self.index.select_report(issuer, key, SELECTOR_LATEST)
                self.assertEqual(result.status, CORRECTION_AMBIGUOUS)
                declined += 1
        self.assertGreater(declined, 0)

    def test_a_resolved_latest_is_the_maximum_reference_date(self) -> None:
        for (issuer, key) in self.index._by_pair:
            result = self.index.select_report(issuer, key, SELECTOR_LATEST)
            if not result.resolved:
                continue
            with self.subTest(pair=(issuer, key)):
                newest = max(r.reference_date
                             for r in self.index.enumerate_reports(issuer, key))
                self.assertEqual(result.selected.reference_date, newest)

    def test_the_index_never_stores_an_empty_holder(self) -> None:
        for (issuer, key), records in self.index._by_pair.items():
            self.assertTrue(key)
            for record_ in records:
                self.assertTrue(record_.reporter_key)


class GeneratorCompletenessGateTests(unittest.TestCase):
    """The generator refuses to write an index it cannot call complete.

    An index that quietly ships a partial corpus is the failure this whole
    target exists to prevent: "latest" would be the latest of whatever happened
    to be ingested.
    """

    def _generator(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_b2_generator", ROOT / "scripts/build_holding_report_index.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_an_incomplete_source_is_refused_and_writes_nothing(self) -> None:
        generator = self._generator()
        original = generator.build
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "index.json"
            generator.build = lambda: ([], {
                "manifest_holding_documents": 1083,
                "processed_holding_documents": 900,
                "manifest_matches_processed": False,
                "documents_represented_in_index": 900,
                "documents_without_report_projection": ["holding_missing"],
                "projections_without_reporter": [],
                "projections_without_reference_date": [],
                "counters": {},
                "complete": False,
                "schema_versions": [], "chunking_versions": [],
                "projection_provenance_revisions": [],
            })
            try:
                import sys as _sys
                argv = _sys.argv
                _sys.argv = ["build", "--out", str(out)]
                try:
                    # The generator reports coverage on stdout; the test asserts
                    # on its exit code and the file it did or did not write.
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = generator.main()
                finally:
                    _sys.argv = argv
            finally:
                generator.build = original

            self.assertNotEqual(code, 0)
            self.assertFalse(out.exists())

    def test_a_complete_source_writes_an_index_that_loads(self) -> None:
        generator = self._generator()
        original = generator.build
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "index.json"
            generator.build = lambda: ([record().to_dict()], {
                "manifest_holding_documents": 1,
                "processed_holding_documents": 1,
                "manifest_matches_processed": True,
                "documents_represented_in_index": 1,
                "documents_without_report_projection": [],
                "projections_without_reporter": [],
                "projections_without_reference_date": [],
                "counters": {"projections_total": 1},
                "complete": True,
                "schema_versions": ["2.0"], "chunking_versions": ["2.1"],
                "projection_provenance_revisions": ["v2.1-release-1"],
            })
            try:
                import sys as _sys
                argv = _sys.argv
                _sys.argv = ["build", "--out", str(out)]
                try:
                    # The generator reports coverage on stdout; the test asserts
                    # on its exit code and the file it did or did not write.
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = generator.main()
                finally:
                    _sys.argv = argv
            finally:
                generator.build = original

            self.assertEqual(code, 0)
            loaded = load_index(out)
            self.assertIsNotNone(loaded)
            self.assertTrue(loaded.complete)
            self.assertFalse(loaded.correction_finality_available)


class ProjectionRoleTests(unittest.TestCase):
    """Which fields of the selected report, read from that report alone."""

    def test_current_reads_the_reports_own_holding(self) -> None:
        projected = project_role(record(after_shares="150", after_ratio="1.50"),
                                 ROLE_CURRENT)

        self.assertEqual(projected.status, PROJECTION_RESOLVED)
        self.assertEqual(projected.values["shares"], "150")
        self.assertEqual(projected.values["ratio"], "1.50")

    def test_previous_reads_the_reports_own_previous_state(self) -> None:
        """Not the neighbouring filing, whose current state is a different number."""

        projected = project_role(
            record(previous_date="20230101", before_shares="100",
                   before_ratio="1.00"),
            ROLE_PREVIOUS)

        self.assertEqual(projected.status, PROJECTION_RESOLVED)
        self.assertEqual(projected.values["reference_date"], "20230101")
        self.assertEqual(projected.values["shares"], "100")

    def test_change_reads_the_reports_own_change(self) -> None:
        projected = project_role(
            record(change_shares="50", change_ratio="0.50",
                   change_direction="increase"),
            ROLE_CHANGE)

        self.assertEqual(projected.status, PROJECTION_RESOLVED)
        self.assertEqual(projected.values["shares"], "50")
        self.assertEqual(projected.values["direction"], "increase")

    def test_a_first_report_has_no_previous_role_to_project(self) -> None:
        projected = project_role(
            record(previous_date=None, before_shares=None, before_ratio=None),
            ROLE_PREVIOUS)

        self.assertEqual(projected.status, PREVIOUS_UNAVAILABLE)
        self.assertEqual(projected.values, {})

    def test_a_missing_previous_value_never_becomes_zero(self) -> None:
        projected = project_role(
            record(previous_date=None, before_shares=None, before_ratio=None),
            ROLE_PREVIOUS)

        self.assertNotIn("shares", projected.values)
        self.assertNotEqual(projected.values.get("shares"), "0")
        self.assertNotEqual(projected.values.get("shares"), 0)

    def test_previous_numbers_without_a_previous_report_are_refused(self) -> None:
        """The corpus anomaly: before values present, no previous report named.

        Returning "0 shares" here would state a previous holding on behalf of a
        report the filing never identifies.
        """

        projected = project_role(
            record(previous_date=None, before_shares="0", before_ratio="0.00"),
            ROLE_PREVIOUS)

        self.assertEqual(projected.status, PREVIOUS_INCONSISTENT)
        self.assertEqual(projected.values, {})

    def test_an_unstated_change_direction_is_not_inferred(self) -> None:
        projected = project_role(
            record(change_shares="0", change_ratio="0.00", change_direction=None),
            ROLE_CHANGE)

        self.assertEqual(projected.status, PROJECTION_RESOLVED)
        self.assertIsNone(projected.values["direction"])

    def test_a_report_stating_no_holding_projects_nothing(self) -> None:
        projected = project_role(
            record(after_shares=None, after_ratio=None), ROLE_CURRENT)

        self.assertEqual(projected.status, CURRENT_UNAVAILABLE)

    def test_an_unknown_role_projects_nothing(self) -> None:
        self.assertEqual(project_role(record(), "invented").status, UNSUPPORTED_ROLE)


class ExecutionReadinessTests(unittest.TestCase):
    """Readiness is decided against a corpus, not when the question is read."""

    @staticmethod
    def _intent(selector, role=ROLE_CURRENT):
        return {"selector": selector, "projection_role": role}

    def test_latest_is_executable_only_against_a_healthy_index(self) -> None:
        intent = self._intent(SELECTOR_LATEST)
        healthy = execute_report_relative(
            intent, index=index_of(record()), issuer_corp_code=ISSUER,
            reporter="Holder")

        self.assertTrue(healthy.executable)
        self.assertEqual(healthy.record.reference_date, "20240101")

        for index, expected in (
            (None, NO_INDEX),
            (index_of(record(), complete=False), INCOMPLETE_CORPUS),
            (index_of(record(reference_date="20240101"),
                      record(doc_id="holding_b",
                             projection_chunk_id="holding_b:ch",
                             reference_date="20240101")), AMBIGUOUS),
            (index_of(record(is_correction=True),
                      correction_finality_available=False), CORRECTION_AMBIGUOUS),
        ):
            with self.subTest(expected=expected):
                result = execute_report_relative(
                    intent, index=index, issuer_corp_code=ISSUER,
                    reporter="Holder")
                self.assertEqual(result.status, expected)
                self.assertFalse(result.executable)
                self.assertIsNone(result.record)

    def test_a_stale_index_makes_nothing_executable(self) -> None:
        result = execute_report_relative(
            self._intent(SELECTOR_LATEST), index=index_of(record()),
            issuer_corp_code=ISSUER, reporter="Holder",
            active_corpus_identity={"corpus_manifest_sha256": "moved-on"})

        self.assertEqual(result.status, STALE_INDEX)
        self.assertFalse(result.executable)

    def test_an_unknown_issuer_or_holder_executes_nothing(self) -> None:
        index = index_of(record())

        for issuer, reporter in ((OTHER_ISSUER, "Holder"), (ISSUER, "Nobody"),
                                 (ISSUER, ""), ("", "Holder")):
            with self.subTest(issuer=issuer, reporter=reporter):
                result = execute_report_relative(
                    self._intent(SELECTOR_LATEST), index=index,
                    issuer_corp_code=issuer, reporter=reporter)
                self.assertEqual(result.status, NO_MATCH)
                self.assertFalse(result.executable)

    def test_the_parsed_intent_is_read_not_widened(self) -> None:
        """A deictic selector stays unexecutable even though a latest exists."""

        index = index_of(record(reference_date="20240101"),
                         record(doc_id="holding_2",
                                projection_chunk_id="holding_2:ch",
                                reference_date="20250101"))

        result = execute_report_relative(
            self._intent(SELECTOR_SELECTED_CONTEXT), index=index,
            issuer_corp_code=ISSUER, reporter="Holder")

        self.assertEqual(result.status, UNSUPPORTED_SELECTOR)
        self.assertFalse(result.executable)
        self.assertIsNone(result.record)

    def test_a_resolved_report_with_no_previous_state_is_not_executable(self) -> None:
        """The report is proven; the fields the question wants are not there."""

        index = index_of(record(previous_date=None, before_shares=None,
                                before_ratio=None))

        result = execute_report_relative(
            self._intent(SELECTOR_LATEST, ROLE_PREVIOUS), index=index,
            issuer_corp_code=ISSUER, reporter="Holder")

        self.assertEqual(result.status, PREVIOUS_UNAVAILABLE)
        self.assertFalse(result.executable)
        self.assertIsNotNone(result.record)

    def test_previous_never_selects_a_neighbouring_report(self) -> None:
        """The one substitution that would look right and be wrong."""

        older = record(doc_id="holding_old", projection_chunk_id="holding_old:ch",
                       reference_date="20230101", after_shares="777",
                       after_ratio="7.77")
        newest = record(doc_id="holding_new", projection_chunk_id="holding_new:ch",
                        reference_date="20240101", previous_date="20230601",
                        before_shares="100", before_ratio="1.00")

        result = execute_report_relative(
            self._intent(SELECTOR_LATEST, ROLE_PREVIOUS),
            index=index_of(older, newest), issuer_corp_code=ISSUER,
            reporter="Holder")

        self.assertTrue(result.executable)
        self.assertEqual(result.projection.values["shares"], "100")
        self.assertNotEqual(result.projection.values["shares"], "777")
        self.assertEqual(result.projection.values["reference_date"], "20230601")
        self.assertNotEqual(result.projection.values["reference_date"], "20230101")

    def test_an_exact_date_intent_keeps_its_calendar_axis(self) -> None:
        index = index_of(record(reference_date="20240314", receipt_date="20240320"))

        by_reference = execute_report_relative(
            self._intent(SELECTOR_EXACT_REFERENCE_DATE), index=index,
            issuer_corp_code=ISSUER, reporter="Holder",
            reference_date="20240314")
        by_receipt = execute_report_relative(
            self._intent(SELECTOR_EXACT_RECEIPT_DATE), index=index,
            issuer_corp_code=ISSUER, reporter="Holder", receipt_date="20240320")
        crossed = execute_report_relative(
            self._intent(SELECTOR_EXACT_RECEIPT_DATE), index=index,
            issuer_corp_code=ISSUER, reporter="Holder", receipt_date="20240314")

        self.assertTrue(by_reference.executable)
        self.assertTrue(by_receipt.executable)
        self.assertEqual(crossed.status, NO_MATCH)

    def test_the_parsers_own_intent_object_is_accepted(self) -> None:
        """The execution contract is the parser's representation, not a new one."""

        intent = report_relative.parse("최신 보고 보유주식수")

        result = execute_report_relative(
            intent, index=index_of(record()), issuer_corp_code=ISSUER,
            reporter="Holder")

        self.assertEqual(intent.selector, SELECTOR_LATEST)
        self.assertTrue(result.executable)

    def test_an_unknown_role_executes_nothing(self) -> None:
        result = execute_report_relative(
            self._intent(SELECTOR_LATEST, "invented"), index=index_of(record()),
            issuer_corp_code=ISSUER, reporter="Holder")

        self.assertEqual(result.status, UNSUPPORTED_ROLE)
        self.assertFalse(result.executable)


class DeicticSelectorsStayAmbiguousTests(unittest.TestCase):
    """The two frozen questions this target must not quietly answer.

    A question saying only "이번 보고" or "직전보고" names no report.  A corpus
    that happens to hold a newest filing does not make it the one meant, and
    answering with it would state a holding the asker never asked about.
    """

    def setUp(self) -> None:
        self.index = index_of(
            record(reference_date="20240101"),
            record(doc_id="holding_2", projection_chunk_id="holding_2:ch",
                   reference_date="20250101"))

    def test_this_report_is_not_promoted_to_latest(self) -> None:
        for role in (ROLE_CURRENT, ROLE_PREVIOUS, ROLE_CHANGE):
            with self.subTest(role=role):
                result = execute_report_relative(
                    {"selector": SELECTOR_SELECTED_CONTEXT,
                     "projection_role": role},
                    index=self.index, issuer_corp_code=ISSUER, reporter="Holder")

                self.assertFalse(result.executable)
                self.assertIsNone(result.record)

    def test_the_parser_still_marks_both_unexecutable(self) -> None:
        """The parse is unchanged by this target, and must stay unchanged."""

        for question in ("이번 보고 보유주식수", "직전보고 보유비율"):
            with self.subTest(question=question):
                intent = report_relative.parse(question)
                self.assertEqual(intent.selector, SELECTOR_SELECTED_CONTEXT)
                self.assertFalse(intent.executable)


class MalformedRecordTests(unittest.TestCase):
    def test_a_record_without_a_reference_date_is_refused(self) -> None:
        row = record().to_dict()
        del row["reference_date"]

        with self.assertRaises(KeyError):
            HoldingReportRecord.from_dict(row)

    def test_a_malformed_reference_date_never_matches_a_real_one(self) -> None:
        """It is stored as written and compared as written, so it matches nothing."""

        index = index_of(record(reference_date="2024-13-99"))

        for wanted in ("20240101", "20241399", "2024-13-99"):
            with self.subTest(wanted=wanted):
                result = index.select_report(
                    ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
                    reference_date=wanted)
                self.assertEqual(result.status, NO_MATCH)

    def test_a_date_with_too_few_digits_selects_nothing(self) -> None:
        for wanted in ("2024", "", None, "not a date"):
            with self.subTest(wanted=wanted):
                result = index_of(record()).select_report(
                    ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
                    reference_date=wanted)
                self.assertEqual(result.status, NO_MATCH)


class CorpusUpdateTests(unittest.TestCase):
    """Two corpus snapshots, written and loaded the way production would."""

    A_IDENTITY = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                  "corpus_snapshot_id": "snapshot_a",
                  "corpus_manifest_sha256": "aaaa"}
    B_IDENTITY = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                  "corpus_snapshot_id": "snapshot_b",
                  "corpus_manifest_sha256": "bbbb"}

    OLD = record(doc_id="holding_2024", projection_chunk_id="holding_2024:ch",
                 reference_date="20240604", receipt_date="20240610")
    ANCHOR = record(doc_id="holding_anchor",
                    projection_chunk_id="holding_anchor:ch",
                    reference_date="20240314", receipt_date="20240401")
    NEW = record(doc_id="holding_2025", projection_chunk_id="holding_2025:ch",
                 reference_date="20250605", receipt_date="20250610")

    def _write(self, path, identity, records):
        path.write_text(json.dumps({
            "identity": identity,
            "complete": True,
            "correction_finality_available": True,
            "records": [r.to_dict() for r in records],
        }, ensure_ascii=False), encoding="utf-8")
        return load_index(path)

    def test_latest_moves_with_the_corpus_and_an_exact_date_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = self._write(Path(tmp) / "a.json", self.A_IDENTITY,
                            [self.ANCHOR, self.OLD])
            b = self._write(Path(tmp) / "b.json", self.B_IDENTITY,
                            [self.ANCHOR, self.OLD, self.NEW])

            for index, expected in ((a, "20240604"), (b, "20250605")):
                with self.subTest(latest_under=expected):
                    result = index.select_report(
                        ISSUER, "Holder", SELECTOR_LATEST,
                        active_corpus_identity=index.identity)
                    self.assertEqual(result.selected.reference_date, expected)

            for index in (a, b):
                result = index.select_report(
                    ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
                    reference_date="20240314",
                    active_corpus_identity=index.identity)
                self.assertEqual(result.selected.reference_date, "20240314")

    def test_yesterdays_artifact_against_todays_corpus_answers_nothing(self) -> None:
        """The failure this identity check exists for.

        Snapshot A's index still resolves a latest -- 2024-06-04 -- and it is
        wrong, because the corpus now holds 2025-06-05.  Answering from it would
        be confidently stale, so the mismatch declines instead of warning.
        """

        with tempfile.TemporaryDirectory() as tmp:
            a = self._write(Path(tmp) / "a.json", self.A_IDENTITY,
                            [self.ANCHOR, self.OLD])

            for selector, kwargs in (
                (SELECTOR_LATEST, {}),
                (SELECTOR_EXACT_REFERENCE_DATE, {"reference_date": "20240314"}),
                (SELECTOR_EXACT_RECEIPT_DATE, {"receipt_date": "20240401"}),
            ):
                with self.subTest(selector=selector):
                    result = a.select_report(
                        ISSUER, "Holder", selector,
                        active_corpus_identity=self.B_IDENTITY, **kwargs)
                    self.assertEqual(result.status, STALE_INDEX)
                    self.assertIsNone(result.selected)

    def test_the_identity_check_covers_the_corpus_content_hash(self) -> None:
        """A snapshot renamed but not rebuilt, and rebuilt but not renamed."""

        with tempfile.TemporaryDirectory() as tmp:
            a = self._write(Path(tmp) / "a.json", self.A_IDENTITY, [self.OLD])

            for drifted in ({**self.A_IDENTITY, "corpus_manifest_sha256": "cccc"},
                            {**self.A_IDENTITY, "corpus_snapshot_id": "other"},
                            {}):
                with self.subTest(drifted=drifted):
                    self.assertFalse(a.matches_corpus(drifted))


@unittest.skipUnless(ARTIFACT.is_file(), "generated index not present")
class GenericTimelineTests(unittest.TestCase):
    """Timelines of every shape the corpus actually contains.

    Named here only to pin the categories down; the selector itself sees an
    issuer code, a canonical key and a selector, and nothing that identifies
    which of these a question came from.
    """

    #: issuer, holder as a question would write it, timeline length.
    TIMELINES = (
        ("long institution", "01160363", "에코프로", 81),
        ("long corporate", "00126380", "삼성물산", 76),
        ("long individual", "00760971", "장병규", 39),
        ("long foreign", "00309503", "Fidelity Management & Research Company LLC", 25),
        ("individual", "00613318", "양현석", 18),
        ("pension institution", "00105961", "국민연금공단", 10),
        ("legal-form corporate", "00102858", "(주)영풍", 9),
        ("foreign", "00126478", "GIC Private Limited", 3),
        ("short, single report", "00102858", "HMG Global LLC", 1),
        ("holder that is itself a corpus company", "00106641", "현대자동차(주)", 1),
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = load_index(ARTIFACT)

    def test_every_timeline_shape_enumerates(self) -> None:
        for label, issuer, holder, expected in self.TIMELINES:
            with self.subTest(label=label, holder=holder):
                reports = self.index.enumerate_reports(issuer, holder)
                self.assertEqual(len(reports), expected)

    def test_a_legal_form_is_written_however_the_asker_wrote_it(self) -> None:
        """One holder, four surfaces, one timeline -- via the frozen key only."""

        for holder in ("(주)영풍", "㈜영풍", "주식회사 영풍", "영풍"):
            with self.subTest(holder=holder):
                self.assertEqual(
                    len(self.index.enumerate_reports("00102858", holder)), 9)

    def test_a_similarly_named_holder_is_a_different_holder(self) -> None:
        self.assertEqual(
            self.index.enumerate_reports("00102858", "영풍정밀"), ())

    def test_latest_is_the_maximum_reference_date_of_that_timeline(self) -> None:
        for label, issuer, holder, _ in self.TIMELINES:
            with self.subTest(label=label, holder=holder):
                reports = self.index.enumerate_reports(issuer, holder)
                result = self.index.select_report(issuer, holder, SELECTOR_LATEST)
                if result.status != RESOLVED:
                    self.assertIn(result.status,
                                  {AMBIGUOUS, CORRECTION_AMBIGUOUS})
                    continue
                self.assertEqual(result.selected.reference_date,
                                 max(r.reference_date for r in reports))

    def test_the_two_date_axes_really_are_different_in_this_corpus(self) -> None:
        """62.5% of records state a reference date the filing did not arrive on."""

        rows = [r for reports in self.index._by_pair.values() for r in reports]
        disagreeing = [r for r in rows if r.receipt_date != r.reference_date]

        self.assertGreater(len(disagreeing), len(rows) // 2)

    def test_latest_follows_reference_date_wherever_the_axes_disagree(self) -> None:
        """A measured caveat, asserted so it cannot regress silently.

        No *pair* in this corpus has its maximum reference date on a different
        filing from its maximum receipt date, so the corpus alone cannot tell
        the two orderings apart -- ``LatestSelectorTests`` carries the synthetic
        case that can, and it is what holds this line today.  Should a
        disagreeing pair ever appear, reference date must win, and this asserts
        that rather than assuming it.
        """

        for (issuer, key), reports in self.index._by_pair.items():
            result = self.index.select_report(issuer, key, SELECTOR_LATEST)
            if not result.resolved:
                continue
            by_receipt = max(reports, key=lambda r: (r.receipt_date or "", r.doc_id))
            if by_receipt.doc_id == result.selected.doc_id:
                continue
            with self.subTest(pair=(issuer, key)):
                self.assertGreaterEqual(result.selected.reference_date,
                                        by_receipt.reference_date)

    def test_every_role_either_projects_or_says_why_not(self) -> None:
        allowed = {
            ROLE_CURRENT: {PROJECTION_RESOLVED, CURRENT_UNAVAILABLE},
            ROLE_PREVIOUS: {PROJECTION_RESOLVED, PREVIOUS_UNAVAILABLE,
                            PREVIOUS_INCONSISTENT},
            ROLE_CHANGE: {PROJECTION_RESOLVED, CHANGE_UNAVAILABLE},
        }
        seen = {role: set() for role in allowed}
        for reports in self.index._by_pair.values():
            for report in reports:
                for role, statuses in allowed.items():
                    status = project_role(report, role).status
                    self.assertIn(status, statuses)
                    seen[role].add(status)

        # The corpus contains first reports, so the previous role must be
        # observed declining as well as resolving.
        self.assertIn(PREVIOUS_UNAVAILABLE, seen[ROLE_PREVIOUS])
        self.assertIn(PROJECTION_RESOLVED, seen[ROLE_PREVIOUS])

    def test_a_first_report_in_the_corpus_declines_the_previous_role(self) -> None:
        firsts = [r for reports in self.index._by_pair.values() for r in reports
                  if not r.previous_date]
        self.assertTrue(firsts)
        for report in firsts:
            self.assertIn(project_role(report, ROLE_PREVIOUS).status,
                          {PREVIOUS_UNAVAILABLE, PREVIOUS_INCONSISTENT})

    def test_no_tied_latest_date_in_the_corpus_is_ever_resolved(self) -> None:
        """Every tie this corpus contains is inside a correction-bearing pair.

        So they decline at the correction gate rather than the tie gate, and
        which gate fired is not the point: two filings state the same holder on
        the same day, and no single report may be returned for either reason.
        The synthetic tie in ``LatestSelectorTests`` covers the tie gate itself.
        """

        tied = 0
        for (issuer, key), reports in self.index._by_pair.items():
            newest = max(r.reference_date for r in reports)
            if len({r.doc_id for r in reports if r.reference_date == newest}) < 2:
                continue
            tied += 1
            with self.subTest(pair=(issuer, key)):
                result = self.index.select_report(issuer, key, SELECTOR_LATEST)
                self.assertIn(result.status, {AMBIGUOUS, CORRECTION_AMBIGUOUS})
                self.assertIsNone(result.selected)
        self.assertGreater(tied, 0)


if __name__ == "__main__":
    unittest.main()
