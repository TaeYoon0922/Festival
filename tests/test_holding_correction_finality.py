import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning import holding_report_relative as report_relative
from app.reasoning.holding_correction_finality import (
    ARTIFACT_SCHEMA_VERSION,
    COLLAPSED,
    STATUS_AMBIGUOUS,
    STATUS_RESOLVED,
    STATUS_UNRESOLVED,
    UNPROVEN,
    CorrectionChain,
    HoldingCorrectionFinality,
    load_finality,
)
from app.reasoning.holding_report_index import (
    AMBIGUOUS,
    CORRECTION_AMBIGUOUS,
    NO_MATCH,
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
    UNSUPPORTED_SELECTOR,
    HoldingReportIndex,
    HoldingReportRecord,
    execute_report_relative,
    load_index,
    project_role,
)

ROOT = Path(__file__).resolve().parents[1]
INDEX_ARTIFACT = ROOT / "data/corpus/holding_report_index.json"
FINALITY_ARTIFACT = ROOT / "data/corpus/holding_correction_finality.json"

ISSUER = "00000001"
OTHER_ISSUER = "00000002"

#: The corpus binding both artifacts must agree on.
IDENTITY = {
    "corpus_snapshot_id": "snapshot_a",
    "corpus_manifest_sha256": "aaaa",
    "source_holding_disclosure_count": 3,
}


def record(**over) -> HoldingReportRecord:
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


def chain(**over) -> CorrectionChain:
    base = dict(
        group_id="holding_original",
        root_doc_id="holding_original",
        members=("holding_original", "holding_correction"),
        final_doc_id="holding_correction",
        status=STATUS_RESOLVED,
        resolution_rule="correction_notice",
        confidence=0.95,
        head_complete=True,
    )
    base.update(over)
    return CorrectionChain(**base)


def finality_of(*chains, **over) -> HoldingCorrectionFinality:
    settings = dict(identity=dict(IDENTITY), complete=True)
    settings.update(over)
    return HoldingCorrectionFinality(chains, **settings)


def index_of(*records, **over) -> HoldingReportIndex:
    settings = dict(complete=True, identity=dict(IDENTITY))
    settings.update(over)
    return HoldingReportIndex(records, **settings)


class SourceIdentityTests(unittest.TestCase):
    """The two artifacts are usable together only for the same corpus."""

    def test_a_matching_identity_attaches(self) -> None:
        index = index_of(record(), correction_finality=finality_of(chain()))

        self.assertEqual(index.correction_finality_status, "attached")
        self.assertIsNotNone(index.correction_finality)

    def test_a_source_from_another_corpus_is_refused(self) -> None:
        """It may name a final document this corpus does not hold."""

        for drift in ("corpus_snapshot_id", "corpus_manifest_sha256",
                      "source_holding_disclosure_count"):
            with self.subTest(field=drift):
                other = {**IDENTITY, drift: "moved-on"}
                index = index_of(
                    record(), correction_finality=finality_of(
                        chain(), identity=other))

                self.assertEqual(index.correction_finality_status,
                                 "identity_mismatch")
                self.assertIsNone(index.correction_finality)

    def test_a_missing_binding_field_is_a_mismatch_not_a_pass(self) -> None:
        for absent in ("corpus_snapshot_id", "corpus_manifest_sha256",
                       "source_holding_disclosure_count"):
            with self.subTest(missing=absent):
                partial = {k: v for k, v in IDENTITY.items() if k != absent}
                source = finality_of(chain(), identity=partial)

                self.assertFalse(source.matches_identity(IDENTITY))
                self.assertEqual(
                    index_of(record(),
                             correction_finality=source
                             ).correction_finality_status,
                    "identity_mismatch")

    def test_an_incomplete_source_is_refused(self) -> None:
        index = index_of(record(),
                         correction_finality=finality_of(chain(), complete=False))

        self.assertEqual(index.correction_finality_status, "unusable")

    def test_a_self_contradicting_source_is_refused(self) -> None:
        """A duplicate group or a final member outside its own group."""

        duplicated = finality_of(chain(), chain())
        detached = finality_of(chain(final_doc_id="holding_elsewhere"))

        for source in (duplicated, detached):
            with self.subTest(problems=source.problems):
                self.assertTrue(source.problems)
                self.assertFalse(source.usable)

    def test_a_malformed_artifact_yields_no_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (
                ("not_json.json", "{{{"),
                ("no_groups.json", json.dumps({"header": {}})),
                ("wrong_version.json", json.dumps(
                    {"header": {"artifact_schema_version": "0.0"},
                     "groups": []})),
                ("bad_group.json", json.dumps(
                    {"header": {"artifact_schema_version":
                                ARTIFACT_SCHEMA_VERSION},
                     "groups": [{"group_id": "g"}]})),
                ("memberless.json", json.dumps(
                    {"header": {"artifact_schema_version":
                                ARTIFACT_SCHEMA_VERSION},
                     "groups": [{"group_id": "g", "root_doc_id": "g",
                                 "members": [], "status": "resolved"}]})),
            ):
                with self.subTest(name=name):
                    path = Path(tmp) / name
                    path.write_text(text, encoding="utf-8")
                    self.assertIsNone(load_finality(path))

    def test_a_missing_artifact_yields_no_source(self) -> None:
        self.assertIsNone(load_finality(Path("does/not/exist.json")))


class CollapseTests(unittest.TestCase):
    """What a proven chain removes, and what an unproven one refuses."""

    ORIGINAL = record(doc_id="holding_original",
                      projection_chunk_id="holding_original:ch",
                      reference_date="20240101", after_shares="100")
    CORRECTION = record(doc_id="holding_correction",
                        projection_chunk_id="holding_correction:ch",
                        reference_date="20240101", receipt_date="20240210",
                        after_shares="150", is_correction=True)

    def _index(self, *chains, **over):
        return index_of(self.ORIGINAL, self.CORRECTION,
                        correction_finality=finality_of(*chains, **over))

    def test_a_proven_chain_drops_the_superseded_document(self) -> None:
        result = self._index(chain()).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.selected.doc_id, "holding_correction")
        self.assertEqual(result.selected.after_shares, "150")
        self.assertEqual(result.detail["superseded_doc_ids"],
                         ["holding_original"])

    def test_an_ambiguous_chain_declines(self) -> None:
        result = self._index(
            chain(status=STATUS_AMBIGUOUS, final_doc_id=None)).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)
        self.assertIsNone(result.selected)

    def test_an_unresolved_chain_declines(self) -> None:
        result = self._index(
            chain(status=STATUS_UNRESOLVED, final_doc_id=None)).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)

    def test_a_correction_in_no_group_declines(self) -> None:
        """An absent group is an unproven correction, not an absent one."""

        result = self._index(
            chain(group_id="holding_elsewhere", root_doc_id="holding_elsewhere",
                  members=("holding_elsewhere",), final_doc_id=None,
                  status=STATUS_AMBIGUOUS)).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)

    def test_no_source_at_all_still_declines(self) -> None:
        result = index_of(self.ORIGINAL, self.CORRECTION).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)
        self.assertEqual(result.detail["finality_source"], "absent")

    def test_a_final_document_absent_from_the_timeline_declines(self) -> None:
        """Collapsing would delete this holder's reports and leave none."""

        result = self._index(
            chain(members=("holding_original", "holding_offstage"),
                  final_doc_id="holding_offstage")).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)

    def test_a_correction_free_timeline_needs_no_source(self) -> None:
        plain = index_of(record(), correction_finality=finality_of(
            chain(), complete=False))

        self.assertEqual(
            plain.select_report(ISSUER, "Holder", SELECTOR_LATEST).status,
            RESOLVED)


class CrossIdentityGuardTests(unittest.TestCase):
    """A chain may only collapse a timeline it stays inside."""

    def test_a_chain_crossing_reporters_is_refused(self) -> None:
        ours = record(doc_id="holding_original",
                      projection_chunk_id="holding_original:ch",
                      reference_date="20240101")
        theirs = record(doc_id="holding_correction",
                        projection_chunk_id="holding_correction:ch",
                        reporter_key="other", raw_reporter="Other",
                        reference_date="20240201", is_correction=True)
        ours_correction = record(doc_id="holding_correction",
                                 projection_chunk_id="holding_correction:ch2",
                                 reference_date="20240201", is_correction=True)

        index = index_of(ours, theirs, ours_correction,
                         correction_finality=finality_of(chain()))
        result = index.select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)
        self.assertIn("issuer or reporter", result.detail["reason"])

    def test_a_chain_reaching_into_another_timeline_is_refused(self) -> None:
        """A member that exists, but only under a different holder."""

        ours = record(doc_id="holding_original",
                      projection_chunk_id="holding_original:ch",
                      reference_date="20240101")
        ours_correction = record(doc_id="holding_correction",
                                 projection_chunk_id="holding_correction:ch",
                                 reference_date="20240201", is_correction=True)
        elsewhere = record(doc_id="holding_offstage",
                           projection_chunk_id="holding_offstage:ch",
                           reporter_key="other", raw_reporter="Other",
                           reference_date="20240301")

        index = index_of(ours, ours_correction, elsewhere,
                         correction_finality=finality_of(chain(
                             members=("holding_original", "holding_offstage",
                                      "holding_correction"))))
        result = index.select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, CORRECTION_AMBIGUOUS)
        self.assertEqual(result.detail["foreign_members"], ["holding_offstage"])

    def test_a_chain_crossing_issuers_is_refused(self) -> None:
        ours = record(doc_id="holding_original",
                      projection_chunk_id="holding_original:ch")
        elsewhere = record(doc_id="holding_correction",
                           projection_chunk_id="holding_correction:ch",
                           issuer_corp_code=OTHER_ISSUER,
                           reference_date="20240201", is_correction=True)
        ours_correction = record(doc_id="holding_correction",
                                 projection_chunk_id="holding_correction:ch2",
                                 reference_date="20240201", is_correction=True)

        index = index_of(ours, elsewhere, ours_correction,
                         correction_finality=finality_of(chain()))

        self.assertEqual(
            index.select_report(ISSUER, "Holder", SELECTOR_LATEST).status,
            CORRECTION_AMBIGUOUS)


class CollapseBeforeOrderingTests(unittest.TestCase):
    """The order of the two steps, proven by a case where it decides."""

    def test_the_superseded_document_never_wins_on_date(self) -> None:
        """The original states the newest date; the correction restates an older
        one and arrives months later.  Ordering first returns the withdrawn
        report; collapsing first returns the one that still stands.
        """

        original = record(doc_id="holding_original",
                          projection_chunk_id="holding_original:ch",
                          reference_date="20241201", receipt_date="20241205",
                          after_shares="100")
        correction = record(doc_id="holding_correction",
                            projection_chunk_id="holding_correction:ch",
                            reference_date="20240701", receipt_date="20250110",
                            after_shares="150", is_correction=True)

        index = index_of(original, correction,
                         correction_finality=finality_of(chain()))
        result = index.select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.selected.doc_id, "holding_correction")
        self.assertEqual(result.selected.reference_date, "20240701")
        self.assertNotEqual(result.selected.reference_date, "20241201")

    def test_an_unrelated_later_report_still_wins(self) -> None:
        """Collapse removes superseded filings; it does not promote a chain."""

        original = record(doc_id="holding_original",
                          projection_chunk_id="holding_original:ch",
                          reference_date="20240101")
        correction = record(doc_id="holding_correction",
                            projection_chunk_id="holding_correction:ch",
                            reference_date="20240101", receipt_date="20250110",
                            is_correction=True)
        independent = record(doc_id="holding_later",
                             projection_chunk_id="holding_later:ch",
                             reference_date="20240601", receipt_date="20240605")

        result = index_of(original, correction, independent,
                          correction_finality=finality_of(chain())).select_report(
            ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.selected.doc_id, "holding_later")
        self.assertEqual(result.selected.reference_date, "20240601")


class SelectorsAfterCollapseTests(unittest.TestCase):
    """Collapse changes which reports are eligible, never what a selector means."""

    ORIGINAL = record(doc_id="holding_original",
                      projection_chunk_id="holding_original:ch",
                      reference_date="20240101", receipt_date="20240105",
                      after_shares="100")
    CORRECTION = record(doc_id="holding_correction",
                        projection_chunk_id="holding_correction:ch",
                        reference_date="20240301", receipt_date="20240310",
                        after_shares="150", previous_date="20231201",
                        before_shares="90", before_ratio="0.90",
                        change_shares="60", change_ratio="0.60",
                        change_direction="increase", is_correction=True)

    def setUp(self) -> None:
        self.index = index_of(self.ORIGINAL, self.CORRECTION,
                              correction_finality=finality_of(chain()))

    def test_a_reference_date_changing_correction_uses_the_final_date(self) -> None:
        result = self.index.select_report(ISSUER, "Holder", SELECTOR_LATEST)

        self.assertEqual(result.selected.reference_date, "20240301")

    def test_exact_reference_date_sees_only_eligible_reports(self) -> None:
        superseded = self.index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20240101")
        final = self.index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20240301")

        self.assertEqual(superseded.status, NO_MATCH)
        self.assertEqual(final.status, RESOLVED)
        self.assertEqual(final.selected.doc_id, "holding_correction")

    def test_exact_receipt_date_keeps_its_own_axis(self) -> None:
        by_receipt = self.index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_RECEIPT_DATE,
            receipt_date="20240310")
        crossed = self.index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_RECEIPT_DATE,
            receipt_date="20240301")
        superseded = self.index.select_report(
            ISSUER, "Holder", SELECTOR_EXACT_RECEIPT_DATE,
            receipt_date="20240105")

        self.assertEqual(by_receipt.status, RESOLVED)
        self.assertEqual(crossed.status, NO_MATCH)
        self.assertEqual(superseded.status, NO_MATCH)

    def test_previous_still_reads_the_selected_report_itself(self) -> None:
        """Not the correction predecessor, which is a different relationship."""

        result = execute_report_relative(
            {"selector": SELECTOR_LATEST, "projection_role": ROLE_PREVIOUS},
            index=self.index, issuer_corp_code=ISSUER, reporter="Holder")

        self.assertTrue(result.executable)
        self.assertEqual(result.projection.values["shares"], "90")
        self.assertEqual(result.projection.values["reference_date"], "20231201")
        self.assertNotEqual(result.projection.values["shares"], "100")
        self.assertNotEqual(result.projection.values["reference_date"],
                            self.ORIGINAL.reference_date)

    def test_change_still_reads_the_selected_report_itself(self) -> None:
        result = execute_report_relative(
            {"selector": SELECTOR_LATEST, "projection_role": ROLE_CHANGE},
            index=self.index, issuer_corp_code=ISSUER, reporter="Holder")

        self.assertEqual(result.projection.values["shares"], "60")
        self.assertEqual(result.projection.values["direction"], "increase")

    def test_selected_context_is_still_unsupported(self) -> None:
        result = execute_report_relative(
            {"selector": SELECTOR_SELECTED_CONTEXT,
             "projection_role": ROLE_CURRENT},
            index=self.index, issuer_corp_code=ISSUER, reporter="Holder")

        self.assertEqual(result.status, UNSUPPORTED_SELECTOR)
        self.assertFalse(result.executable)

    def test_a_first_report_still_has_no_previous_state(self) -> None:
        first = record(doc_id="holding_correction",
                       projection_chunk_id="holding_correction:ch",
                       reference_date="20240301", previous_date=None,
                       before_shares=None, before_ratio=None,
                       is_correction=True)
        index = index_of(self.ORIGINAL, first,
                         correction_finality=finality_of(chain()))

        result = execute_report_relative(
            {"selector": SELECTOR_LATEST, "projection_role": ROLE_PREVIOUS},
            index=index, issuer_corp_code=ISSUER, reporter="Holder")

        self.assertEqual(result.status, PREVIOUS_UNAVAILABLE)


class IntraDocumentAmbiguityTests(unittest.TestCase):
    """Collapse is document-level, and stops there."""

    def test_every_projection_of_the_final_document_survives(self) -> None:
        original = record(doc_id="holding_original",
                          projection_chunk_id="holding_original:ch",
                          reference_date="20240101")
        rows = [record(doc_id="holding_correction",
                       projection_chunk_id=f"holding_correction:ch{n}",
                       reference_date="20240101", after_shares=shares,
                       is_correction=True)
                for n, shares in enumerate(("100", "150", "150"))]

        index = index_of(original, *rows,
                         correction_finality=finality_of(chain()))
        eligible = index.correction_finality.collapse(
            index.enumerate_reports(ISSUER, "Holder"))

        self.assertEqual(eligible.status, COLLAPSED)
        self.assertEqual(len(eligible.eligible), 3)
        self.assertEqual(
            {r.doc_id for r in eligible.eligible}, {"holding_correction"})

    def test_several_projections_of_the_final_document_stay_ambiguous(self) -> None:
        """Choosing between a 정정신고's own rows is a semantics B.3 has not got.

        The document is proven final; which of the values it prints is the
        holder's state is a different question, and answering it by position,
        size or section would be inventing the answer.
        """

        original = record(doc_id="holding_original",
                          projection_chunk_id="holding_original:ch",
                          reference_date="20240101")
        rows = [record(doc_id="holding_correction",
                       projection_chunk_id=f"holding_correction:ch{n}",
                       reference_date="20240101", after_shares=shares,
                       is_correction=True)
                for n, shares in enumerate(("100", "150", "150"))]

        index = index_of(original, *rows,
                         correction_finality=finality_of(chain()))

        for selector, kwargs in (
            (SELECTOR_LATEST, {}),
            (SELECTOR_EXACT_REFERENCE_DATE, {"reference_date": "20240101"}),
            (SELECTOR_EXACT_RECEIPT_DATE, {"receipt_date": "20240105"}),
        ):
            with self.subTest(selector=selector):
                result = index.select_report(
                    ISSUER, "Holder", selector, **kwargs)
                self.assertEqual(result.status, AMBIGUOUS)
                self.assertIsNone(result.selected)


class DeicticSelectorsUnchangedTests(unittest.TestCase):
    """Correction finality does not name a report a question failed to name."""

    def test_this_report_and_previous_report_stay_clarification(self) -> None:
        for question in ("이번 보고 보유주식수", "직전보고 보유비율"):
            with self.subTest(question=question):
                intent = report_relative.parse(question)
                self.assertEqual(intent.selector, SELECTOR_SELECTED_CONTEXT)
                self.assertFalse(intent.executable)


@unittest.skipUnless(FINALITY_ARTIFACT.is_file() and INDEX_ARTIFACT.is_file(),
                     "generated artifacts not present")
class GeneratedArtifactTests(unittest.TestCase):
    """The artifacts actually shipped, over the whole holding corpus."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_finality(FINALITY_ARTIFACT)
        cls.base = load_index(INDEX_ARTIFACT)
        cls.index = load_index(INDEX_ARTIFACT, finality_path=FINALITY_ARTIFACT)
        cls.active = dict(cls.index.identity)

    def test_the_source_binds_to_the_same_corpus_as_the_index(self) -> None:
        self.assertTrue(self.source.usable)
        self.assertTrue(self.source.matches_identity(self.base.identity))
        self.assertEqual(self.index.correction_finality_status, "attached")

    def test_every_group_carries_a_status_p0a_could_have_produced(self) -> None:
        for group in self.source.chains:
            with self.subTest(group=group.group_id):
                self.assertIn(group.status, {STATUS_RESOLVED, STATUS_AMBIGUOUS,
                                             STATUS_UNRESOLVED})
                if group.is_resolved:
                    self.assertIn(group.final_doc_id, group.members)
                    self.assertGreater(len(group.members), 1)
                else:
                    self.assertIsNone(group.final_doc_id)

    def test_an_unproven_group_supersedes_nothing(self) -> None:
        for group in self.source.chains:
            if not group.is_resolved:
                self.assertEqual(group.superseded, ())

    def test_the_weakest_frozen_rule_is_preserved_not_filtered(self) -> None:
        """Three chains rest on event_title_key at confidence 0.70.

        They are resolved under the frozen P0-A contract, so they are carried
        through as resolved.  Adding a holding-only confidence floor here would
        be a second correction policy, which is the one thing this artifact is
        not allowed to become.
        """

        weak = [g for g in self.source.chains
                if g.is_resolved and "event_title_key" in g.resolution_rule]

        self.assertTrue(weak)
        for group in weak:
            with self.subTest(group=group.group_id):
                self.assertEqual(group.status, STATUS_RESOLVED)
                self.assertLess(group.confidence, 0.9)
                self.assertIsNotNone(group.final_doc_id)

    def test_an_incomplete_head_is_recorded_rather_than_disguised(self) -> None:
        """One resolved chain begins at a correction whose target is not here.

        Its finality is unaffected -- nothing supersedes the tail -- but it does
        not begin at an original filing, and the artifact says so.
        """

        incomplete = [g for g in self.source.chains
                      if g.is_resolved and not g.head_complete]

        self.assertEqual(len(incomplete), 1)
        self.assertIsNotNone(incomplete[0].final_doc_id)

    def test_a_multi_step_chain_ends_at_its_last_member(self) -> None:
        longest = max(self.source.chains, key=lambda g: len(g.members))

        self.assertGreater(len(longest.members), 2)
        self.assertEqual(longest.final_doc_id, longest.members[-1])
        self.assertEqual(len(longest.superseded), len(longest.members) - 1)

    def test_correction_free_timelines_are_completely_unaffected(self) -> None:
        changed = 0
        for (issuer, key), records in self.index._by_pair.items():
            if any(r.is_correction for r in records):
                continue
            before = self.base.select_report(issuer, key, SELECTOR_LATEST)
            after = self.index.select_report(
                issuer, key, SELECTOR_LATEST,
                active_corpus_identity=self.active)
            if (before.status, before.selected) != (after.status, after.selected):
                changed += 1
        self.assertEqual(changed, 0)

    def test_every_correction_bearing_pair_resolves_or_declines(self) -> None:
        allowed = {RESOLVED, CORRECTION_AMBIGUOUS, AMBIGUOUS}
        resolved = declined = 0
        for (issuer, key), records in self.index._by_pair.items():
            if not any(r.is_correction for r in records):
                continue
            result = self.index.select_report(
                issuer, key, SELECTOR_LATEST,
                active_corpus_identity=self.active)
            self.assertIn(result.status, allowed)
            if result.status == RESOLVED:
                resolved += 1
                # A resolved pair may never still contain a superseded filing.
                self.assertNotIn(result.selected.doc_id,
                                 result.detail.get("superseded_doc_ids", []))
            else:
                declined += 1
                self.assertIsNone(result.selected)
        self.assertGreater(resolved, 0)
        self.assertGreater(declined, 0)

    def test_correction_finality_only_ever_adds_answers(self) -> None:
        """No pair that answered before may answer differently now."""

        for (issuer, key) in self.index._by_pair:
            before = self.base.select_report(issuer, key, SELECTOR_LATEST)
            after = self.index.select_report(
                issuer, key, SELECTOR_LATEST,
                active_corpus_identity=self.active)
            if before.status != RESOLVED:
                continue
            with self.subTest(pair=(issuer, key)):
                self.assertEqual(after.status, RESOLVED)
                self.assertEqual(after.selected.doc_id, before.selected.doc_id)

    def test_a_tied_latest_date_resolves_only_through_a_correction(self) -> None:
        resolved_by_chain = still_tied = 0
        for (issuer, key), records in self.index._by_pair.items():
            newest = max(r.reference_date for r in records)
            top = {r.doc_id for r in records if r.reference_date == newest}
            if len(top) < 2:
                continue
            result = self.index.select_report(
                issuer, key, SELECTOR_LATEST,
                active_corpus_identity=self.active)
            if result.status == RESOLVED:
                # The tie went away because a chain proved one filing
                # superseded, never because something broke the tie.
                self.assertTrue(result.detail.get("superseded_doc_ids"))
                resolved_by_chain += 1
            else:
                self.assertIn(result.status, {AMBIGUOUS, CORRECTION_AMBIGUOUS})
                self.assertIsNone(result.selected)
                still_tied += 1
        self.assertGreater(resolved_by_chain, 0)
        self.assertGreater(still_tied, 0)

    def test_a_resolved_pair_carries_the_final_documents_own_date(self) -> None:
        for (issuer, key), records in self.index._by_pair.items():
            result = self.index.select_report(
                issuer, key, SELECTOR_LATEST,
                active_corpus_identity=self.active)
            superseded = set(result.detail.get("superseded_doc_ids", ()))
            if result.status != RESOLVED or not superseded:
                continue
            with self.subTest(pair=(issuer, key)):
                eligible = [r for r in records if r.doc_id not in superseded]
                self.assertEqual(
                    result.selected.reference_date,
                    max(r.reference_date for r in eligible))

    def test_a_stale_source_leaves_every_correction_pair_declining(self) -> None:
        stale = HoldingReportIndex(
            [r for records in self.index._by_pair.values() for r in records],
            identity={**self.index.identity, "corpus_manifest_sha256": "moved"},
            complete=True,
            correction_finality=self.source,
        )
        self.assertEqual(stale.correction_finality_status, "identity_mismatch")

        for (issuer, key), records in stale._by_pair.items():
            if not any(r.is_correction for r in records):
                continue
            with self.subTest(pair=(issuer, key)):
                self.assertEqual(
                    stale.select_report(issuer, key, SELECTOR_LATEST).status,
                    CORRECTION_AMBIGUOUS)

    def test_every_role_still_projects_or_says_why_not(self) -> None:
        for (issuer, key) in self.index._by_pair:
            result = self.index.select_report(
                issuer, key, SELECTOR_LATEST,
                active_corpus_identity=self.active)
            if result.status != RESOLVED:
                continue
            for role in (ROLE_CURRENT, ROLE_PREVIOUS, ROLE_CHANGE):
                projected = project_role(result.selected, role)
                self.assertTrue(projected.status)


class ResolvedStatusContractTests(unittest.TestCase):
    """A chain is proven because P0-A said so, not because it looks proven."""

    def test_a_final_member_alone_does_not_make_a_chain_resolved(self) -> None:
        """P0-A marks its lone ambiguous corrections ``is_latest`` too.

        Reading a recorded terminal as proof of finality is exactly how "we
        could not tell which filing supersedes which" becomes "this one does".
        """

        for status in (STATUS_AMBIGUOUS, STATUS_UNRESOLVED):
            with self.subTest(status=status):
                mislabelled = chain(status=status,
                                    final_doc_id="holding_correction")

                self.assertFalse(mislabelled.is_resolved)
                self.assertEqual(mislabelled.superseded, ())

    def test_several_members_alone_do_not_make_a_chain_resolved(self) -> None:
        long_but_unproven = chain(
            status=STATUS_AMBIGUOUS, final_doc_id=None,
            members=("a", "b", "c"), root_doc_id="a", group_id="a")

        self.assertFalse(long_but_unproven.is_resolved)
        self.assertEqual(long_but_unproven.superseded, ())


class GeneratorClassificationTests(unittest.TestCase):
    """What the generator writes down, driven by synthetic P0-A rows.

    These run without rebuilding the graph, so the rules that turn P0-A's member
    rows into artifact groups are checked on every test run rather than only
    when the corpus is regenerated.
    """

    @staticmethod
    def _member(doc_id, order, *, latest=False, parent=None,
                status="resolved", source="correction_notice", confidence=0.95,
                root="holding_a", correction=True):
        from app.reasoning.correction_graph import CorrectionGroupMember

        return CorrectionGroupMember(
            doc_id=doc_id, correction_group_id=root, root_doc_id=root,
            parent_doc_id=parent, correction_order=order, is_latest=latest,
            resolution_status=status, resolution_source=source,
            confidence=confidence, is_correction=correction)

    def _materialize(self, members, *, roots_are_corrections=False):
        import importlib

        from app.reasoning.correction_graph import DisclosureRecord

        generator = importlib.import_module(
            "scripts.build_holding_correction_finality")
        by_doc = {
            m.doc_id: DisclosureRecord(
                doc_id=m.doc_id, corp_code="1", doc_group="holding",
                report_nm="r", rcept_no="1",
                is_correction=roots_are_corrections or m.is_correction)
            for m in members
        }
        return generator.materialize_group(
            members[0].root_doc_id, members, by_doc,
            {m.doc_id for m in members})

    def test_the_member_p0a_marked_latest_is_the_final_one(self) -> None:
        """Not the last row, and not the newest arrival -- the marked one."""

        group = self._materialize([
            self._member("holding_a", 0, correction=False),
            self._member("holding_b", 1, latest=True, parent="holding_a"),
        ])

        self.assertEqual(group["status"], STATUS_RESOLVED)
        self.assertEqual(group["final_doc_id"], "holding_b")
        self.assertEqual(group["members"], ["holding_a", "holding_b"])

    def test_a_group_with_two_terminals_is_not_called_resolved(self) -> None:
        """Two filings claiming to be final is not a chain with a final one."""

        group = self._materialize([
            self._member("holding_a", 0, correction=False),
            self._member("holding_b", 1, latest=True, parent="holding_a"),
            self._member("holding_c", 2, latest=True, parent="holding_a"),
        ])

        self.assertNotEqual(group["status"], STATUS_RESOLVED)
        self.assertIsNone(group["final_doc_id"])

    def test_a_group_with_no_terminal_is_not_called_resolved(self) -> None:
        group = self._materialize([
            self._member("holding_a", 0, correction=False),
            self._member("holding_b", 1, parent="holding_a"),
        ])

        self.assertNotEqual(group["status"], STATUS_RESOLVED)
        self.assertIsNone(group["final_doc_id"])

    def test_a_member_p0a_left_unproven_blocks_the_whole_group(self) -> None:
        group = self._materialize([
            self._member("holding_a", 0, correction=False),
            self._member("holding_b", 1, latest=True, parent="holding_a",
                         status="ambiguous"),
        ])

        self.assertNotEqual(group["status"], STATUS_RESOLVED)
        self.assertIsNone(group["final_doc_id"])

    def test_a_lone_correction_keeps_its_own_unproven_status(self) -> None:
        for status in ("ambiguous", "unresolved"):
            with self.subTest(status=status):
                group = self._materialize([
                    self._member("holding_a", 0, latest=True, status=status,
                                 source="multiple_candidates", confidence=0.0),
                ])

                self.assertEqual(group["status"], status)
                self.assertIsNone(group["final_doc_id"])

    def test_a_root_that_is_itself_a_correction_is_marked_incomplete(self) -> None:
        group = self._materialize([
            self._member("holding_a", 0),
            self._member("holding_b", 1, latest=True, parent="holding_a"),
        ], roots_are_corrections=True)

        self.assertFalse(group["head_complete"])
        self.assertEqual(group["final_doc_id"], "holding_b")

    def test_the_weakest_rule_and_confidence_are_recorded_not_filtered(self) -> None:
        group = self._materialize([
            self._member("holding_a", 0, correction=False),
            self._member("holding_b", 1, latest=True, parent="holding_a",
                         source="event_title_key", confidence=0.7),
        ])

        self.assertEqual(group["status"], STATUS_RESOLVED)
        self.assertEqual(group["resolution_rule"], "event_title_key")
        self.assertEqual(group["confidence"], 0.7)


class GeneratorTests(unittest.TestCase):
    """The generator refuses what it cannot verify, and repeats itself."""

    def _generator(self):
        import importlib

        return importlib.import_module("scripts.build_holding_correction_finality")

    def test_the_serialization_round_trips(self) -> None:
        generator = self._generator()
        payload = {"header": {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION},
                   "complete": True,
                   "groups": [chain().to_dict(), chain(group_id="b").to_dict()]}

        restored = json.loads(generator._serialize(payload))

        self.assertEqual(restored["groups"][0]["group_id"], "holding_original")
        self.assertEqual(len(restored["groups"]), 2)

    def test_an_incomplete_source_is_refused_and_writes_nothing(self) -> None:
        import contextlib
        import io
        import sys as _sys

        generator = self._generator()
        original = generator.build
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "finality.json"
            generator.build = lambda: ([], {
                "disclosures_loaded": 4204,
                "holding_documents": 1083,
                "holding_correction_documents": 41,
                "holding_correction_documents_classified": 40,
                "unclassified_holding_correction_documents": ["holding_missing"],
                "notices_extracted": 771,
                "groups_touching_holding": 0,
                "status_counts": {},
                "graph": {},
                "complete": False,
            })
            try:
                argv = _sys.argv
                _sys.argv = ["build", "--out", str(out)]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = generator.main()
                finally:
                    _sys.argv = argv
            finally:
                generator.build = original

            self.assertNotEqual(code, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
