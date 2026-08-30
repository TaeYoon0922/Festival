"""A holder written with its legal form is the same holder written without it.

The corpus records a filer the way the filer typed it, so one entity arrives as
``(주)하이브``, ``㈜하이브``, ``주식회사 하이브`` and ``하이브``.  These tests fix
the boundary: those forms are one holder, and names that merely resemble each
other are not.  The second half matters more than the first -- a rule that
recovers ``(주)영풍`` by also merging ``영풍정밀`` has made the answer wrong.
"""

import unittest

from app.reasoning.answer_composer import (
    _holding_reporter_matches,
    _reported_holding_events,
)
from app.reasoning.holding_event_resolver import _reporter_matches
from app.reasoning.holding_reporter import (
    canonical_reporter_key,
    reporter_matches,
)


class CanonicalKeyTests(unittest.TestCase):
    """Section 7: every designator form the corpus actually contains."""

    def test_a_bracketed_prefix_is_a_legal_form_not_part_of_the_name(self) -> None:
        for raw in ("(주)하이브", "㈜하이브", "( 주 )하이브"):
            with self.subTest(raw=raw):
                self.assertEqual(canonical_reporter_key(raw), "하이브")

    def test_a_bracketed_suffix_is_stripped_too(self) -> None:
        for raw in ("하이브(주)", "하이브㈜"):
            with self.subTest(raw=raw):
                self.assertEqual(canonical_reporter_key(raw), "하이브")

    def test_a_spelled_legal_form_is_stripped_at_either_edge(self) -> None:
        for raw in ("주식회사 하이브", "하이브 주식회사", "하이브주식회사"):
            with self.subTest(raw=raw):
                self.assertEqual(canonical_reporter_key(raw), "하이브")

    def test_limited_company_forms_are_recognized(self) -> None:
        for raw in ("(유)대한", "유한회사 대한", "대한 유한회사", "유한책임회사 대한"):
            with self.subTest(raw=raw):
                self.assertEqual(canonical_reporter_key(raw), "대한")

    def test_association_and_foundation_forms_are_recognized(self) -> None:
        self.assertEqual(canonical_reporter_key("사단법인 대한"), "대한")
        self.assertEqual(canonical_reporter_key("재단법인 대한"), "대한")

    def test_whitespace_and_punctuation_still_normalize(self) -> None:
        self.assertEqual(
            canonical_reporter_key("BlackRock  Fund Advisors"),
            canonical_reporter_key("BlackRockFundAdvisors"),
        )
        self.assertEqual(
            canonical_reporter_key("T. Rowe Price Associates, Inc"),
            canonical_reporter_key("T Rowe Price Associates Inc"),
        )

    def test_an_undecorated_name_is_unchanged(self) -> None:
        self.assertEqual(canonical_reporter_key("국민연금공단"), "국민연금공단")

    def test_only_one_designator_is_removed_per_edge(self) -> None:
        """Stacked forms are not corpus-attested, so stripping stays conservative.

        Removing designators repeatedly would eat into names on a pattern no
        filing actually uses, so exactly one comes off each edge.
        """

        self.assertEqual(canonical_reporter_key("(주)주식회사하이브"),
                         "주식회사하이브")

    def test_a_name_that_is_only_a_legal_form_is_not_stripped_away(self) -> None:
        """The non-empty guard: a holder recorded as its form compares as itself."""

        self.assertEqual(canonical_reporter_key("주식회사"), "주식회사")
        self.assertEqual(canonical_reporter_key("(주)"), "주")

    def test_canonicalization_does_not_mutate_the_display_value(self) -> None:
        raw = "(주)하이브"
        canonical_reporter_key(raw)
        self.assertEqual(raw, "(주)하이브")


class DesignatorMatchTests(unittest.TestCase):
    def test_designator_variants_match_the_bare_name(self) -> None:
        for corpus in ("(주)하이브", "㈜하이브", "주식회사 하이브",
                       "하이브(주)", "하이브 주식회사"):
            with self.subTest(corpus=corpus):
                self.assertTrue(reporter_matches(corpus, "하이브"))
                self.assertTrue(reporter_matches("하이브", corpus))

    def test_the_measured_corpus_pairs_now_match(self) -> None:
        pairs = (("삼성생명보험(주)", "삼성생명보험"),
                 ("주식회사 한국기업투자홀딩스", "한국기업투자홀딩스"),
                 ("(주)한국기업투자홀딩스", "한국기업투자홀딩스"),
                 ("삼성물산주식회사", "삼성물산"),
                 ("영풍정밀㈜", "영풍정밀"))
        for corpus, asked in pairs:
            with self.subTest(corpus=corpus):
                self.assertTrue(reporter_matches(corpus, asked))


class FamilyPreservationTests(unittest.TestCase):
    """Section 6: the behaviour that already worked must keep working."""

    def test_the_pension_family_is_unchanged(self) -> None:
        for value in ("국민연금공단", "국민연금기금"):
            with self.subTest(value=value):
                self.assertTrue(reporter_matches(value, "국민연금"))
                self.assertTrue(reporter_matches("국민연금", value))

    def test_family_suffixes_still_apply_on_canonical_keys(self) -> None:
        for suffix in ("공단", "기금", "조합", "법인", "회사"):
            with self.subTest(suffix=suffix):
                self.assertTrue(reporter_matches(f"대한{suffix}", "대한"))


class FalsePositiveTests(unittest.TestCase):
    """Section 8: what must never match, whatever it costs in recall."""

    def test_a_longer_name_is_a_different_holder(self) -> None:
        for corpus, asked in (("하이브아이엠", "하이브"),
                              ("(주)하이브아이엠", "하이브"),
                              ("SK스퀘어", "SK"),
                              ("삼성전자", "삼성"),
                              ("영풍정밀", "영풍"),
                              ("영풍정밀(주)", "영풍"),
                              ("BlackRock Fund Advisors", "BlackRock"),
                              ("국민연금공단", "국민")):
            with self.subTest(corpus=corpus, asked=asked):
                self.assertFalse(reporter_matches(corpus, asked))

    def test_a_name_beginning_or_ending_in_a_form_syllable_is_intact(self) -> None:
        """Fails if the implementation strips a bare syllable instead of a form."""

        for name in ("주성엔지니어링", "주연테크", "유미개발", "유한양행",
                     "대주", "신유", "한유"):
            with self.subTest(name=name):
                self.assertEqual(canonical_reporter_key(name), name.casefold())

    def test_two_names_differing_only_past_the_designator_stay_apart(self) -> None:
        self.assertFalse(reporter_matches("(주)영풍", "영풍정밀"))
        self.assertFalse(reporter_matches("영풍전자(주)", "영풍정밀"))


class PlaceholderTests(unittest.TestCase):
    """Section 5: an unstated holder is not an identity."""

    def test_placeholders_do_not_match_each_other(self) -> None:
        self.assertFalse(reporter_matches("-", "…"))
        self.assertFalse(reporter_matches("…", "-"))

    def test_a_placeholder_does_not_even_match_itself(self) -> None:
        self.assertFalse(reporter_matches("-", "-"))

    def test_a_placeholder_never_answers_a_real_constraint(self) -> None:
        self.assertFalse(reporter_matches("-", "하이브"))
        self.assertFalse(reporter_matches("하이브", "-"))

    def test_empty_and_none_match_nothing(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(canonical_reporter_key(value), "")
                self.assertFalse(reporter_matches(value, "하이브"))
                self.assertFalse(reporter_matches("하이브", value))


class NonCompanyReporterTests(unittest.TestCase):
    """Section 9: 99.8% of holders have no corp_code and must still work."""

    def test_funds_individuals_and_foreign_entities_match_themselves(self) -> None:
        for name in ("BlackRock Fund Advisors", "이수만", "방시혁",
                     "㈜우리은행 우리사주조합", "국민연금공단",
                     "T. Rowe Price Associates, Inc",
                     "CITIGROUP INC. - INTERNATIONAL SMALL CAP"):
            with self.subTest(name=name):
                self.assertTrue(reporter_matches(name, name))

    def test_an_employee_association_keeps_its_distinguishing_prefix(self) -> None:
        self.assertTrue(reporter_matches("㈜우리은행 우리사주조합",
                                         "우리은행 우리사주조합"))
        self.assertFalse(reporter_matches("㈜우리은행 우리사주조합",
                                          "㈜우리금융지주 우리사주조합"))


class ResolverComposerConsistencyTests(unittest.TestCase):
    """Section 10: one contract, asserted rather than assumed."""

    MATRIX = ("하이브", "(주)하이브", "㈜하이브", "주식회사 하이브", "하이브(주)",
              "하이브아이엠", "국민연금", "국민연금공단", "국민연금기금",
              "영풍", "영풍정밀", "(주)영풍", "-", "…", "", "BlackRock")

    def test_both_predicates_agree_on_every_pair(self) -> None:
        for left in self.MATRIX:
            for right in self.MATRIX:
                with self.subTest(left=left, right=right):
                    expected = reporter_matches(left, right)
                    self.assertEqual(_reporter_matches(left, right), expected)
                    self.assertEqual(_holding_reporter_matches(left, right),
                                     expected)


class RenderedReporterTextTests(unittest.TestCase):
    """The canonical key is for comparison; the answer shows the filing's words."""

    class _Event:
        def __init__(self, matches_query, reporter) -> None:
            self.matches_query = matches_query
            self.reporter = reporter
            self.evidence_chunk_ids = ("c",)

    class _Resolution:
        def __init__(self, events, requested, constraint=None) -> None:
            self.events = tuple(events)
            self.requested_fields = tuple(requested)
            self.reporter_constraint = constraint

    def test_a_rendered_event_keeps_the_raw_corpus_spelling(self) -> None:
        events = [self._Event(True, "(주)하이브")]

        reported = _reported_holding_events(
            self._Resolution(events, ["after_ratio"], "하이브")
        )

        self.assertEqual([e.reporter for e in reported], ["(주)하이브"])

    def test_a_designator_holder_is_no_longer_suppressed(self) -> None:
        """The composer guard used to blank this whole answer."""

        events = [self._Event(False, "(주)하이브")]

        reported = _reported_holding_events(
            self._Resolution(events, ["after_ratio"], "하이브")
        )

        self.assertEqual(len(reported), 1)


class OfflinePlanControlTests(unittest.TestCase):
    """Section 12: downstream compatibility only -- P0-D is untouched."""

    def test_a_counterparty_named_without_its_legal_form_now_matches(self) -> None:
        self.assertTrue(reporter_matches("(주)하이브", "하이브"))

    def test_the_issuer_is_never_confused_with_the_holder(self) -> None:
        self.assertFalse(reporter_matches("(주)하이브", "에스엠"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
