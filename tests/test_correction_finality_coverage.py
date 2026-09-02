"""When a correction notice written as prose may close a finality gap.

A Korean 정정신고 states the filing it corrects in a labelled header.  The DART
template renders that header as one-row tables, and the frozen notice reader
reads table cells; some filers' HTML emits the same labelled lines as prose
instead, and those filings reach the correction graph with no notice at all.
The chain then falls through to the title rule, which cannot tell one holding
report from the issuer's other filings, so the group stays ambiguous and the
holding finality artifact records no final member.

Reading that prose closes the gap, but a notice that becomes readable *after*
the rules already ran must not be allowed to relitigate what they settled.
These tests pin the three gates that keep it additive: a table notice is never
second-guessed, a chain that already has a proven final member is never
reopened, and a notice naming a target this corpus does not hold is not
admitted at all -- the last one because the frozen rules treat a known-missing
target as a reason to stop, which would *withdraw* an answer rather than add
one.

What happens once a notice is admitted is deliberately not re-tested here: it
is the frozen resolution rule, and a target it cannot tell apart from its
same-day twins still fails closed.
"""

import unittest

from app.reasoning.correction_graph import (
    NOTICE_SOURCE_TABLE,
    NOTICE_SOURCE_TEXT,
    CorrectionNotice,
    DisclosureRecord,
    build_correction_graph,
)
from scripts.build_correction_graph import recover_prose_notices


HOLDING = "주식등의대량보유상황보고서(일반)"
CORRECTED = "[기재정정]" + HOLDING


def _record(
    doc_id: str,
    rcept_dt: str,
    *,
    corp_code: str = "00126380",
    report_nm: str = HOLDING,
    is_correction: bool = False,
) -> DisclosureRecord:
    return DisclosureRecord(
        doc_id=doc_id,
        corp_code=corp_code,
        doc_group="holding",
        report_nm=report_nm,
        rcept_no=doc_id.rsplit("_", 1)[-1],
        rcept_dt=rcept_dt,
        doc_subtype=None,
        base_year=None,
        base_month=None,
        is_correction=is_correction,
    )


def _notice(doc_id: str, submitted_on: str | None, kind: str) -> CorrectionNotice:
    return CorrectionNotice(
        doc_id=doc_id,
        target_submitted_on=submitted_on,
        target_report_nm="주식등의 대량보유상황보고서",
        corrected_on=None,
        source_table_id="s0001" if kind == NOTICE_SOURCE_TEXT else "t0002",
        source_label="정정대상공시서류의최초제출일",
        source_kind=kind,
    )


#: One issuer, three earlier reports, and a correction that names the middle
#: one.  Without a notice the title rule sees three indistinguishable
#: candidates, which is the shape that leaves a real filing ambiguous.
ORIGINAL = "holding_20240802000389"
SIBLING_A = "holding_20240722000100"
SIBLING_B = "holding_20240815000200"
CORRECTION = "holding_20241025000530"


def _corpus() -> list[DisclosureRecord]:
    return [
        _record(SIBLING_A, "2024-07-22"),
        _record(ORIGINAL, "2024-08-02"),
        _record(SIBLING_B, "2024-08-15"),
        _record(CORRECTION, "2024-10-25", report_nm=CORRECTED, is_correction=True),
    ]


def _final_for(records, notices, doc_id):
    """The final member of ``doc_id``'s chain, on the artifact's own test."""

    graph = build_correction_graph(records, notices)
    members = graph.get_correction_chain(doc_id)
    if len(members) < 2:
        return None
    terminals = [member for member in members if member.is_latest]
    if len(terminals) != 1:
        return None
    if any(member.resolution_status != "resolved" for member in members):
        return None
    return terminals[0].doc_id


class ProseRecoveryGateTests(unittest.TestCase):
    def test_a_gap_is_closed_by_the_prose_notice(self) -> None:
        records = _corpus()
        prose = {CORRECTION: _notice(CORRECTION, "2024-08-02", NOTICE_SOURCE_TEXT)}

        self.assertIsNone(_final_for(records, {}, CORRECTION))

        recovered = recover_prose_notices(records, {}, prose)
        self.assertEqual(set(recovered), {CORRECTION})
        self.assertEqual(recovered[CORRECTION].source_kind, NOTICE_SOURCE_TEXT)
        self.assertEqual(_final_for(records, recovered, ORIGINAL), CORRECTION)

    def test_a_table_notice_is_never_second_guessed(self) -> None:
        """The filing said it in a table; its prose is not consulted at all."""

        records = _corpus()
        table = {CORRECTION: _notice(CORRECTION, "2024-08-15", NOTICE_SOURCE_TABLE)}
        prose = {CORRECTION: _notice(CORRECTION, "2024-08-02", NOTICE_SOURCE_TEXT)}

        recovered = recover_prose_notices(records, table, prose)
        self.assertEqual(recovered, dict(table))
        self.assertEqual(_final_for(records, recovered, SIBLING_B), CORRECTION)

    def test_a_chain_with_a_proven_final_member_is_not_reopened(self) -> None:
        """A later-readable notice does not revise what the rules settled."""

        records = _corpus()
        table = {CORRECTION: _notice(CORRECTION, "2024-08-15", NOTICE_SOURCE_TABLE)}
        settled = _final_for(records, table, SIBLING_B)

        # A second correction of the same chain, whose only notice is prose and
        # which the table pass has already placed by another rule.
        later = _record("holding_20241120000300", "2024-11-20",
                        report_nm=CORRECTED, is_correction=True)
        records.append(later)
        table[later.doc_id] = _notice(later.doc_id, "2024-10-25", NOTICE_SOURCE_TABLE)
        prose = {later.doc_id: _notice(later.doc_id, "2024-08-02", NOTICE_SOURCE_TEXT)}

        recovered = recover_prose_notices(records, table, prose)
        self.assertNotIn(later.doc_id, set(recovered) - set(table))
        self.assertEqual(recovered[later.doc_id].source_kind, NOTICE_SOURCE_TABLE)
        self.assertEqual(_final_for(records, recovered, SIBLING_B), later.doc_id)
        self.assertIsNotNone(settled)

    def test_a_target_this_corpus_does_not_hold_is_not_admitted(self) -> None:
        """Naming a missing original must not withdraw the weaker rules' answer."""

        records = _corpus()
        prose = {CORRECTION: _notice(CORRECTION, "2019-01-01", NOTICE_SOURCE_TEXT)}
        self.assertEqual(recover_prose_notices(records, {}, prose), {})

    def test_a_target_filed_later_than_the_correction_is_not_admitted(self) -> None:
        records = _corpus()
        prose = {CORRECTION: _notice(CORRECTION, "2025-06-01", NOTICE_SOURCE_TEXT)}
        records.append(_record("holding_20250601000900", "2025-06-01"))
        self.assertEqual(recover_prose_notices(records, {}, prose), {})

    def test_a_prose_notice_naming_no_date_is_not_admitted(self) -> None:
        records = _corpus()
        prose = {CORRECTION: _notice(CORRECTION, None, NOTICE_SOURCE_TEXT)}
        self.assertEqual(recover_prose_notices(records, {}, prose), {})

    def test_a_notice_for_a_document_outside_the_corpus_is_not_admitted(self) -> None:
        records = _corpus()
        prose = {"holding_absent_1": _notice("holding_absent_1", "2024-08-02",
                                             NOTICE_SOURCE_TEXT)}
        self.assertEqual(recover_prose_notices(records, {}, prose), {})

    def test_an_admitted_notice_still_fails_closed_on_same_day_twins(self) -> None:
        """Admission is not resolution: the frozen uniqueness rule still decides."""

        records = _corpus()
        records.append(_record("holding_20240802000777", "2024-08-02"))
        prose = {CORRECTION: _notice(CORRECTION, "2024-08-02", NOTICE_SOURCE_TEXT)}

        recovered = recover_prose_notices(records, {}, prose)
        self.assertEqual(set(recovered), {CORRECTION})
        self.assertIsNone(_final_for(records, recovered, ORIGINAL))


if __name__ == "__main__":
    unittest.main()
