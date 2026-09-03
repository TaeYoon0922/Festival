"""Whether the report a question compares against exists at all.

A holding question can be asked *relative to the report before*: ``직전 보고
대비 증감``, ``직전보고 보유주식수``.  Both need a previous state, and a filer's
**first** report on an issuer has none -- the corpus writes its ``직전 보고일``
and ``직전 보유주식수`` cells as a literal ``-``.  That is an absence, not a
zero, and not the current position restated.

Left alone, such a question answers.  The resolver reads no field name out of
``직전 보고 대비 증감``, so nothing is required of the evidence, and every served
report is reported as satisfying it -- including the filer's *next* filing,
whose previous state belongs to the target report rather than the other way
round.  The reader is then shown a first holding presented as a change from
nothing, beside a later report standing in for the predecessor that does not
exist.

So this producer answers one question and refuses to answer any other: does the
report this question is about record a previous state?  It selects nothing.  The
target report is the one the question's own exact reference date names, read
through the frozen date parser; the holder is the one the resolver already
constrained to; and the previous state is read from that report's own cells and
no other filing's.  When those cells carry a value, this producer declines
outright and the question keeps exactly the answer it had.  When they are there
and blank, it says so, with the row it read -- which is what lets the refusal
cite the first report instead of citing nothing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.reasoning.field_evidence import (
    DOMAIN_HOLDING,
    FieldEvidence,
    FieldReason,
    FieldStatus,
)
from app.reasoning.holding_date_intent import question_reference_date
from app.reasoning.holding_event_resolver import _FIELD_LABELS
from app.reasoning.holding_report_relative import (
    ROLE_CHANGE,
    ROLE_PREVIOUS,
    _AGAINST_PREVIOUS,
    _PREVIOUS_TERMS,
    parse as parse_report_relative,
)
from app.reasoning.holding_reporter import canonical_reporter_key, reporter_matches

#: The two fields the previous report is read for.  Canonical resolver names,
#: so a guard reading them sees the same field a resolved answer would have.
BASELINE_FIELDS = ("before_shares", "before_ratio")

#: The wording that makes a question relative to the report before.  The frozen
#: parser's own terms, imported rather than restated, so "which report" cannot
#: come to mean one thing where it is read and another here.
_PREVIOUS_REPORT_TERMS = frozenset((*_AGAINST_PREVIOUS, *_PREVIOUS_TERMS))

#: How a filing writes "there is nothing here".  Same cells the report index
#: treats as absent, so the two agree about what a first report looks like.
_BLANK_CELLS = frozenset({"", "-", "--", "―", "—", "–", "…", "N/A", "n/a", "해당사항없음"})


def previous_report_baseline_evidence(
    *,
    question: Any,
    resolution: Any,
    evidence_items: Sequence[Any],
) -> tuple[FieldEvidence, ...]:
    """What the target report says about the state before it, or nothing.

    An empty tuple means this producer has no finding -- the question is not
    about a previous report, the target report is not identified, or that report
    records a previous state after all.  Declining leaves the question exactly
    where it was answered before.
    """

    if not _asks_about_the_previous_report(question):
        return ()
    reference_date = question_reference_date(question)
    if not reference_date:
        # No single day is named, so no one report is the target and no filing
        # may be reported as having no predecessor.
        return ()

    events = _target_events(resolution, reference_date)
    if not events:
        return ()
    if any(
        getattr(event, field, None) is not None
        for event in events
        for field in BASELINE_FIELDS
    ):
        # The target report states where it came from.  Whether that answers the
        # question is the ordinary path's decision, not this producer's.
        return ()

    doc_ids = {_text(getattr(event, "doc_id", None)) for event in events}
    if len(doc_ids) != 1 or not next(iter(doc_ids)):
        # Several filings answer to one holder and one day.  Which of them the
        # question means is a selection, and this producer makes none.
        return ()

    event = events[0]
    common = {
        "domain": DOMAIN_HOLDING,
        "semantic_key": _semantic_key(event, reference_date),
        "corp_code": _text(getattr(event, "corp_code", None)) or None,
        "doc_id": next(iter(doc_ids)),
        "member_role": _text(getattr(event, "reporter", None)) or None,
    }
    served = _served_items(events, evidence_items)
    return tuple(
        finding
        for field in BASELINE_FIELDS
        for finding in _field_findings(field, served, common)
    )


def _asks_about_the_previous_report(question: Any) -> bool:
    """Whether the question reads a value from, or against, the report before.

    ``has_exact_date`` is asserted because the caller has already required one;
    it settles the selector, which this producer does not read.  What is read is
    the role, and the role is computed from the wording alone.
    """

    intent = parse_report_relative(
        str(question or ""),
        date_semantics={"role": "holding_reference"},
        has_exact_date=True,
    )
    if intent is None or intent.projection_role not in {ROLE_PREVIOUS, ROLE_CHANGE}:
        return False
    return str(intent.evidence or "") in _PREVIOUS_REPORT_TERMS


def _target_events(resolution: Any, reference_date: str) -> tuple[Any, ...]:
    """The events the question's own day and holder identify, or nothing.

    Every event carrying that reference date has to belong to one holder.  Two
    holders reporting the same day are two timelines, and saying that one of
    them has no predecessor would be saying it about the other as well.
    """

    events = tuple(getattr(resolution, "events", ()) or ())
    dated = [
        event
        for event in events
        if _digits(getattr(event, "reference_date", None)) == reference_date
    ]
    constraint = _text(getattr(resolution, "reporter_constraint", None))
    if constraint:
        dated = [
            event
            for event in dated
            if reporter_matches(getattr(event, "reporter", None), constraint)
        ]
    if not dated:
        return ()
    keys = {canonical_reporter_key(getattr(event, "reporter", None)) for event in dated}
    if len(keys) != 1 or not next(iter(keys)):
        return ()
    return tuple(dated)


def _served_items(events: Sequence[Any], evidence_items: Sequence[Any]) -> tuple[Any, ...]:
    """The served evidence the target events were resolved from."""

    wanted = {
        _text(chunk_id)
        for event in events
        for chunk_id in (getattr(event, "evidence_chunk_ids", ()) or ())
    }
    return tuple(
        item for item in evidence_items if _text(getattr(item, "chunk_id", None)) in wanted
    )


def _field_findings(
    field: str, served: Sequence[Any], common: Mapping[str, Any]
) -> tuple[FieldEvidence, ...]:
    """One reading per served row that carries this field's own cell.

    Every reading agrees -- the caller has already established that no target
    event resolved a value -- so several rows collapse to one state downstream
    rather than competing.  A field no served row carries at all is reported
    absent instead, which can be neither cited nor read as a number.
    """

    findings: list[FieldEvidence] = []
    for item in served:
        label, ref = _labelled_cell(field, item)
        if label is None:
            continue
        findings.append(
            FieldEvidence(
                field=field,
                status=FieldStatus.UNAVAILABLE,
                reason=FieldReason.NOT_STATED,
                chunk_id=_text(getattr(item, "chunk_id", None)) or None,
                table_id=_text(ref.get("table_id")) or None,
                row_start=_int(ref.get("row_start")),
                row_end=_int(ref.get("row_end", ref.get("row_start"))),
                **common,
            )
        )
    if findings:
        return tuple(findings)
    return (FieldEvidence(field=field, status=FieldStatus.MISSING, **common),)


def _labelled_cell(field: str, item: Any) -> tuple[str | None, Mapping[str, Any]]:
    """The blank cell this item states for ``field``, and where it sits.

    A label the projection does not carry proves nothing, and a label carrying a
    value is not this producer's to report -- the caller has already excluded
    that case, so a non-blank cell here means the projections disagree and the
    row is skipped rather than reconciled.
    """

    holding = dict(getattr(item, "holding", None) or {})
    fields = dict(holding.get("projection_fields") or {})
    refs = dict(holding.get("projection_field_refs") or {})
    for label in _FIELD_LABELS.get(field, ()):
        if label not in fields:
            continue
        if _text(fields.get(label)) not in _BLANK_CELLS:
            continue
        stated = [ref for ref in (refs.get(label) or ()) if isinstance(ref, Mapping)]
        source = stated[0] if stated else None
        if source is None:
            source = next(
                (
                    ref
                    for ref in (getattr(item, "source_refs", ()) or ())
                    if isinstance(ref, Mapping)
                ),
                {},
            )
        return label, source
    return None, {}


def _semantic_key(event: Any, reference_date: str) -> str:
    return "|".join(
        (
            _text(getattr(event, "corp_code", None)),
            canonical_reporter_key(getattr(event, "reporter", None)),
            reference_date,
            _text(getattr(event, "doc_id", None)),
        )
    )


def _digits(value: Any) -> str | None:
    found = "".join(character for character in str(value or "") if character.isdigit())
    return found[:8] if len(found) >= 8 else None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["BASELINE_FIELDS", "previous_report_baseline_evidence"]
