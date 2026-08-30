"""What report a holding question is speaking relative to, and which of its fields.

A holding filing states three different things about the same holder: what they
hold now, what they held at the previous report, and the change between the two.
Korean expresses which one is wanted with a small closed set of phrases --
``이번 보고``, ``직전보고``, ``최신 보고``, ``직전 보고 대비`` -- and those phrases
answer two independent questions at once:

``selector``          *which* report the question is about;
``projection_role``   *which fields* of that report are wanted.

They are independent, and conflating them is the failure this module exists to
prevent.  ``최신 보고의 직전보고 보유비율`` selects the **latest** report and asks
for its **previous** ratio; the previous ratio is a field stored inside the
latest filing, not a different filing.  Reading ``직전보고`` as "the
second-newest document" would answer with another filing's current state --
a different holder-state on a different date, presented as the requested one.

This module only *reads intent*.  It never selects a filing, and it deliberately
cannot: proving "the latest report for this issuer and this reporter" requires
enumerating every holding report for that pair, which the deterministic corpus
path cannot currently do -- the disclosure table carries no reporter and no
holding reference date, and the reporter is only known from a projection inside
a chunk that retrieval had to return first.  So an unanchored ``최신``/``현재``
question is recorded as *asking* for the latest and marked not executable, and
the frozen ambiguity machinery keeps presenting alternatives rather than picking
one.  Naming the intent is what makes the gap visible; guessing a filing would
hide it.

Date semantics are not re-derived here.  Which calendar axis a date belongs to
-- holding reference date versus receipt date -- is already decided by the
frozen period parser, and this module reads that decision rather than repeating
it, so the two ontologies cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

#: Which report the question is about.
SELECTOR_LATEST = "latest"
SELECTOR_EXACT_REFERENCE_DATE = "exact_reference_date"
SELECTOR_EXACT_RECEIPT_DATE = "exact_receipt_date"
#: A deictic "this report" that only means something once a report is already in
#: hand.  Standing alone it names nothing.
SELECTOR_SELECTED_CONTEXT = "selected_context"
SELECTOR_NONE = "none"

#: Which fields of the selected report are wanted.
ROLE_CURRENT = "current"
ROLE_PREVIOUS = "previous"
ROLE_CHANGE = "change"

#: Selectors that cannot be executed from the deterministic corpus path, because
#: proving them needs reporter-aware enumeration that does not exist yet.
_UNPROVABLE_SELECTORS = frozenset({SELECTOR_LATEST, SELECTOR_SELECTED_CONTEXT})

#: The period parser's own name for a date that is a holding reference date, and
#: for one that is a receipt date.  Imported meanings, not new ones.
_REFERENCE_ROLE = "holding_reference"
_RECEIPT_ROLE = "receipt"

#: "이번 보고", "금번 보고서" -- deictic: the report already under discussion.
_DEICTIC_TERMS = ("이번보고", "금번보고", "본보고", "당보고")
#: "최신 보고", "최근 보고서", "가장 최근 보고" -- an ordering over reports.
_LATEST_TERMS = ("최신보고", "최근보고", "가장최근보고", "마지막보고", "최종보고")
#: "직전보고", "이전 보고서", "종전 보고" -- the report before the selected one.
_PREVIOUS_TERMS = ("직전보고", "이전보고", "종전보고", "전보고서")
#: Bare "현재" / "현재 기준" with no date attached asks for the newest known
#: state, which is the same unprovable ordering as 최신.
_NOW_TERMS = ("현재", "현시점", "지금")

#: "직전 보고 대비", "전기 대비" -- the previous report as a *baseline* for a
#: change, rather than as the source of the reported values.
_AGAINST_PREVIOUS = (
    "직전보고대비", "직전대비", "이전보고대비", "종전대비", "직전보고서대비",
)
#: Words that make the requested quantity the difference itself.
_CHANGE_TERMS = ("증감", "증가", "감소", "변동", "차이", "늘었", "줄었", "달라")


@dataclass(frozen=True)
class ReportRelativeIntent:
    """Which report a holding question means, and which of its fields."""

    selector: str
    projection_role: str
    #: Whether the answer moves as the corpus gains newer filings.  An explicit
    #: date is stable; "latest" is not, and saying so is part of the answer's
    #: meaning.
    dynamic: bool
    evidence: str | None = None

    @property
    def executable(self) -> bool:
        """Whether the selector can be resolved to one report deterministically.

        ``False`` does not mean "guess": it means the question named a report
        this system cannot yet prove, so nothing may be selected on its behalf.
        """

        return self.selector not in _UNPROVABLE_SELECTORS

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "projection_role": self.projection_role,
            "dynamic": self.dynamic,
            "executable": self.executable,
            "evidence": self.evidence,
        }


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _first_hit(compact: str, terms: tuple[str, ...]) -> str | None:
    for term in terms:
        if term in compact:
            return term
    return None


def _date_role(date_semantics: Mapping[str, Any] | None) -> str | None:
    if not isinstance(date_semantics, Mapping):
        return None
    return str(date_semantics.get("role") or "") or None


def parse(
    question: str,
    *,
    date_semantics: Mapping[str, Any] | None = None,
    has_exact_date: bool = False,
) -> ReportRelativeIntent | None:
    """Read which report the question means and which of its fields.

    ``date_semantics`` and ``has_exact_date`` come from the frozen period
    parser, so an explicit date keeps whichever calendar axis that parser
    already assigned to it.  Returns ``None`` when the question says nothing
    about report relativity at all, which must stay distinct from saying
    "current fields of an unnamed report".
    """

    compact = _compact(question)

    # Role first: what is being asked for is independent of which report.
    against_previous = _first_hit(compact, _AGAINST_PREVIOUS)
    previous_term = _first_hit(compact, _PREVIOUS_TERMS)
    if against_previous:
        # "직전 보고 대비 …" makes the previous report a baseline. Whether the
        # answer is the delta or the previous value itself depends on the noun
        # that follows, and a bare 증감률 is genuinely ambiguous between a
        # percentage-point difference and a relative rate -- so it is not
        # resolved here.
        role = ROLE_CHANGE if _first_hit(compact, _CHANGE_TERMS) else ROLE_PREVIOUS
        evidence = against_previous
    elif previous_term:
        role = ROLE_PREVIOUS
        evidence = previous_term
    else:
        role = ROLE_CURRENT
        evidence = None

    # Selector second.  An explicit date outranks a relative word: "2024년 3월
    # 14일 보고서의 직전보고 비율" names its report and asks for that report's
    # previous ratio.
    date_role = _date_role(date_semantics)
    if has_exact_date and date_role == _RECEIPT_ROLE:
        selector, dynamic = SELECTOR_EXACT_RECEIPT_DATE, False
    elif has_exact_date and date_role == _REFERENCE_ROLE:
        selector, dynamic = SELECTOR_EXACT_REFERENCE_DATE, False
    elif has_exact_date:
        selector, dynamic = SELECTOR_EXACT_REFERENCE_DATE, False
    else:
        latest_term = _first_hit(compact, _LATEST_TERMS)
        deictic_term = _first_hit(compact, _DEICTIC_TERMS)
        now_term = _first_hit(compact, _NOW_TERMS)
        if latest_term:
            selector, dynamic = SELECTOR_LATEST, True
            evidence = evidence or latest_term
        elif deictic_term:
            # "이번 보고" points at a report the question never names. It is not
            # a synonym for the latest one, and turning it into one would answer
            # about a filing the asker never mentioned.
            selector, dynamic = SELECTOR_SELECTED_CONTEXT, False
            evidence = evidence or deictic_term
        elif now_term:
            selector, dynamic = SELECTOR_LATEST, True
            evidence = evidence or now_term
        elif previous_term or against_previous:
            # "직전보고 보유비율" with nothing to be previous *to*.
            selector, dynamic = SELECTOR_SELECTED_CONTEXT, False
        else:
            return None

    return ReportRelativeIntent(
        selector=selector,
        projection_role=role,
        dynamic=dynamic,
        evidence=evidence,
    )
