"""Compile one question into a deterministic ``MultiDocumentPlan`` (P0-C Step 3).

This module is a plan *compiler*, not an executor.  It reads a question and the
``QueryPlan`` that query understanding already produced, and emits the set
definitions an answer would need.  It opens no connection, imports no
repository, and fills no ``expected_ids``/``found_ids`` -- those are Step 4's
output, not Step 3's.

The engagement rule is deliberately strict.  Keyword gating scored 0% precision
on the real Gold60 set (3 hits, 3 false positives), so a plan is produced only
when *every* part of a set definition resolves:

    resolved corp_code
  + a family P0-B or the disclosure metadata can enumerate
  + explicit set/cardinality intent
  + a bounded, absolute date range
  + a date basis the question actually stated

Any one missing yields ``NOT_APPLICABLE`` with a reason.  Declining is the
correct outcome for 60 of the 60 real local questions; engaging on a guess would
change answers that are currently right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.reasoning.multi_document_plan import (
    DATE_FIELD_OPENED_AT,
    DATE_FIELD_RCEPT_DT,
    MEMBER_ROLE_CONTRACT,
    PLAN_NOT_APPLICABLE,
    REASON_MIXED_DATE_BASIS,
    REASON_NO_CORP_CODE,
    REASON_NO_DATE_RANGE,
    REASON_NO_FAMILY,
    REASON_NO_SET_INTENT,
    REASON_UNRESOLVED_DATE_BASIS,
    REASON_UNSUPPORTED_CALCULATION,
    REASON_UNSUPPORTED_TRUST_CONTRACT_BASIS,
    EvidenceSlot,
    MultiDocumentIntent,
    MultiDocumentPlan,
    SlotType,
)
from app.reasoning.query_plan import DateBasis


#: The one place ``QueryUnderstanding``'s event vocabulary is mapped onto P0-B's
#: ``event_family``.  The two vocabularies differ (understanding names the
#: *filing*, P0-B names the *lifecycle*), and both a conclusion and a
#: termination filing belong to the same lifecycle family.
_EVENT_TYPE_TO_FAMILY = {
    "supply_contract": "supply_contract",
    "contract_termination": "supply_contract",
    "treasury_share_trust_contract": "treasury_trust_contract",
    "treasury_share_trust_termination": "treasury_trust_contract",
}

#: The disclosure metadata each family occupies, for the Tier 2 receipt-date
#: plan.  Kept beside the family map so the two never drift apart.
_FAMILY_DOCUMENTS = {
    "supply_contract": ("exchange", "단일판매공급계약체결"),
    "treasury_trust_contract": ("major", None),
}

#: Ways a question asks for a *set* rather than a value.  Aggregation words are
#: deliberately absent: "평균", "총액", and "합계" mean arithmetic P0-C cannot do,
#: and in the real corpus they appear inside field labels such as 평균가동률 and
#: 권면총액, where they are not set intent at all.
_SET_INTENT_MARKERS = (
    "몇 건", "몇건", "건인가", "건수", "몇 개", "몇개",
    "모두", "전체", "전부",
    "목록", "리스트", "나열",
    "각각",
    "중에서", "가운데",
    "들을", "들은", "들이", "들 중",
)

#: A lifecycle question over a bounded set is itself set intent: "이 중 해지된
#: 것이 있는가" cannot be answered without listing the set first.
_LIFECYCLE_MARKERS = ("해지", "종료", "취소")
_EXISTENTIAL_MARKERS = ("있는가", "있나", "존재하는가", "존재하나", "있습니까", "없는가")

#: Arithmetic P0-C v1 does not do.  These are recognized only so the planner can
#: decline explicitly instead of enumerating and calling the result complete.
#: A typed numeric fact layer is a separate step.
_CALCULATION_MARKERS = (
    "합계", "총액", "총합", "평균", "증감률", "증가율", "감소율",
    "더 큰", "더 작은", "가장 큰", "가장 작은", "최댓값", "최솟값",
)

#: A question naming a contract without naming its family.  The official
#: reference question ("2025년에 체결한 주요 계약") is exactly this shape.
_BARE_CONTRACT_MARKERS = ("주요 계약", "주요계약", "계약")

#: Contract and finance families this corpus actually contains that are *not*
#: ``supply_contract``.  Grounded in the audited taxonomy rather than invented:
#: the 계약 filings here are 단일판매ㆍ공급계약 (supply), 자기주식취득신탁계약
#: (trust), and a long tail of 라이선스 / 공동연구 / 주주간 agreements filed as
#: 투자판단관련주요경영사항 -- plus the major-report financing families.  P0-B
#: models only supply and trust, so any of these means "not a supply contract",
#: and declining beats guessing.
_INCOMPATIBLE_FAMILY_MARKERS = (
    # trust -- P0-B's other family, with different date semantics entirely
    "신탁",
    # other contract kinds the corpus files under 투자판단관련주요경영사항
    "라이선스", "라이센스", "공동연구", "공동개발", "주주간", "기술이전",
    # financing / capital structure (주요사항보고서 families)
    "자금조달", "차입", "전환사채", "교환사채", "신주인수권", "조건부자본증권",
    "유상증자", "무상증자", "감자", "자기주식", "자사주",
    # corporate structure
    "합병", "분할", "주식교환", "영업양수", "영업양도",
    # investment
    "시설투자", "설비투자", "신규시설투자",
)

#: How the family was resolved.  Diagnostic only: it never changes execution,
#: but a plan that engaged through the fallback must be identifiable.
FAMILY_EXPLICIT_EVENT = "explicit_event_family"
FAMILY_BARE_CONTRACT_FALLBACK = "bare_contract_fallback"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class _SetDefinition:
    """The resolved parts of a set definition, or the reason one is missing."""

    corp_code: str | None = None
    event_family: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    date_basis: DateBasis = DateBasis.UNSPECIFIED
    family_resolution: str | None = None
    reason: str | None = None


class MultiDocumentPlanner:
    """Turn a question plus its ``QueryPlan`` into a ``MultiDocumentPlan``.

    Stateless and side-effect free.  Step 4 calls :meth:`plan` once and then
    executes the result; it must never re-parse the question text.
    """

    def plan(self, question: str, query_plan: Any) -> MultiDocumentPlan:
        text = _question_text(question, query_plan)

        # Arithmetic is checked first: an aggregation question may well name a
        # company, a family, and a year, and enumerating it would produce a
        # confidently complete plan for a question P0-C cannot answer.
        if _matches(text, _CALCULATION_MARKERS):
            return _declined(REASON_UNSUPPORTED_CALCULATION)

        intent = self._intent(text)
        if intent is None:
            return _declined(REASON_NO_SET_INTENT)

        definition = self._define_set(text, query_plan)
        if definition.reason is not None:
            return _declined(definition.reason)

        slots = self._slots(definition, intent)
        return MultiDocumentPlan(
            plan_type=intent.value,
            slots=slots,
            family_resolution=definition.family_resolution,
        )

    # ------------------------------------------------------------------ intent

    def _intent(self, text: str) -> MultiDocumentIntent | None:
        lifecycle = _matches(text, _LIFECYCLE_MARKERS) and _matches(
            text, _EXISTENTIAL_MARKERS
        )
        if lifecycle:
            return MultiDocumentIntent.ENUMERATION_PLUS_EVENT
        if _matches(text, _SET_INTENT_MARKERS):
            return MultiDocumentIntent.ENUMERATION
        return None

    # -------------------------------------------------------------- definition

    def _define_set(self, text: str, query_plan: Any) -> _SetDefinition:
        corp_code = getattr(query_plan, "corp_code", None)
        if not corp_code:
            # Either no company was named or several were.  P0-C never guesses a
            # company: the metadata resolver is the only authority.
            return _SetDefinition(reason=REASON_NO_CORP_CODE)

        # The basis is checked before the range. A question that names two
        # periods with two different bases usually resolves to no single range
        # either, and "mixed_date_basis" is the reason that actually explains
        # why -- reporting "no_date_range" would hide the real cause.
        basis = DateBasis(getattr(query_plan, "date_basis", DateBasis.UNSPECIFIED))
        if basis is DateBasis.MIXED:
            return _SetDefinition(reason=REASON_MIXED_DATE_BASIS)
        if basis is DateBasis.UNSPECIFIED:
            return _SetDefinition(reason=REASON_UNRESOLVED_DATE_BASIS)

        date_from, date_to = _bounded_range(getattr(query_plan, "period", None))
        if not (date_from and date_to):
            return _SetDefinition(reason=REASON_NO_DATE_RANGE)

        # Family last, because the bare-contract fallback below is only allowed
        # once the basis and the range are known to be exactly what it requires.
        family, source = self._resolve_family(text, query_plan, basis)
        if family is None:
            return _SetDefinition(reason=REASON_NO_FAMILY)

        # A trust filing states 계약체결 예정일자 -- a *planned* execution date --
        # and P0-B does not parse it: trust ``opened_at`` is derived from
        # 계약기간 시작일. The two coincide in every filing measured (43 of 43),
        # but that is an empirical coincidence, not the stored provenance, so
        # answering "체결한" from ``opened_at`` would be naming the wrong field.
        if family == "treasury_trust_contract" and basis is DateBasis.CONTRACT_DATE:
            return _SetDefinition(reason=REASON_UNSUPPORTED_TRUST_CONTRACT_BASIS)

        return _SetDefinition(
            corp_code=str(corp_code),
            event_family=family,
            date_from=date_from,
            date_to=date_to,
            date_basis=basis,
            family_resolution=source,
        )

    # ------------------------------------------------------------------ family

    def _resolve_family(
        self, text: str, query_plan: Any, basis: DateBasis
    ) -> tuple[str | None, str | None]:
        """Resolve the lifecycle family, with a narrow bare-contract fallback.

        The explicit vocabulary always wins.  ``QueryUnderstanding``'s ``_EVENTS``
        table is frozen -- widening it would move Gold60 routing -- so the
        fallback lives here, is plan-local, and never writes back to
        ``query_plan.event_type``.
        """

        explicit = _EVENT_TYPE_TO_FAMILY.get(
            str(getattr(query_plan, "event_type", "") or "")
        )
        if explicit is not None:
            return explicit, FAMILY_EXPLICIT_EVENT

        # The official reference question says "주요 계약", not "공급계약", and the
        # frozen vocabulary deliberately does not treat a bare 계약 as a supply
        # contract: that word also appears in periodic and correction questions.
        # Inside P0-C the surrounding constraints make it safe, but only when
        # every one of them holds.
        if basis is not DateBasis.CONTRACT_DATE:
            # A bare 계약 on the receipt axis is not evidence of a supply
            # contract, and the two axes disagree for 43.9% of them.
            return None, None
        if not _matches(text, _BARE_CONTRACT_MARKERS):
            return None, None
        if _matches(text, _INCOMPATIBLE_FAMILY_MARKERS):
            # The question names a different contract family. Declining is the
            # honest outcome: P0-B models neither, and guessing supply_contract
            # would answer a question nobody asked.
            return None, None
        return "supply_contract", FAMILY_BARE_CONTRACT_FALLBACK

    # ------------------------------------------------------------------- slots

    def _slots(
        self, definition: _SetDefinition, intent: MultiDocumentIntent
    ) -> tuple[EvidenceSlot, ...]:
        receipt = definition.date_basis is DateBasis.RECEIPT_DATE
        if receipt:
            # The receipt axis lives on ``disclosures``; the event timeline has
            # no receipt date of its own, so this is a Tier 2 plan even for a
            # family P0-B models.
            doc_group, doc_subtype = _FAMILY_DOCUMENTS[definition.event_family]
            enumerate_slot = EvidenceSlot(
                slot_id="documents",
                slot_type=SlotType.ENUMERATE_DOCUMENTS,
                corp_code=definition.corp_code,
                event_family=definition.event_family,
                doc_group=doc_group,
                doc_subtype=doc_subtype,
                date_field=DATE_FIELD_RCEPT_DT,
                date_from=definition.date_from,
                date_to=definition.date_to,
            )
        else:
            enumerate_slot = EvidenceSlot(
                slot_id="contracts",
                slot_type=SlotType.ENUMERATE_EVENTS,
                corp_code=definition.corp_code,
                event_family=definition.event_family,
                member_role=MEMBER_ROLE_CONTRACT,
                date_field=DATE_FIELD_OPENED_AT,
                date_from=definition.date_from,
                date_to=definition.date_to,
            )
        if intent is MultiDocumentIntent.ENUMERATION:
            return (enumerate_slot,)
        return (
            enumerate_slot,
            EvidenceSlot(
                slot_id="lifecycle",
                slot_type=SlotType.EVENT_STATE,
                corp_code=definition.corp_code,
                event_family=definition.event_family,
                depends_on=(enumerate_slot.slot_id,),
            ),
        )


# ------------------------------------------------------------------- helpers


def _question_text(question: str, query_plan: Any) -> str:
    raw = getattr(query_plan, "raw_query", None)
    return str(question or raw or "")


def _matches(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _declined(reason: str) -> MultiDocumentPlan:
    return MultiDocumentPlan(plan_type=PLAN_NOT_APPLICABLE, stop_reason=reason)


def _bounded_range(period: Any) -> tuple[str | None, str | None]:
    """Resolve the period into an absolute half-open ``[from, to)`` interval.

    A relative expression ("최근", "그동안") never reaches here as an absolute
    range, so it falls through as unbounded and the plan declines.
    """

    if period is None:
        return None, None
    from_date = getattr(period, "from_date", None)
    to_date = getattr(period, "to_date", None)
    if from_date and to_date and _ISO_DATE.match(from_date) and _ISO_DATE.match(to_date):
        # Query understanding states an inclusive end date; enumeration is
        # half-open so adjacent periods partition without overlapping.
        return from_date, _day_after(to_date)
    year = getattr(period, "year", None)
    quarter = getattr(period, "quarter", None)
    if isinstance(year, int) and not isinstance(year, bool):
        if isinstance(quarter, int) and not isinstance(quarter, bool):
            start_month = quarter * 3 - 2
            end_month = start_month + 3
            end_year = year + (1 if end_month > 12 else 0)
            end_month = end_month - 12 if end_month > 12 else end_month
            return f"{year:04d}-{start_month:02d}-01", f"{end_year:04d}-{end_month:02d}-01"
        return f"{year:04d}-01-01", f"{year + 1:04d}-01-01"
    return None, None


def _day_after(value: str) -> str:
    from datetime import date, timedelta

    parsed = date.fromisoformat(value) + timedelta(days=1)
    return parsed.isoformat()


__all__ = [
    "FAMILY_BARE_CONTRACT_FALLBACK",
    "FAMILY_EXPLICIT_EVENT",
    "MultiDocumentPlanner",
]
