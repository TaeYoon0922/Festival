"""Deterministic query understanding with safe lexical-query normalization."""

from __future__ import annotations

import re
from calendar import monthrange
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from app.parsing.metadata_filtered_retrieval import extract_metadata_filters
from app.reasoning import holding_report_relative
from app.reasoning.contract_lifecycle import (
    LIFECYCLE_OUTCOME_KEY,
    lifecycle_outcome_requested,
)
from app.reasoning.holding_reporter import canonical_reporter_key
from app.reasoning.query_plan import QueryPeriod, QueryPlan


CompanyResolver = Callable[[str], Mapping[str, Any] | None]


_FINANCIAL_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("매출액", ("매출액", "매출", "영업수익")),
    ("영업이익", ("영업이익", "영업손익")),
    ("당기순이익", ("당기순이익", "순이익", "당기순손익")),
    ("자산총계", ("자산총계", "총자산")),
    ("부채총계", ("부채총계", "총부채")),
    ("자본총계", ("자본총계", "총자본")),
    ("주당순이익", ("주당순이익", "EPS")),
)
_FINANCIAL_METRIC_SEARCH_TERMS = {
    "당기순이익": ("분기순이익", "연결분기순이익", "당기순손익", "분기순손익"),
    "자산총계": ("자산 총계", "자산 총 계", "자 산 총 계"),
    "부채총계": ("부채 총계", "부채 총 계", "부 채 총 계"),
    "자본총계": ("자본 총계", "자본 총 계", "자 본 총 계"),
}

_EVENTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("capital_increase", ("유상증자",), "major"),
    ("convertible_bond", ("전환사채",), "major"),
    ("treasury_share_disposal", ("자기주식처분", "자사주처분"), "major"),
    (
        "treasury_share_trust_termination",
        (
            "자기주식취득신탁계약해지",
            "자기주식 취득 신탁계약 해지",
            "신탁계약해지",
            "신탁계약 해지",
        ),
        "major",
    ),
    (
        "treasury_share_trust_contract",
        (
            "자기주식취득신탁계약체결",
            "자기주식 취득 신탁계약 체결",
            "신탁계약체결",
            "신탁계약 체결",
            "자기주식취득신탁계약",
            "자기주식 취득 신탁계약",
        ),
        "major",
    ),
    (
        "write_down_contingent_capital_security",
        (
            "상각형조건부자본증권",
            "상각형 조건부자본증권",
            "조건부자본증권",
            "자본증권발행",
            "자본증권 발행",
        ),
        "major",
    ),
    ("spin_off", ("회사분할", "분할신설", "분할비율"), "major"),
    ("merger", ("합병", "흡수합병"), "major"),
    (
        "supply_contract",
        (
            "단일판매", "공급계약", "수주계약",
            # Ways a question names a supply contract or its life. Bare field
            # names such as "계약금액" are deliberately absent: they also appear
            # in periodic and correction questions, which must not route here.
            "계약체결", "계약상대", "계약의최초", "계약이후",
        ),
        "exchange",
    ),
    (
        "contract_termination",
        ("계약해지", "공급계약해지", "해지된계약", "해지계약", "계약이해지"),
        "exchange",
    ),
    ("facility_investment", ("시설투자", "신규시설투자"), "exchange"),
)
_EVENT_DOC_SUBTYPES = {
    "facility_investment": "신규시설투자등",
}
_EVENT_SECTION_BOOSTS: dict[str, dict[str, float]] = {
    "treasury_share_trust_termination": {
        "자기주식취득 신탁계약 해지 결정": 1.0,
        "자기주식취득신탁계약해지결정": 1.0,
    },
    "write_down_contingent_capital_security": {
        "상각형 조건부자본증권 발행결정": 1.0,
        "상각형조건부자본증권발행결정": 1.0,
    },
}

_SECTION_BOOSTS: dict[str, dict[str, float]] = {
    "매출액": {
        "매출 및 수주상황": 1.0,
        "고객과의 계약에서 생기는 수익": 1.0,
        "고객과의 계약으로 인한 수익": 0.98,
        "포괄손익계산서": 0.98,
        "손익계산서": 0.95,
        "수익의 인식": 0.90,
        "영업수익": 0.88,
    },
    "영업이익": {
        "포괄손익계산서": 1.0,
        "손익계산서": 0.98,
        "영업이익": 0.95,
        "영업손익": 0.95,
    },
    "당기순이익": {
        "포괄손익계산서": 1.0,
        "손익계산서": 0.98,
        "당기순이익": 0.95,
        "당기순손익": 0.95,
    },
    "자산총계": {
        "재무상태표": 1.0,
        "첨부연결재무제표": 1.0,
        "첨부재무제표": 0.98,
        "자산총계": 0.95,
    },
    "부채총계": {
        "재무상태표": 1.0,
        "첨부연결재무제표": 1.0,
        "첨부재무제표": 0.98,
        "부채총계": 0.95,
    },
    "자본총계": {
        "재무상태표": 1.0,
        "첨부연결재무제표": 1.0,
        "첨부재무제표": 0.98,
        "자본총계": 0.95,
    },
    "주당순이익": {
        "주당이익": 1.0,
        "포괄손익계산서": 0.85,
        "손익계산서": 0.82,
    },
    "holding_ratio": {"보유 주식": 1.0, "대량보유": 0.95, "보유비율": 0.90},
    "holding_shares": {"보유 주식": 1.0, "대량보유": 0.95, "주식수": 0.90},
}

_PERIODIC_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "business_product",
        (
            r"주요\s*제품",
            r"제품\s*(?:및|과)\s*서비스",
            r"사업\s*(?:설명|내용|개요)",
            r"주요\s*사업",
            r"장비\s*사업",
            r"(?:기술|제품)\s*설명",
        ),
    ),
    (
        "listing_history",
        (
            r"상장\s*일(?:자)?",
            r"(?:유가증권|코스피|코스닥|증권|주식)\s*시장\s*상장",
            r"(?:설립|상장).{0,12}(?:상장|이력)",
        ),
    ),
    (
        "merger_history",
        (
            r"합병\s*기일",
            r"흡수\s*합병",
            r"합병\s*이력",
            r"회사의\s*중요한\s*변동",
        ),
    ),
)

_PERIODIC_SECTION_BOOSTS: dict[str, dict[str, float]] = {
    "business_product": {
        "사업의 개요": 1.0,
        "주요 제품 및 서비스": 1.0,
        "사업의 내용": 0.82,
        "회사의 개요": 0.65,
    },
    "listing_history": {
        "회사의 개요": 1.0,
    },
    "merger_history": {
        "회사의 연혁": 1.0,
        "회사의 중요한 변동": 0.95,
        "합병": 0.85,
    },
}


class QueryUnderstanding:
    """Rule-first parser with an optional backend company resolver interface."""

    def __init__(
        self,
        company_aliases: Mapping[str, Iterable[str]] | None = None,
        *,
        company_resolver: CompanyResolver | None = None,
    ) -> None:
        self._company_aliases = {
            str(alias): _canonical_names(names)
            for alias, names in (company_aliases or {}).items()
            if str(alias).strip()
        }
        self._company_resolver = company_resolver

    def understand(
        self,
        query: str,
        *,
        top_k: int = 10,
        section_path: str | None = None,
        correction_policy: str | None = None,
    ) -> QueryPlan:
        raw_query = query.strip()
        if not raw_query:
            raise ValueError("query must not be empty")

        aliases = {alias: set(names) for alias, names in self._company_aliases.items()}
        resolved = self._company_resolver(raw_query) if self._company_resolver else None
        corp_codes: tuple[str, ...] = ()
        if resolved:
            canonical = str(
                resolved.get("corp_name") or resolved.get("listed_name") or ""
            ).strip()
            if canonical:
                for alias in (canonical, resolved.get("listed_name")):
                    if alias:
                        aliases.setdefault(str(alias), set()).add(canonical)
            corp_code = str(resolved.get("corp_code") or "").strip()
            if corp_code:
                corp_codes = (corp_code,)

        extracted = extract_metadata_filters(raw_query, aliases)
        companies = tuple(extracted["companies"])
        if not companies:
            inferred = _infer_company_prefix(raw_query)
            if inferred:
                companies = (inferred,)

        # The bounded two-company ownership rule must read the comparison
        # firewall before it can classify the task.  Report comparison payloads
        # still wait for period parsing below; the firewall itself depends only
        # on the question and canonical company mentions.
        company_mentions = _company_mentions(raw_query, companies, aliases)
        comparison_frame = _comparison_frame(raw_query, companies, company_mentions)

        financial_metric, metric_evidence = _find_financial_metric(raw_query)
        event_type, event_evidence, event_route = _find_event(raw_query)
        holding_metric, holding_evidence = _find_holding_metric(raw_query)
        ownership_intent = False
        ownership_metric: str | None = None
        ownership_evidence: str | None = None
        ownership_metric_evidence: str | None = None
        if holding_metric is None:
            (
                ownership_intent,
                ownership_metric,
                ownership_evidence,
                ownership_metric_evidence,
            ) = _find_bounded_ownership_intent(
                raw_query,
                companies,
                company_mentions,
                comparison_frame=comparison_frame,
                conflicting_intent=bool(
                    event_type
                    or financial_metric
                    or extracted.get("doc_group") not in {None, "holding"}
                ),
            )
        # The named-holder shapes stay in precedence order: an explicitly
        # labelled reporter, then the one this question relates to its issuer.
        reporter = (
            _find_reporter(raw_query)
            or _natural_holding_reporter(
                raw_query, company_mentions, holding_metric=holding_metric
            )
            or _reporter_first_holding_reporter(
                raw_query, company_mentions, holding_metric=holding_metric
            )
        )
        # Syntax's half of the role question: which surface stood in the actor
        # slot.  Whether that surface names a holder of this issuer is a corpus
        # question, and validation is where the corpus is readable.
        actor_candidate = (
            directed_holder_candidate(raw_query, company_mentions[0])
            if ownership_evidence == "company_acquires_company_shares"
            and len(company_mentions) == 1
            else None
        )

        if holding_metric or ownership_intent:
            task_type = "holding_change"
            metric = holding_metric or ownership_metric
        elif event_type:
            task_type = "corporate_event"
            metric = None
        elif financial_metric:
            task_type = "financial_metric"
            metric = financial_metric
        else:
            task_type = "disclosure_lookup"
            metric = None

        routes, route_confidence, route_evidence = _routes(
            raw_query=raw_query,
            extracted=extracted,
            task_type=task_type,
            event_type=event_type,
            event_route=event_route,
            event_evidence=event_evidence,
        )
        periodic_intent, periodic_intent_evidence = _find_periodic_intent(raw_query)
        if not _periodic_intent_allowed(task_type, event_type, routes):
            periodic_intent, periodic_intent_evidence = None, None
        period, period_spans, mentioned_years, date_semantics = _period_from_query(
            raw_query,
            task_type=task_type,
            routes=routes,
        )
        # Which report the question speaks relative to, and which of that
        # report's fields. Read from the period parser's own date decision so
        # reference-date and receipt-date wording keep the axis it assigned.
        report_relative = (
            holding_report_relative.parse(
                _latest_report_wording(raw_query),
                date_semantics=date_semantics,
                has_exact_date=bool(period.from_date and period.from_date == period.to_date),
            )
            if "holding" in routes
            else None
        )
        basis, basis_evidence, basis_span = _basis_from_query(raw_query)
        parsed_correction, correction_confidence, correction_evidence = (
            _correction_from_query(raw_query)
        )
        correction_intent, correction_intent_evidence = _correction_intent_from_query(
            raw_query
        )
        operation = _operation_from_query(
            raw_query,
            task_type=task_type,
            event_type=event_type,
            correction_intent=correction_intent,
        )
        if correction_policy is not None:
            parsed_correction = correction_policy
            correction_confidence = 1.0
            correction_evidence = "caller_override"
        comparison = _comparison_from_query(
            raw_query, companies, mentioned_years, company_mentions
        )

        subtype = extracted["doc_subtype"]
        if not subtype:
            subtype = _EVENT_DOC_SUBTYPES.get(event_type or "")
        if subtype:
            explicit_report = any(
                term in re.sub(r"\s+", "", raw_query)
                for term in ("분기보고서", "반기보고서", "사업보고서")
            )
            route_confidence["doc_subtype"] = 0.99 if explicit_report else 0.90
            route_evidence["doc_subtype"] = (
                extracted["doc_subtype_evidence"] or event_evidence
            )
        if basis != "unspecified":
            route_confidence["basis"] = 0.99
            route_evidence["basis"] = basis_evidence
        if parsed_correction in {"corrected_only", "original_only"}:
            route_confidence["is_correction"] = correction_confidence
            route_evidence["is_correction"] = correction_evidence
        elif correction_evidence:
            route_confidence["is_correction"] = correction_confidence
            route_evidence["is_correction"] = correction_evidence
        if section_path:
            route_confidence["section_path"] = 1.0
            route_evidence["section_path"] = "caller_override"

        structured_spans: list[tuple[int, int]] = [*period_spans]
        structured_spans.extend(_company_spans(raw_query, companies, aliases))
        if basis_span:
            structured_spans.append(basis_span)
        structured_spans.extend(_metadata_term_spans(raw_query, extracted, period.period_type))
        if task_type == "financial_metric":
            structured_spans.extend(_period_column_spans(raw_query))
        structured_spans.extend(_correction_spans(raw_query, correction_evidence))
        lexical_query = _normalize_lexical_query(
            raw_query,
            structured_spans,
            metric=metric,
            metric_evidence=(
                metric_evidence or holding_evidence or ownership_metric_evidence
            ),
            event_type=event_type,
            event_evidence=event_evidence,
        )

        fiscal_years = mentioned_years if period.is_fiscal else ()
        annual_preferred = (
            task_type == "financial_metric"
            and period.period_type == "fiscal_year"
            and period.quarter is None
        )
        return QueryPlan(
            query=lexical_query,
            raw_query=raw_query,
            companies=companies,
            corp_codes=corp_codes,
            years=fiscal_years,
            period=period,
            task_type=task_type,
            metric=metric,
            event_type=event_type,
            disclosure_route=routes,
            basis=basis,
            reporter=reporter,
            correction_policy=parsed_correction,
            comparison=comparison,
            doc_subtype=subtype,
            section_path=section_path,
            date_basis=_date_basis_from_query(raw_query),
            section_boosts=_section_boosts(metric, periodic_intent, event_type),
            route_confidence=route_confidence,
            route_evidence=route_evidence,
            top_k=top_k,
            evidence={
                "mentioned_years": list(mentioned_years),
                "period_type": period.period_type,
                "date_semantics": date_semantics,
                "report_preference": "annual" if annual_preferred else None,
                "company_resolved": bool(resolved),
                "company_resolution_attempted": self._company_resolver is not None,
                "metric": (
                    metric_evidence or holding_evidence or ownership_metric_evidence
                ),
                "event_type": event_evidence,
                "operation": operation,
                "holding_ownership_intent": ownership_evidence,
                HOLDING_ACTOR_CANDIDATE_KEY: actor_candidate,
                "periodic_intent": periodic_intent,
                "periodic_intent_evidence": periodic_intent_evidence,
                "correction_intent": correction_intent,
                "correction_intent_evidence": correction_intent_evidence,
                "comparison_frame": comparison_frame,
                "holding_report_relative": (
                    report_relative.to_dict() if report_relative else None
                ),
                "structured_spans": [list(span) for span in sorted(set(structured_spans))],
                # Whether the question follows a contract forward, decided
                # once here so composition never re-reads the question.
                LIFECYCLE_OUTCOME_KEY: lifecycle_outcome_requested(raw_query),
            },
        )


def understand_query(
    query: str,
    company_aliases: Mapping[str, Iterable[str]] | None = None,
    *,
    company_resolver: CompanyResolver | None = None,
    top_k: int = 10,
) -> QueryPlan:
    return QueryUnderstanding(
        company_aliases, company_resolver=company_resolver
    ).understand(query, top_k=top_k)


def _find_financial_metric(query: str) -> tuple[str | None, str | None]:
    compact = re.sub(r"\s+", "", query).casefold()
    for canonical, aliases in _FINANCIAL_METRICS:
        match = next((alias for alias in aliases if alias.casefold() in compact), None)
        if match:
            return canonical, match
    return None, None


#: A contract named plainly, then asked what became of it.  The filing families
#: above are recognized by their own names -- ``공급계약``, ``계약해지`` -- but a
#: question about a contract's outcome need not repeat any of them: it names the
#: contract once and then asks about it.  ``<계약> ... 해지됐나`` and
#: ``<계약>의 최종 상태`` are contract-lifecycle questions written the way people
#: write them, and without this they reach retrieval as unrouted prose, where
#: the lifecycle expansion that would answer them declines for want of an event
#: type.
#:
#: Bounded on both sides.  The contract noun and the outcome have to be close
#: enough to be about each other -- a metric question that happens to mention a
#: termination elsewhere in the sentence is not this -- and the outcome
#: vocabulary is the lifecycle's own, never a field name.  ``계약금액`` and other
#: bare field nouns stay deliberately unmatched here, exactly as they are in
#: ``_EVENTS``: they appear in periodic and correction questions that must keep
#: their present routing.
_CONTRACT_NOUN = r"계약"
#: What a lifecycle question asks about the contract it just named.  Only
#: ``해지``: it is the one lifecycle word this corpus never also uses as a field
#: name.  ``종료`` and ``취소`` read the same way in a sentence but ``계약기간
#: 종료일`` is a date *stated inside* a contract filing, so accepting them turns
#: a single-document field question into a lifecycle question and sends it
#: looking for a termination that was never asked about.
_CONTRACT_OUTCOME = r"해지"
#: The two shapes, each anchored on the contract noun so the outcome cannot be
#: picked up from an unrelated clause.  At most a few characters of particle or
#: adverb may stand between them.
_CONTRACT_LIFECYCLE = re.compile(
    _CONTRACT_NOUN + r"[^\s]{0,6}?" + _CONTRACT_OUTCOME
)
#: ``계약의 최종 상태``: the outcome asked for by name rather than by predicate.
_CONTRACT_FINAL_STATE = re.compile(_CONTRACT_NOUN + r".{0,4}?최종상태")


def _find_event(query: str) -> tuple[str | None, str | None, str | None]:
    compact = re.sub(r"\s+", "", query).casefold()
    for event_type, aliases, route in _EVENTS:
        match = next((alias for alias in aliases if alias.casefold() in compact), None)
        if match:
            return event_type, match, route
    lifecycle = _contract_lifecycle_event(compact)
    if lifecycle:
        # The lifecycle family, not the termination filing: the question starts
        # from a contract and asks where it ended up, so the conclusion filing
        # is as much a part of the answer as the termination is.
        return "supply_contract", lifecycle, "exchange"
    return None, None, None


def _contract_lifecycle_event(compact: str) -> str | None:
    """A contract whose outcome this question asks about, or nothing.

    Read only after every named filing family has declined, so a question that
    names its own family keeps the routing that family already gives it.
    """

    for pattern in (_CONTRACT_LIFECYCLE, _CONTRACT_FINAL_STATE):
        match = pattern.search(compact)
        if match:
            return match.group(0)
    return None


def _find_holding_metric(query: str) -> tuple[str | None, str | None]:
    compact = re.sub(r"[ㆍ·・‧\s]+", "", query)
    if any(term in compact for term in ("보유비율", "지분율", "보유주식비율")):
        return "holding_ratio", "보유비율/지분율"
    if any(term in compact for term in ("보유주식", "주식수", "보유수량")):
        return "holding_shares", "보유주식/주식수/보유수량"
    if "지분" in compact and any(term in compact for term in ("변동", "증감", "보유")):
        return "holding_ratio", "지분+변동/증감/보유"
    report_terms = (
        "소유상황보고서",
        "대량보유상황보고서",
        "대량보유상황",
        "임원주요주주",
        "특정증권등소유",
        "주식등의대량보유",
    )
    matched_report = next((term for term in report_terms if term in compact), None)
    if matched_report:
        # Report-title questions still need the holding pipeline / latest_holding.
        return "holding_shares", matched_report
    if "특수관계자" in compact and any(
        term in compact
        for term in ("보고서", "주주", "임원", "소유상황", "대량보유")
    ):
        return "holding_shares", "특수관계자"
    return None, None


#: A directed acquisition question names its acquirer in subject position.  The
#: acquirer is matched as the *whole* subject and as a single token: anything
#: else in front of the recognized company is structure this stage cannot bound.
#: A leading legal form is part of the token, and ``canonical_reporter_key``
#: removes it, so ``(주)가상인수자``, ``㈜가상인수자`` and ``가상인수자`` reduce to one acquirer.
_HOLDER_SUBJECT = re.compile(
    r"\s*(?P<holder>[0-9A-Za-z가-힣(㈜㈝][0-9A-Za-z가-힣()㈜㈝&.\-]{1,31})"
    r"\s*[이가]\s*"
)
#: Words that fill the subject slot without naming anybody.  Compared against
#: the canonical holder key, never against raw text.
_NON_HOLDER_SUBJECTS = frozenset(
    {
        "회사", "기업", "법인", "당사", "본사", "상대방",
        "취득자", "매수자", "인수자", "보유자", "주주",
        "어디", "누구", "그곳", "이곳", "어느곳",
    }
)
#: A date or a quantity standing where an acquirer should.
_BARE_PERIOD_SUBJECT = re.compile(r"\d+(?:년|월|일|분기|기|주)?")
#: How a holding question attaches the holder it asks about to the issuer.  Only
#: the relational bridges such a question actually uses, plus bare whitespace for
#: the commonest shape (``<issuer> <holder>의 최근 보고``).  Whitespace is safe
#: here only because the holder must also carry a possessive or subject particle
#: below; without that requirement it would match any following word.
_REPORTER_BRIDGE = r"(?:\s*에\s*(?:대|관)(?:한|해서?|하여)\s*|\s+)"
#: A role noun may name the holder's standing before naming the holder itself.
#: It is consumed, never returned: the holder is the name that follows it.
_REPORTER_ROLE = r"(?:(?:최대|대|주요|지배)\s*주주\s+)?"
#: A leading legal form is written as its own word.  It is kept in the surface
#: rather than stripped: canonical identity is ``canonical_reporter_key``'s job,
#: and this parser must not do that work a second time.
_REPORTER_LEGAL_FORM = r"(?:(?:주식회사|유한회사|유한책임회사|재단법인|사단법인)\s+)?"
#: One name token, with no internal whitespace.  The only multi-word holder
#: accepted is a legal form followed by one token, which is why the legal form
#: above is a separate group rather than part of this character class.
_REPORTER_TOKEN = r"[0-9A-Za-z가-힣(㈜㈝][0-9A-Za-z가-힣()㈜㈝&.\-]{0,31}"
#: The holder must stand in a possessive or subject relation to the report.
#: Object and topic particles are deliberately absent: with 을/를/은/는 accepted,
#: ``보유주식수는`` and ``보유비율을`` would themselves read as holders.  The
#: trailing space ends the surface, so the boundary is a token boundary rather
#: than a guess about how much of the sentence belongs to the holder.
_REPORTER_PARTICLE = r"\s*(?:의|이|가)\s"
#: ``<issuer><bridge>[role ]<holder><particle>``, anchored at the end of the
#: issuer mention.  The anchoring is the whole safety argument: the holder is
#: read out of the issuer's own relational structure and never searched for in
#: the sentence, so metrics, dates, document nouns and question suffixes
#: elsewhere in the question can never be reached.
_NATURAL_REPORTER = re.compile(
    _REPORTER_BRIDGE
    + _REPORTER_ROLE
    + r"(?P<holder>"
    + _REPORTER_LEGAL_FORM
    + _REPORTER_TOKEN
    + r")"
    + _REPORTER_PARTICLE
)
#: The inverse surface order of ``_NATURAL_REPORTER``: the filer is the
#: sentence subject, and the issuer follows the reporting verb it governs --
#: ``<holder>가 ... 보고한 <issuer> 보유주식수``.  This expression reads only
#: the leading subject.  The relation between it and the issuer is checked
#: separately below rather than hidden inside an unbounded regex.
_REPORTER_FIRST_SUBJECT = re.compile(
    r"\s*(?P<holder>"
    + _REPORTER_LEGAL_FORM
    + _REPORTER_TOKEN
    + r")\s*(?:이|가)\s*"
)
#: A reporting verb must close the bridge immediately before the issuer.  The
#: bounded endings describe a filer reporting an issuer's holding state; a
#: generic verb elsewhere in the sentence proves no role.
_REPORTER_FIRST_RELATION = re.compile(r"(?:보고|신고)\s*(?:한|하는)\s*$")
#: Another grammatical actor inside the bridge makes the leading subject's
#: role ambiguous.  This also rejects coordinated filers instead of choosing
#: the first one.
_REPORTER_FIRST_NESTED_ACTOR = re.compile(
    _REPORTER_TOKEN + r"\s*(?:이|가|와|과)\s+"
)
#: The issuer must be followed by the holding object the report is about.  It
#: may carry a possessive particle, but a company followed by unrelated prose
#: is not enough merely because holding vocabulary appears later.
_REPORTER_FIRST_HOLDING_OBJECT = re.compile(
    r"^\s*(?:의\s*)?(?:보유\s*주식|주식|지분|보유\s*비율)"
)
#: Surfaces that fill the holder slot without naming a holder: a role, a bare
#: document qualifier, or the requested quantity itself.  ``최대주주의 보유비율``
#: names no holder, and reading the role noun as one would answer about nobody.
#: The filing *names* themselves are not listed: they are a productive family,
#: so ``_names_a_filing`` derives them instead of enumerating them.  Compared
#: against the canonical key, never against raw text.
_NON_REPORTER_SURFACES = _NON_HOLDER_SUBJECTS | frozenset(
    {
        "최대주주", "대주주", "주요주주", "지배주주", "소액주주", "특수관계자",
        "지분", "지분율", "주식", "보유주식", "보유주식수", "보유비율", "보유수량",
    }
)
#: The head noun that makes a phrase name a *filing* rather than a filer.  A
#: filing name always ends in one of these; a holder's name does not.
_REPORT_HEAD_NOUNS = ("보고서", "보고", "공시", "신고서", "신고")
#: What a holding filing's own name is built out of, ahead of that head.  Same
#: disclosure vocabulary ``_find_holding_metric`` already reads report titles
#: with, kept as its own constant because this asks a different question of it:
#: not "is this a holding question" but "is this surface the document".
_REPORT_QUALIFIERS = (
    "대량보유상황", "대량보유", "소유상황", "특정증권등소유", "특정증권등",
    "임원주요주주", "주요주주", "임원", "주식등의", "주식등", "특수관계자", "소유",
)
#: The words that make a report the newest one.  These are the modifiers inside
#: the frozen parser's own latest terms, which are written as one contiguous
#: ``<modifier><head>`` string -- so a question that names the document between
#: the two says the same thing in wording that parser cannot see.
_LATEST_REPORT_MODIFIERS = ("최신", "최근", "마지막", "최종")
#: ``최근 대량보유상황보고서`` -- a latest modifier, then this corpus's own holding
#: document vocabulary, then the report head.  At least one qualifier is
#: required, so ``최근 보고`` (already understood) is not touched and ``최근 계약``,
#: ``최근 사업보고서`` and ``최근 실적`` are not holding-report wording at all: the
#: qualifiers are the holding disclosure families and nothing else.
_LATEST_HOLDING_REPORT = re.compile(
    r"(?P<latest>" + "|".join(_LATEST_REPORT_MODIFIERS) + r")"
    r"\s*(?:(?:" + "|".join(_REPORT_QUALIFIERS) + r")\s*)+"
    r"(?P<head>" + "|".join(_REPORT_HEAD_NOUNS) + r")"
)
#: ``plan.evidence`` key carrying the actor a directed acquisition named, before
#: anything corpus-aware has confirmed that such a holder exists.  Read by
#: validation, which owns the identity question; never a reporter by itself.
HOLDING_ACTOR_CANDIDATE_KEY = "holding_actor_candidate"
#: What produced that candidate.  A consumer that cannot read this provenance
#: has no business acting on the candidate.
ACTOR_SOURCE_DIRECTED_HOLDER = "directed_holder_syntax"
#: An acquisition unit price named outright: the price word bound directly to
#: the acquisition word with nothing between them.  ``취득가액``, ``총 취득금액``
#: and a bare ``단가`` all fail this on purpose.
_ACQUISITION_UNIT_PRICE = re.compile(r"(?:취득|매수)단가")
#: A disposal price is a different field.  A source column can be labelled
#: 취득/처분단가, but only the row-level producer reads that header; a question
#: that asks for the disposal side is asking a different question.
_DISPOSAL_UNIT_PRICE = re.compile(r"(?:처분|매도|양도).{0,12}?단가")


def _find_bounded_ownership_intent(
    query: str,
    companies: tuple[str, ...],
    mentions: tuple[tuple[int, int, str], ...],
    *,
    comparison_frame: str | None,
    conflicting_intent: bool,
) -> tuple[bool, str | None, str | None, str | None]:
    """Recognize closed securities-ownership constructions.

    This classifier determines only the holding task family and, when the
    wording is explicit, its metric.  The canonical company tuple remains
    untouched: issuer/reporter direction belongs to validation and cannot be
    inferred from mention order here.

    Two canonical mentions are the ordinary case.  A directed acquisition is
    the one construction that also runs on a single mention, because the
    acquirer it names is a holding-report filer and the company universe is
    scoped to issuers -- see ``_directed_acquisition_from_unresolved_holder``.
    """

    if comparison_frame in {"cross_company", "uncertain"} or conflicting_intent:
        return False, None, None, None

    if len(companies) == 1 and len(mentions) == 1:
        return _directed_acquisition_from_unresolved_holder(query, mentions[0])

    if len(companies) != 2 or len(mentions) != 2:
        return False, None, None, None

    (_first_start, first_end, _first), (second_start, second_end, _second) = mentions
    gap = re.sub(r"\s+", "", query[first_end:second_start])
    tail = re.sub(r"\s+", "", query[second_end:])

    family: str | None = None
    acquisition_unit_price = (
        _directed_acquisition_unit_price(tail)
        if re.fullmatch(r"[이가]", gap)
        else None
    )
    if acquisition_unit_price:
        return (
            True,
            "acquisition_unit_price",
            "company_acquires_company_shares",
            acquisition_unit_price,
        )
    if re.fullmatch(r"[이가]보유한", gap) and re.match(
        r"^주식(?:은|는|이|가|을|를)?", tail
    ):
        family = "company_holds_company_shares"
    elif re.fullmatch(r"[이가]", gap) and re.match(
        r"^주식(?:은|는|이|가|을|를)?.*들고(?:있|계)", tail
    ):
        family = "company_has_company_shares"
    elif gap == "의" and re.match(r"^지분(?:은|는|이|가|을|를|율|비율)?", tail):
        family = "company_ownership_interest"

    if family is None:
        return False, None, None, None

    # Generic "얼마나" does not select an answer unit.  Only explicit count or
    # ratio vocabulary promotes the otherwise valid ownership intent to a
    # concrete holding metric.
    ratio = re.search(r"지분\s*비율|보유\s*비율|지분율|퍼센트|%", query)
    if ratio:
        return True, "holding_ratio", family, ratio.group(0)
    shares = re.search(r"몇\s*주|주식\s*수(?:량)?|보유\s*수량", query)
    if shares:
        return True, "holding_shares", family, shares.group(0)
    # The holding family is already established, so an explicit acquisition
    # unit-price noun phrase names the answer unit the same way 몇 주 and 지분율
    # do above.  It stays a noun phrase: the price word has to be bound to the
    # acquisition word, which is what separates it from 취득가액, 총 취득금액 and
    # a bare 단가.
    unit_price = _acquisition_unit_price_phrase(tail)
    if unit_price:
        return True, "acquisition_unit_price", family, unit_price
    return True, None, family, None


def _directed_acquisition_from_unresolved_holder(
    query: str,
    mention: tuple[int, int, str],
) -> tuple[bool, str | None, str | None, str | None]:
    """Recognize a directed acquisition whose acquirer no alias map can name.

    The company universe is issuer-scoped, and a holding report's filer is an
    asset manager, a fund, an individual or a foreign entity far more often
    than it is one of those issuers.  So a real directed acquisition question
    names one company this stage can canonicalize -- the issuer whose shares
    were bought -- and one it never will.  Demanding two canonical mentions
    asks this stage for corpus knowledge it does not have.

    The acquirer is therefore bounded by *shape*, never resolved: it is the
    whole subject of the sentence, and it is carried onward only as the
    candidate ``directed_holder_candidate`` builds.  No company, no reporter
    and no role is written from it, so issuer/reporter direction still belongs
    entirely to validation.
    """

    _start, end, _canonical = mention
    if directed_holder_candidate(query, mention) is None:
        return False, None, None, None
    evidence = _directed_acquisition_unit_price(re.sub(r"\s+", "", query[end:]))
    if not evidence:
        return False, None, None, None
    return (
        True,
        "acquisition_unit_price",
        "company_acquires_company_shares",
        evidence,
    )


def directed_holder_candidate(
    query: str,
    mention: tuple[int, int, str],
) -> dict[str, Any] | None:
    """The acquirer a directed acquisition names, as syntax alone can see it.

    Deliberately a *candidate*.  Syntax can prove that a surface stood in the
    actor slot of a directed acquisition; it cannot prove that the surface is a
    holder this corpus knows, and it must not claim to.  ``resolved`` stays
    false until something that can read the corpus says otherwise.
    """

    start, _end, _canonical = mention
    if start == 0:
        return None
    # ``_comparison_frame`` needs two canonical mentions to bind an operator,
    # and an unresolved acquirer gives it only one.  Comparative or set
    # vocabulary anywhere in the question therefore declines outright rather
    # than passing an unchecked comparison into the holding lane.
    if _has_comparison_vocabulary(query):
        return None
    subject = _HOLDER_SUBJECT.fullmatch(query[:start])
    if subject is None:
        return None
    surface = subject.group("holder").strip()
    key = canonical_reporter_key(surface)
    if (
        len(key) < 2
        or key in _NON_HOLDER_SUBJECTS
        or _BARE_PERIOD_SUBJECT.fullmatch(key)
    ):
        return None
    return {
        "surface": surface,
        "reporter_key": key,
        "source": ACTOR_SOURCE_DIRECTED_HOLDER,
        "resolved": False,
    }


def _has_comparison_vocabulary(query: str) -> bool:
    """Any wording ``_comparison_frame`` would weigh, read without operands."""

    compact = re.sub(r"\s+", "", query)
    return any(
        term in compact
        for group in (
            _EXPLICIT_COMPARISON,
            _CHOICE_MARKERS,
            _ENUMERATION_MARKERS,
            _COMPARATIVE_PREDICATES,
            _COMPARISON_OPERATORS,
        )
        for term in group
    )


def _directed_acquisition_unit_price(tail: str) -> str | None:
    """Match only a directed shares-acquisition request for a unit price."""

    if not re.match(r"^주식(?:은|는|이|가|을|를)?", tail):
        return None
    # Query understanding activates only the acquisition side before the
    # row-level producer interprets any source header.
    if _DISPOSAL_UNIT_PRICE.search(tail):
        return None
    match = re.search(r"(?:취득|매수).{0,24}?단가", tail)
    return match.group(0) if match else None


def _acquisition_unit_price_phrase(tail: str) -> str | None:
    """The acquisition unit price named as a noun phrase, if the tail names it.

    Used where the holding family is already established by the construction
    itself, so no acquisition verb has to carry the reading.  Only adjacency
    counts: the price word bound to the acquisition word and nothing else.
    """

    if _DISPOSAL_UNIT_PRICE.search(tail):
        return None
    match = _ACQUISITION_UNIT_PRICE.search(tail)
    return match.group(0) if match else None


def _find_periodic_intent(query: str) -> tuple[str | None, str | None]:
    for intent, patterns in _PERIODIC_INTENTS:
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return intent, match.group(0)
    return None, None


def _periodic_intent_allowed(
    task_type: str,
    event_type: str | None,
    routes: tuple[str, ...],
) -> bool:
    if task_type == "disclosure_lookup":
        return not routes or "periodic" in routes
    return event_type == "merger" and "periodic" in routes


def _section_boosts(
    metric: str | None, periodic_intent: str | None, event_type: str | None = None
) -> dict[str, float]:
    merged = dict(_SECTION_BOOSTS.get(metric or "", {}))
    for section, weight in _PERIODIC_SECTION_BOOSTS.get(
        periodic_intent or "", {}
    ).items():
        merged[section] = max(merged.get(section, 0.0), weight)
    for section, weight in _EVENT_SECTION_BOOSTS.get(event_type or "", {}).items():
        merged[section] = max(merged.get(section, 0.0), weight)
    return merged


def _routes(
    *,
    raw_query: str,
    extracted: Mapping[str, Any],
    task_type: str,
    event_type: str | None,
    event_route: str | None,
    event_evidence: str | None,
) -> tuple[tuple[str, ...], dict[str, float], dict[str, Any]]:
    candidates: dict[str, float] = {}
    evidence: dict[str, Any] = {}
    compact = re.sub(r"\s+", "", raw_query)
    extracted_route = extracted.get("doc_group")
    extracted_evidence = extracted.get("doc_group_evidence")

    if task_type == "holding_change":
        candidates["holding"] = 0.99
        evidence["disclosure_route.holding"] = extracted_evidence or "holding metric"
    elif task_type == "financial_metric":
        confidence = 0.99 if any(
            term in compact for term in ("사업보고서", "분기보고서", "반기보고서")
        ) else 0.85
        candidates["periodic"] = confidence
        evidence["disclosure_route.periodic"] = extracted_evidence or "financial metric"
    elif event_type == "merger" and extracted_route is None:
        candidates.update({"major": 0.58, "periodic": 0.42})
        evidence["disclosure_route.major"] = event_evidence
        evidence["disclosure_route.periodic"] = "merger context can appear in periodic reports"
    elif event_route:
        candidates[event_route] = 0.98
        evidence[f"disclosure_route.{event_route}"] = event_evidence
    elif extracted_route:
        candidates[str(extracted_route)] = 0.96
        evidence[f"disclosure_route.{extracted_route}"] = extracted_evidence

    ordered = tuple(
        route for route, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
    )
    confidence = {
        f"disclosure_route.{route}": value for route, value in candidates.items()
    }
    return ordered, confidence, evidence


#: Wording that ties two dates together as the ends of one span.  A range says
#: so; dates that merely both appear do not.
_DATE_RANGE_MARKERS = ("부터", "까지", "사이", "~")


def _states_one_date_range(
    query: str, dates: Sequence[tuple[str | None, tuple[int, int]]]
) -> bool:
    """Whether these dates are the two ends of one span the question stated.

    Several dates in one question are usually several *anchors*, not one
    period: "A의 <date> 계약과 B의 <date> 계약 중 어느 쪽이 큰가" names one date
    per item and no span at all.  Reading the first two as ``from``/``to``
    invents a range the asker never asked for -- and when the earlier item
    happens to carry the later date, it invents an inverted one, which is not a
    period any more.

    So a span has to be claimed rather than inferred.  Exactly two dates, said
    to be a span, in the order a span runs.  Anything else keeps its dates and
    leaves the period to the rest of this parser, which is what the itemized
    lanes downstream read anyway.
    """

    if len(dates) != 2:
        # Three anchors are not two ends.  Two ends is what a span has.
        return False
    (start, _start_span), (end, _end_span) = dates
    if not start or not end:
        return False
    compact = re.sub(r"\s+", "", query)
    if not any(marker in compact for marker in _DATE_RANGE_MARKERS):
        return False
    # A span runs forwards.  A backwards pair is not silently reordered -- the
    # question did not say which end was meant, and guessing is how the wrong
    # window gets searched.
    return start <= end


def _period_from_query(
    query: str,
    *,
    task_type: str,
    routes: tuple[str, ...],
) -> tuple[
    QueryPeriod,
    list[tuple[int, int]],
    tuple[int, ...],
    dict[str, Any],
]:
    years_with_spans = [
        (int(match.group(1)), match.span())
        for match in re.finditer(
            r"(?<!\d)(20\d{2})(?:\s*년|\s*회계연도)?(?:\s*(?:에|의|부터|까지))?(?!\d)",
            query,
        )
    ]
    years = tuple(sorted({year for year, _ in years_with_spans}))
    compact_query = re.sub(r"\s+", "", query)
    if len(years) == 2 and "부터" in compact_query and "까지" in compact_query:
        start_year, end_year = years
        if 0 <= end_year - start_year <= 20:
            years = tuple(range(start_year, end_year + 1))
    spans = [span for _, span in years_with_spans]
    quarter_match = re.search(r"(?<!\d)([1-4])\s*분기", query)
    quarter = int(quarter_match.group(1)) if quarter_match else None
    if quarter_match:
        spans.append(quarter_match.span())

    dates = [
        (_iso_date(match), match.span())
        for match in re.finditer(
            r"(20\d{2})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})(?:일)?",
            query,
        )
    ]
    date_role, date_marker = _date_semantic_role(query, task_type)
    if _states_one_date_range(query, dates):
        selected_spans = [span for _, span in dates[:2]]
        if date_role != "holding_reference":
            spans.extend(selected_spans)
        period_type = (
            "holding_reference_range"
            if date_role == "holding_reference"
            else "date_range"
        )
        return (
            QueryPeriod(
                from_date=dates[0][0],
                to_date=dates[1][0],
                period_type=period_type,
            ),
            [] if date_role == "holding_reference" else spans,
            years,
            _date_semantics(date_role, date_marker, [value for value, _ in dates[:2]]),
        )

    year = years[0] if len(years) == 1 else None
    compact = compact_query
    if date_role == "receipt" and year is not None:
        if dates:
            value, date_span = dates[0]
            spans.append(date_span)
            return (
                QueryPeriod(
                    year=year,
                    from_date=value,
                    to_date=value,
                    period_type="receipt_date",
                ),
                spans,
                years,
                _date_semantics(date_role, date_marker, [value]),
            )
        year_month = _single_year_month(query, excluded_spans=[span for _, span in dates])
        if year_month is not None:
            month_year, month, month_span = year_month
            spans.append(month_span)
            start = f"{month_year:04d}-{month:02d}-01"
            end = f"{month_year:04d}-{month:02d}-{monthrange(month_year, month)[1]:02d}"
            return (
                QueryPeriod(
                    year=month_year,
                    from_date=start,
                    to_date=end,
                    period_type="receipt_date",
                ),
                spans,
                years,
                _date_semantics(date_role, date_marker, [start, end]),
            )
        period = _whole_year_period(year, "receipt_date")
        return (
            period,
            spans,
            years,
            _date_semantics(date_role, date_marker, [period.from_date, period.to_date]),
        )
    if date_role == "holding_reference" and task_type == "holding_change":
        if dates:
            value = dates[0][0]
            return (
                QueryPeriod(
                    year=year,
                    from_date=value,
                    to_date=value,
                    period_type="holding_reference_date",
                ),
                [],
                years,
                _date_semantics(date_role, date_marker, [value]),
            )
        if year is not None:
            period = _whole_year_period(year, "holding_reference_year")
            return (
                period,
                [],
                years,
                _date_semantics(
                    date_role, date_marker, [period.from_date, period.to_date]
                ),
            )
        return (
            QueryPeriod(period_type="holding_reference"),
            [],
            years,
            _date_semantics(date_role, date_marker, []),
        )
    if year is not None and task_type == "holding_change":
        period = _whole_year_period(year, "holding_reference_year")
        return (
            period,
            [],
            years,
            _date_semantics(
                "holding_reference",
                "holding_change",
                [period.from_date, period.to_date],
            ),
        )
    if task_type == "financial_metric" or "periodic" in routes:
        return (
            QueryPeriod(
                year=year,
                quarter=quarter,
                period_type=(
                    "fiscal_quarter"
                    if quarter
                    else "fiscal_year"
                    if years
                    else "latest_valid_periodic"
                ),
            ),
            spans,
            years,
            _date_semantics("fiscal", None, [str(year)] if year else []),
        )
    if year is not None:
        return (
            QueryPeriod(year=year, period_type="reference_year"),
            spans,
            years,
            _date_semantics("reference", None, [str(year)]),
        )
    if quarter is not None:
        return (
            QueryPeriod(quarter=quarter, period_type="fiscal_quarter"),
            spans,
            years,
            _date_semantics("fiscal", None, []),
        )
    if task_type == "holding_change":
        return (
            QueryPeriod(period_type="latest_holding"),
            spans,
            years,
            _date_semantics(None, None, []),
        )
    if task_type == "corporate_event":
        if _latest_event_requested(query):
            return (
                QueryPeriod(period_type="latest_event"),
                spans,
                years,
                _date_semantics(None, None, []),
            )
        return (
            QueryPeriod(),
            spans,
            years,
            _date_semantics(None, None, []),
        )
    return QueryPeriod(), spans, years, _date_semantics(None, None, [])


def _latest_event_requested(query: str) -> bool:
    compact = re.sub(r"\s+", "", query)
    return any(
        term in compact
        for term in (
            "최근",
            "최신",
            "가장최근",
            "최근공시",
            "최근접수",
            "공시한",
            "공시된",
            "접수된",
        )
    )



#: Markers that say *which* date a period expression refers to.  Matched only in
#: a short window immediately after the period expression, because Korean
#: attaches this modifier right there ("2025년에 체결한 ...").  A marker further
#: away is describing something else -- in "2025년에 체결한 계약 중 공시 이후
#: 해지된 것", the 공시 belongs to the termination clause, not to the year.
_DATE_BASIS_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "contract_date",
        ("체결한", "체결된", "체결하", "체결일", "계약체결", "계약일", "수주한", "수주일", "맺은"),
    ),
    (
        "receipt_date",
        ("공시된", "공시한", "공시일", "공시", "접수된", "접수한", "접수일", "접수",
         "제출된", "제출한", "제출일", "제출"),
    ),
    ("period_start", ("시작한", "시작된", "시작일", "시작", "개시한", "개시된", "개시")),
)

#: How far after a period expression a basis marker may sit. Deliberately tight.
_DATE_BASIS_WINDOW = 10

_PERIOD_EXPRESSION = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*년(?:\s*\d{1,2}\s*월)?")


def _date_basis_from_query(query: str) -> str:
    """Which real-world date the question's period expression refers to.

    Returns one of ``contract_date``/``receipt_date``/``period_start`` when
    exactly one basis modifies the period expressions, ``mixed`` when two
    different bases each modify one, and ``unspecified`` when none does.

    This never guesses.  "2025년 계약" stays ``unspecified`` rather than being
    promoted to ``contract_date``, because the two select different documents and
    the question did not say which was meant.
    """

    text = query or ""
    found: list[str] = []
    for match in _PERIOD_EXPRESSION.finditer(text):
        window = text[match.end() : match.end() + _DATE_BASIS_WINDOW]
        # Earliest marker wins: the nearest modifier is the one that binds.
        best: tuple[int, str] | None = None
        for basis, markers in _DATE_BASIS_MARKERS:
            for marker in markers:
                index = window.find(marker)
                if index >= 0 and (best is None or index < best[0]):
                    best = (index, basis)
        if best is not None:
            found.append(best[1])
    distinct = set(found)
    if len(distinct) > 1:
        return "mixed"
    if len(distinct) == 1:
        return found[0]
    return "unspecified"

def _date_semantic_role(query: str, task_type: str) -> tuple[str | None, str | None]:
    receipt = re.search(
        r"(?:공시(?:한|된|일|시점)?|접수(?:한|된|일)?|제출(?:한|된|일)?)",
        query,
    )
    if receipt:
        return "receipt", receipt.group(0)
    if task_type == "holding_change":
        reference = re.search(
            r"(?:현재|기준일|기준|변동일|보유일)",
            query,
        )
        if reference:
            return "holding_reference", reference.group(0)
        if re.search(r"(?<!\d)20\d{2}(?!\d)", query):
            return "holding_reference", "holding_change_date"
    return None, None


def _date_semantics(
    role: str | None,
    marker: str | None,
    values: Sequence[str | None],
) -> dict[str, Any]:
    return {
        "role": role,
        "marker": marker,
        "values": [value for value in values if value],
    }


def _single_year_month(
    query: str,
    *,
    excluded_spans: Sequence[tuple[int, int]],
) -> tuple[int, int, tuple[int, int]] | None:
    for match in re.finditer(
        r"(20\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*(?:월)?"
        r"(?!\s*(?:일|[.\-/])\s*\d)",
        query,
    ):
        if any(
            match.start() < excluded_end and excluded_start < match.end()
            for excluded_start, excluded_end in excluded_spans
        ):
            continue
        year = int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12:
            return year, month, match.span()
    return None


def _whole_year_period(year: int, period_type: str) -> QueryPeriod:
    return QueryPeriod(
        year=year,
        from_date=f"{year:04d}-01-01",
        to_date=f"{year:04d}-12-31",
        period_type=period_type,
    )


def _iso_date(match: re.Match[str]) -> str:
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def _basis_from_query(query: str) -> tuple[str, str | None, tuple[int, int] | None]:
    consolidated = re.search(r"연결\s*(?:기준|재무제표|실적)?", query)
    if consolidated and "연결대상회사" not in consolidated.group(0):
        return "consolidated", consolidated.group(0), consolidated.span()
    standalone = re.search(r"(?:별도|개별)\s*(?:기준|재무제표|실적)?", query)
    if standalone:
        return "standalone", standalone.group(0), standalone.span()
    return "unspecified", None, None


#: Asking for the whole correction history of one report. Checked first because
#: these phrasings also contain the original and latest markers.
_CORRECTION_HISTORY_TERMS = (
    "정정이력", "정정내역", "정정경위", "정정경과", "변경이력", "변경내역",
    "어떻게정정", "어떤정정", "최초공시부터", "최초부터최종", "정정된내역",
    "정정사항을시간순", "정정이어떻게",
)
#: Asking for the final valid version of one report.
_CORRECTION_LATEST_TERMS = (
    "최종정정", "최종기준", "최종공시", "최종본", "정정후기준",
    "현재정정기준", "최종정정기준", "가장최근정정", "최신정정",
)
#: Asking for the report as it was first filed.
_CORRECTION_ORIGINAL_TERMS = (
    "최초공시", "원공시", "정정전", "최초제출", "최초로공시", "원본공시",
)
#: Naming the before state and the after state together: "정정 전후",
#: "정정 전과 정정 후", "정정 전 대비 정정 후".  That is a history question, not
#: a question about the original alone, but every one of these phrasings
#: contains the literal "정정전" that ``_CORRECTION_ORIGINAL_TERMS`` matches once
#: whitespace is stripped, so the pair has to be read before that scan runs.
#:
#: Written as structure rather than as a list of phrasings because the pair is
#: productive: the "전" marker anchored on 정정, an optional connector that
#: coordinates the two sides and may repeat the 정정 anchor, then the "후"
#: marker.  The window is deliberately narrow -- only a connector may sit
#: between the two markers -- so an unrelated noun starting with 후 cannot close
#: a pair, and "정정 전 금액" stays the original question it is.  Whitespace is
#: significant here and is not stripped first: "정정 전후" fuses the two markers
#: into one compound, while the space in "정정 전 후속조치" separates them.
_CORRECTION_PAIR = re.compile(
    r"정정\s*전"
    r"(?:\s*(?:과|와|랑|이랑|하고|및|에서|대비|[,·/~\-])\s*(?:정정\s*)?|)"
    r"후"
    # The "후" has to end its own token, so "후속"/"후반" is not an after marker.
    r"(?=[\s\W]|$|[은는이가을를에의로도만와과랑으])"
)


def _correction_intent_from_query(query: str) -> tuple[str | None, str | None]:
    """Which document of a correction chain the question is about.

    This is separate from ``correction_policy``: the policy says how to filter
    and rank the documents already retrieved, while the intent says whether the
    rest of the chain has to be fetched at all.  History is tested first because
    a history question mentions both the first and the final filing.
    """

    compact = re.sub(r"\s+", "", query)
    for intent, terms in (
        ("history", _CORRECTION_HISTORY_TERMS),
        ("latest", _CORRECTION_LATEST_TERMS),
        ("original", _CORRECTION_ORIGINAL_TERMS),
    ):
        for term in terms:
            if term in compact:
                if intent == "original":
                    # The term matched on the compacted query, where a
                    # before/after pair collapses into the original marker.
                    # Only the pair can rescue it, and only from the spacing the
                    # question actually used.
                    pair = _CORRECTION_PAIR.search(query)
                    if pair is not None:
                        return "history", pair.group(0)
                return intent, term
    return None, None


def _correction_from_query(query: str) -> tuple[str, float, str | None]:
    compact = re.sub(r"\s+", "", query)
    if any(term in compact for term in ("정정제외", "원공시만", "최초공시만")):
        return "original_only", 0.99, "정정 제외/원공시만"
    if any(term in compact for term in ("정정공시만", "정정본만", "정정만")):
        return "corrected_only", 0.99, "정정 공시만"
    intent, evidence = _correction_intent_from_query(query)
    if intent == "original":
        # "최초 공시한 ...": the question names the original, so corrections of
        # it must not stand in for it.
        return "original_only", 0.99, evidence
    if "정정" in compact:
        return "latest_preferred", 0.85, "정정"
    return "latest_preferred", 0.0, None


#: Postpositional comparison operators.  Korean attaches these to the *left*
#: operand, so what precedes one says whether it compares two companies or a
#: company against a period or a metric.  That is why the operator is resolved
#: structurally instead of by blacklisting the words it can follow.
_COMPARISON_OPERATORS = ("대비", "보다")
#: Connectors that coordinate two company mentions into one operand list.
#: Bare whitespace is deliberately absent: two names side by side name a
#: subject rather than coordinate it, and reading that as coordination would
#: claim every keyword-style two-company holding question.
_COMPANY_CONNECTOR = re.compile(r"^\s*(?:와|과|랑|이랑|하고|및|,|·|/)\s*$")
#: A particle may sit between a company mention and an operator.
_OPERATOR_GAP = re.compile(r"^\s*[은는이가의을를와과]?\s*$")
#: Asking which member of a named set the answer is.
_CHOICE_MARKERS = ("중어디", "중누가", "중어느", "어느쪽", "어느회사", "중더", "중가장")
#: Asking for each company separately.  Not arithmetic comparison, but equally
#: unsafe to reinterpret as one issuer and one reporter.
_ENUMERATION_MARKERS = ("각각", "각회사", "각사", "둘다", "양사", "양쪽")
#: Explicit comparison nouns.
_EXPLICIT_COMPARISON = ("비교", "차이")
#: Comparative predicates.  Never sufficient alone -- see ``_comparison_frame``.
_COMPARATIVE_PREDICATES = (
    "더많", "더높", "더적", "더낮", "더큰", "더작", "더크",
    "가장많", "가장높", "가장큰", "가장작", "가장적",
)
#: References that bind a comparison to a period or a prior filing rather than
#: to another company.  Matched against the whitespace-stripped query so
#: "직전 보고 대비" and "직전보고 대비" are one case.
_TEMPORAL_ANCHORS = (
    "전년", "전기", "전분기", "전년동기", "직전보고", "직전공시", "직전분기",
    "직전연도", "직전사업", "작년", "지난해", "예년", "종전", "동기",
    "이전보다", "이전공시", "이전과", "변동전후", "변경전후", "추이", "변화",
)
_TEMPORAL_YEAR = re.compile(r"(?:19|20)\d{2}년?(?:보다|대비)")


def _company_mentions(
    query: str,
    companies: tuple[str, ...],
    aliases: Mapping[str, set[str]],
) -> tuple[tuple[int, int, str], ...]:
    """Where each canonical company is named, in reading order.

    ``companies`` is sorted rather than ordered by mention, so positional
    structure has to be recovered from the text.  Longer surface forms win an
    overlap, and each canonical company is kept once so a repeated name cannot
    look like a second operand.
    """

    surface: dict[str, str] = {}
    for company in companies:
        surface[company] = company
    for alias, canonical_names in aliases.items():
        for canonical in canonical_names:
            if canonical in companies:
                surface.setdefault(alias, canonical)
    found: list[tuple[int, int, str]] = []
    for term in sorted(surface, key=len, reverse=True):
        canonical = surface[term]
        for match in re.finditer(re.escape(term), query, re.IGNORECASE):
            start, end = match.span()
            if any(start < other_end and other_start < end
                   for other_start, other_end, _ in found):
                continue
            found.append((start, end, canonical))
    found.sort()
    mentions: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for start, end, canonical in found:
        if canonical in seen:
            continue
        seen.add(canonical)
        mentions.append((start, end, canonical))
    return tuple(mentions)


def _operator_binds_companies(
    query: str,
    mentions: tuple[tuple[int, int, str], ...],
) -> bool:
    """Does a comparison operator take a company as its left operand?

    A company name followed by ``대비``/``보다`` compares companies;
    ``직전 보고 대비``, ``자기자본 대비`` and ``작년보다`` compare a period or a
    metric.  The difference is which token the operator attaches to, not which
    words appear elsewhere in the sentence.
    """

    for operator in _COMPARISON_OPERATORS:
        for match in re.finditer(re.escape(operator), query):
            start = match.start()
            if any(
                end <= start and _OPERATOR_GAP.fullmatch(query[end:start])
                for _mention_start, end, _canonical in mentions
            ):
                return True
    return False


def _companies_are_coordinated(
    query: str,
    mentions: tuple[tuple[int, int, str], ...],
) -> bool:
    """Are two company mentions joined into a single operand list?"""

    return any(
        _COMPANY_CONNECTOR.fullmatch(query[first_end:second_start])
        for (_first_start, first_end, _first), (second_start, _second_end, _second)
        in zip(mentions, mentions[1:])
    )


def _comparison_frame(
    query: str,
    companies: tuple[str, ...],
    mentions: tuple[tuple[int, int, str], ...],
) -> str | None:
    """Whether the question relates several named companies to each other.

    This is a **firewall** signal, not an execution signal: ``cross_company``
    says that reading one company as the disclosure issuer and another as its
    reporter would be unsafe, and says nothing about whether a cross-company
    comparison can be answered.  ``uncertain`` says comparative language stands
    between two companies without a structure that resolves it, which fails
    closed the same way.

    Only the question text and the canonical companies are read.  Routes, task
    type, and any issuer/reporter relation are deliberately not consulted, so
    the same sentence classifies identically wherever it is routed.
    """

    if len(companies) < 2:
        return None

    compact = re.sub(r"\s+", "", query)
    has_explicit = any(term in compact for term in _EXPLICIT_COMPARISON)
    has_choice = any(term in compact for term in _CHOICE_MARKERS)
    has_enumeration = any(term in compact for term in _ENUMERATION_MARKERS)
    has_predicate = any(term in compact for term in _COMPARATIVE_PREDICATES)
    has_operator = any(term in compact for term in _COMPARISON_OPERATORS)
    comparative = (
        has_explicit or has_choice or has_enumeration or has_predicate or has_operator
    )
    if not comparative:
        return None

    # Two companies are named but at most one could be located, so no operand
    # structure can be checked.  Comparative language is present, so decline.
    if len(mentions) < 2:
        return "uncertain"

    # Structure first: an operator or a connector that actually joins the
    # company mentions outranks vocabulary that merely appears nearby.
    if _operator_binds_companies(query, mentions):
        return "cross_company"
    if _companies_are_coordinated(query, mentions) and (
        has_choice or has_enumeration or has_explicit or has_predicate
    ):
        return "cross_company"

    # No cross-company structure.  A period or prior-filing reference explains
    # the comparative wording, so the question stays a single-subject fact
    # request and remains available to later role interpretation.
    if any(anchor in compact for anchor in _TEMPORAL_ANCHORS):
        return None
    if _TEMPORAL_YEAR.search(compact):
        return None

    # Comparative language between two companies that no structure resolves.
    return "uncertain"


def _comparison_from_query(
    query: str,
    companies: tuple[str, ...],
    years: tuple[int, ...],
    mentions: tuple[tuple[int, int, str], ...] = (),
) -> Mapping[str, Any] | None:
    compact = re.sub(r"\s+", "", query)
    if any(term in compact for term in ("변동전후", "변경전후", "이전과이후")):
        return {"type": "before_after"}
    if "전년대비" in compact or "전년동기대비" in compact:
        payload: dict[str, Any] = {"type": "year_over_year"}
        if len(years) == 1:
            payload["years"] = [years[0] - 1, years[0]]
        return payload
    # "비교" states the comparison outright and keeps its existing reach. "대비"
    # does not: it is the operator that also builds 자기자본 대비, 매출액 대비 and
    # 직전 보고 대비, so it counts only when a company is its left operand.
    if len(companies) > 1 and (
        "비교" in compact
        or ("대비" in compact and _operator_binds_companies(query, mentions))
    ):
        return {"type": "company_comparison", "companies": list(companies)}
    if len(years) > 1:
        return {"type": "period_comparison", "years": list(years)}
    if any(term in compact for term in ("추이", "변화", "최근3년", "최근5년")):
        return {"type": "trend"}
    return None


def _find_reporter(query: str) -> str | None:
    match = re.search(r"(국민연금(?:공단|기금)?|보고자\s*[:：]?\s*[0-9A-Za-z가-힣()㈜·._-]+)", query)
    if not match:
        return None
    value = match.group(1)
    return re.sub(r"^보고자\s*[:：]?\s*", "", value).strip()


def _latest_report_wording(query: str) -> str:
    """The question as the frozen report-relative parser can read its selector.

    That parser recognizes "the newest report" as one contiguous string --
    ``최신보고``, ``최근보고`` -- so naming the document in between (``최근 대량보유
    보고 기준``) says exactly the same thing in wording it cannot see.  Dropping
    the document's own qualifiers restores the adjacency and leaves the head
    noun the parser is looking for.

    Only the text handed to that one parser is rewritten; ``raw_query`` is what
    every other stage still reads.  The rewrite is deliberately not a selector
    decision: which report the wording then names, and whether an explicit date
    outranks it, stay entirely that parser's answers.  An exact date still wins,
    because it is tested before any latest term.
    """

    return _LATEST_HOLDING_REPORT.sub(r"\g<latest>\g<head>", query)


def _names_a_filing(key: str) -> bool:
    """Whether a canonical surface names the filing rather than the filer.

    A holding question can put the document in the same relational slot the
    holder occupies -- ``<issuer> 대량보유보고의 보유주식수`` asks about a filing,
    not about anybody -- and reading the document as a holder invents a filer
    the question never named.

    Decided by composition, not by containment: the surface names a filing when
    *all* of it is disclosure vocabulary -- qualifiers, optionally closed by a
    head noun.  So ``대량보유상황보고서`` and ``주식등의대량보유상황보고서`` are filings,
    and so is a bare fragment such as ``주식등``, which is what a spaced document
    name leaves in the holder slot.  A holder whose name merely ends in one of
    those words is not: ``보고펀드`` keeps its head elsewhere and ``대한보고`` has a
    stem no qualifier explains, so both survive as reporters.
    """

    stem = key
    for head in _REPORT_HEAD_NOUNS:
        if stem.endswith(head):
            stem = stem[: -len(head)]
            break
    while stem:
        for qualifier in _REPORT_QUALIFIERS:
            if stem.endswith(qualifier):
                stem = stem[: -len(qualifier)]
                break
        else:
            # Something no disclosure qualifier accounts for.  That is a name.
            return False
    # Nothing was left over, so the surface was the document all the way down.
    # ``key`` is never empty here: the caller rejects a surface shorter than two
    # characters before this runs.
    return True


def _natural_holding_reporter(
    query: str,
    mentions: tuple[tuple[int, int, str], ...],
    *,
    holding_metric: str | None,
) -> str | None:
    """The holder a holding question names in relation to its issuer.

    Additive to :func:`_find_reporter`, which keeps its own two shapes.  This
    reads the one construction that names a holder the company universe cannot
    canonicalize -- an individual, a fund, a foreign entity -- and which is
    therefore invisible to company-mention extraction.

    Bounded by structure, not by vocabulary.  A surface qualifies only when the
    question names exactly one corpus company, asks a holding-metric question
    about it, and attaches the surface to that company through a possessive or
    subject relation.  Two named companies are never read this way: an
    issuer/reporter role pair is a corpus question, and
    ``HoldingCompanyRoleResolver`` is what answers it.

    The returned value is the surface as written.  Whether it names a holder
    this corpus knows is decided downstream by ``canonical_reporter_key`` and
    the report index, and this function must not anticipate that answer.
    """

    if holding_metric is None:
        # Not a holding-metric question.  A possessive phrase in a financial or
        # event question names something else entirely.
        return None
    if len(mentions) != 1:
        # No issuer to be relative to, or two companies -- which is a role pair
        # or a comparison, and neither is this parser's to interpret.
        return None
    if _has_comparison_vocabulary(query):
        # Comparative or per-company wording puts more than one subject in play.
        # ``_comparison_frame`` cannot bind an operator with a single canonical
        # mention, so declining outright is the only fail-closed reading.
        return None

    _start, end, _canonical = mentions[0]
    match = _NATURAL_REPORTER.match(query, end)
    if match is None:
        return None
    surface = match.group("holder").strip()
    key = canonical_reporter_key(surface)
    if (
        len(key) < 2
        or key in _NON_REPORTER_SURFACES
        or _names_a_filing(key)
        or _BARE_PERIOD_SUBJECT.fullmatch(key)
    ):
        return None
    return surface


def _reporter_first_holding_reporter(
    query: str,
    mentions: tuple[tuple[int, int, str], ...],
    *,
    holding_metric: str | None,
) -> str | None:
    """The filer in ``<filer>가 ... 보고한 <issuer> <holding field>``.

    This is the subject-first counterpart of ``_natural_holding_reporter``.
    Exactly one corpus company must already identify the issuer.  If the
    leading surface is itself another corpus company, ``mentions`` contains two
    entries and the existing corpus-backed T8 role resolver keeps authority.

    The reporting relation, not "whatever was not the issuer", proves the
    role.  A second actor or coordinated filer in the bridge declines, and a
    non-holding question never enters this rule.
    """

    if holding_metric is None or len(mentions) != 1:
        return None
    if _has_comparison_vocabulary(query):
        return None

    issuer_start, issuer_end, _canonical = mentions[0]
    subject = _REPORTER_FIRST_SUBJECT.match(query)
    if subject is None or subject.end() > issuer_start:
        return None

    bridge = query[subject.end():issuer_start]
    relation = _REPORTER_FIRST_RELATION.search(bridge)
    if relation is None:
        return None
    modifier = bridge[:relation.start()]
    if _REPORTER_FIRST_NESTED_ACTOR.search(modifier):
        return None
    if not _REPORTER_FIRST_HOLDING_OBJECT.match(query[issuer_end:]):
        return None

    surface = subject.group("holder").strip()
    key = canonical_reporter_key(surface)
    if (
        len(key) < 2
        or key in _NON_REPORTER_SURFACES
        or _names_a_filing(key)
        or _BARE_PERIOD_SUBJECT.fullmatch(key)
    ):
        return None
    return surface


def _operation_from_query(
    query: str,
    *,
    task_type: str,
    event_type: str | None,
    correction_intent: str | None,
) -> str:
    """Return the deterministic operation without interpreting unknown prose.

    The value is additive plan evidence. Retrieval does not read it; P0-D uses
    it to distinguish a fully specified query from one that needs semantic
    clarification.
    """

    compact = re.sub(r"\s+", "", query)
    if correction_intent:
        return "correction_lookup"
    if task_type == "financial_metric":
        return "lookup_metric"
    if task_type == "holding_change":
        return "lookup_holding"
    if task_type == "corporate_event":
        if event_type in {
            "contract_termination",
            "treasury_share_trust_termination",
        } or any(term in compact for term in ("해지된", "해지한", "종료된")):
            return "find_terminated"
        if any(
            term in compact
            for term in ("몇건", "몇개", "모두", "전체", "목록", "나열")
        ):
            return "enumerate"
        return "inspect_event"
    return "lookup_disclosure"


def _infer_company_prefix(query: str) -> str | None:
    marker = re.search(
        r"(?<!\d)20\d{2}(?:\s*년|\s*회계연도)?|[1-4]\s*분기|"
        r"매출액?|영업이익|당기순이익|자산총계|부채총계|자본총계|"
        r"유상증자|전환사채|합병|공급계약|시설투자|보유주식|보유비율|지분율",
        query,
    )
    if not marker:
        return None
    prefix = query[: marker.start()].strip(" ,:：의은는이가을를")
    if not prefix or len(prefix) > 30 or re.search(r"[?？]", prefix):
        return None
    if any(term in prefix for term in ("최근", "올해", "작년", "어느", "무슨")):
        return None
    return prefix


def _company_spans(
    query: str,
    companies: tuple[str, ...],
    aliases: Mapping[str, set[str]],
) -> list[tuple[int, int]]:
    terms = set(companies)
    for alias, canonical_names in aliases.items():
        if set(companies).intersection(canonical_names):
            terms.add(alias)
    spans: list[tuple[int, int]] = []
    for term in sorted(terms, key=len, reverse=True):
        for match in re.finditer(re.escape(term), query, re.IGNORECASE):
            start, end = match.span()
            if end < len(query) and query[end] in "은는이가의을를와과":
                end += 1
            spans.append((start, end))
    return spans


def _metadata_term_spans(
    query: str,
    extracted: Mapping[str, Any],
    period_type: str | None,
) -> list[tuple[int, int]]:
    terms = ["사업보고서", "분기보고서", "반기보고서"]
    if period_type == "receipt_date":
        terms.extend(["공시한", "공시된", "접수된", "공시일"])
    spans: list[tuple[int, int]] = []
    for term in terms:
        spans.extend(match.span() for match in re.finditer(re.escape(term), query))
    return spans


def _period_column_spans(query: str) -> list[tuple[int, int]]:
    return [
        match.span()
        for match in re.finditer(r"누적|누계|3\s*개월", query)
    ]


def _correction_spans(query: str, evidence: str | None) -> list[tuple[int, int]]:
    if not evidence:
        return []
    return [match.span() for match in re.finditer(r"정정\s*(?:공시|본)?(?:\s*만|\s*제외)?", query)]


def _normalize_lexical_query(
    raw_query: str,
    spans: list[tuple[int, int]],
    *,
    metric: str | None,
    metric_evidence: str | None,
    event_type: str | None,
    event_evidence: str | None,
) -> str:
    characters = list(raw_query)
    for start, end in _merge_spans(spans):
        for index in range(start, end):
            characters[index] = " "
    value = "".join(characters)
    value = re.sub(r"(?:알려\s*줘|알려주세요|무엇인가요|얼마인가요|얼마야|조회해줘)", " ", value)
    value = re.sub(r"[?？,，:：/|]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .-_~")
    if metric and metric_evidence:
        value = re.sub(re.escape(metric_evidence), metric, value, flags=re.IGNORECASE)
    if metric:
        value = _strip_metric_particles(value, metric)
    value = re.sub(r"\s+", " ", value).strip()
    if value:
        return _expand_metric_lexical_query(value, metric)
    if metric:
        return _expand_metric_lexical_query(metric, metric)
    if event_evidence:
        return event_evidence
    if event_type:
        return event_type
    return raw_query


def _expand_metric_lexical_query(value: str, metric: str | None) -> str:
    terms = [value]
    normalized = re.sub(r"\s+", "", value).casefold()
    for term in _FINANCIAL_METRIC_SEARCH_TERMS.get(metric or "", ()):
        if term.casefold() not in normalized:
            terms.append(term)
    return " ".join(terms)


def _strip_metric_particles(value: str, metric: str) -> str:
    escaped = re.escape(metric)
    return re.sub(
        rf"(?<![0-9A-Za-z가-힣]){escaped}(?:은|는|이|가|을|를|의)(?![0-9A-Za-z가-힣])",
        metric,
        value,
    )


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(set(spans)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _canonical_names(names: Iterable[str]) -> set[str]:
    values = (names,) if isinstance(names, str) else names
    return {str(name) for name in values if str(name).strip()}
