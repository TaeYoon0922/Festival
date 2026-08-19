"""Deterministic query understanding with safe lexical-query normalization."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from app.parsing.metadata_filtered_retrieval import extract_metadata_filters
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

_EVENTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("capital_increase", ("유상증자",), "major"),
    ("convertible_bond", ("전환사채",), "major"),
    ("treasury_share_disposal", ("자기주식처분", "자사주처분"), "major"),
    ("spin_off", ("회사분할", "분할신설", "분할비율"), "major"),
    ("merger", ("합병", "흡수합병"), "major"),
    ("supply_contract", ("단일판매", "공급계약", "수주계약"), "exchange"),
    ("contract_termination", ("계약해지", "공급계약해지"), "exchange"),
    ("facility_investment", ("시설투자", "신규시설투자"), "exchange"),
)

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
    "자산총계": {"재무상태표": 1.0, "자산총계": 0.95},
    "부채총계": {"재무상태표": 1.0, "부채총계": 0.95},
    "자본총계": {"재무상태표": 1.0, "자본총계": 0.95},
    "주당순이익": {
        "주당이익": 1.0,
        "포괄손익계산서": 0.85,
        "손익계산서": 0.82,
    },
    "holding_ratio": {"보유 주식": 1.0, "대량보유": 0.95, "보유비율": 0.90},
    "holding_shares": {"보유 주식": 1.0, "대량보유": 0.95, "주식수": 0.90},
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

        financial_metric, metric_evidence = _find_financial_metric(raw_query)
        event_type, event_evidence, event_route = _find_event(raw_query)
        holding_metric, holding_evidence = _find_holding_metric(raw_query)
        reporter = _find_reporter(raw_query)

        if holding_metric:
            task_type = "holding_change"
            metric = holding_metric
        elif financial_metric:
            task_type = "financial_metric"
            metric = financial_metric
        elif event_type:
            task_type = "corporate_event"
            metric = None
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
        period, period_spans, mentioned_years = _period_from_query(
            raw_query,
            task_type=task_type,
            routes=routes,
        )
        basis, basis_evidence, basis_span = _basis_from_query(raw_query)
        parsed_correction, correction_confidence, correction_evidence = (
            _correction_from_query(raw_query)
        )
        if correction_policy is not None:
            parsed_correction = correction_policy
            correction_confidence = 1.0
            correction_evidence = "caller_override"
        comparison = _comparison_from_query(raw_query, companies, mentioned_years)

        subtype = extracted["doc_subtype"]
        if subtype:
            explicit_report = any(
                term in re.sub(r"\s+", "", raw_query)
                for term in ("분기보고서", "반기보고서", "사업보고서")
            )
            route_confidence["doc_subtype"] = 0.99 if explicit_report else 0.90
            route_evidence["doc_subtype"] = extracted["doc_subtype_evidence"]
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
        structured_spans.extend(_correction_spans(raw_query, correction_evidence))
        lexical_query = _normalize_lexical_query(
            raw_query,
            structured_spans,
            metric=metric,
            metric_evidence=metric_evidence or holding_evidence,
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
            section_boosts=_SECTION_BOOSTS.get(metric or "", {}),
            route_confidence=route_confidence,
            route_evidence=route_evidence,
            top_k=top_k,
            evidence={
                "mentioned_years": list(mentioned_years),
                "period_type": period.period_type,
                "report_preference": "annual" if annual_preferred else None,
                "company_resolved": bool(resolved),
                "metric": metric_evidence or holding_evidence,
                "event_type": event_evidence,
                "structured_spans": [list(span) for span in sorted(set(structured_spans))],
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


def _find_event(query: str) -> tuple[str | None, str | None, str | None]:
    compact = re.sub(r"\s+", "", query).casefold()
    for event_type, aliases, route in _EVENTS:
        match = next((alias for alias in aliases if alias.casefold() in compact), None)
        if match:
            return event_type, match, route
    return None, None, None


def _find_holding_metric(query: str) -> tuple[str | None, str | None]:
    compact = re.sub(r"\s+", "", query)
    if any(term in compact for term in ("보유비율", "지분율", "보유주식비율")):
        return "holding_ratio", "보유비율/지분율"
    if any(term in compact for term in ("보유주식", "주식수", "보유수량")):
        return "holding_shares", "보유주식/주식수/보유수량"
    if "지분" in compact and any(term in compact for term in ("변동", "증감", "보유")):
        return "holding_ratio", "지분+변동/증감/보유"
    return None, None


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


def _period_from_query(
    query: str,
    *,
    task_type: str,
    routes: tuple[str, ...],
) -> tuple[QueryPeriod, list[tuple[int, int]], tuple[int, ...]]:
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
    if len(dates) >= 2:
        spans.extend(span for _, span in dates[:2])
        return (
            QueryPeriod(
                from_date=dates[0][0],
                to_date=dates[1][0],
                period_type="date_range",
            ),
            spans,
            years,
        )

    year = years[0] if len(years) == 1 else None
    compact = compact_query
    if year is not None and any(
        marker in compact for marker in ("공시한", "공시된", "접수된", "공시일")
    ):
        return _whole_year_period(year, "receipt_date"), spans, years
    if year is not None and task_type == "holding_change":
        return _whole_year_period(year, "date_range"), spans, years
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
        )
    if year is not None:
        return QueryPeriod(year=year, period_type="reference_year"), spans, years
    if quarter is not None:
        return QueryPeriod(quarter=quarter, period_type="fiscal_quarter"), spans, years
    if task_type == "holding_change":
        return QueryPeriod(period_type="latest_holding"), spans, years
    if task_type == "corporate_event":
        return QueryPeriod(period_type="latest_event"), spans, years
    return QueryPeriod(), spans, years


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


def _correction_from_query(query: str) -> tuple[str, float, str | None]:
    compact = re.sub(r"\s+", "", query)
    if any(term in compact for term in ("정정제외", "원공시만", "최초공시만")):
        return "original_only", 0.99, "정정 제외/원공시만"
    if any(term in compact for term in ("정정공시만", "정정본만", "정정만")):
        return "corrected_only", 0.99, "정정 공시만"
    if "정정" in compact:
        return "latest_preferred", 0.85, "정정"
    return "latest_preferred", 0.0, None


def _comparison_from_query(
    query: str,
    companies: tuple[str, ...],
    years: tuple[int, ...],
) -> Mapping[str, Any] | None:
    compact = re.sub(r"\s+", "", query)
    if any(term in compact for term in ("변동전후", "변경전후", "이전과이후")):
        return {"type": "before_after"}
    if "전년대비" in compact or "전년동기대비" in compact:
        payload: dict[str, Any] = {"type": "year_over_year"}
        if len(years) == 1:
            payload["years"] = [years[0] - 1, years[0]]
        return payload
    if len(companies) > 1 and any(term in compact for term in ("비교", "대비")):
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
    value = re.sub(r"\s+", " ", value).strip()
    if value:
        return value
    if metric:
        return metric
    if event_evidence:
        return event_evidence
    if event_type:
        return event_type
    return raw_query


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
