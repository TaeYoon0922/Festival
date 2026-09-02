"""Corpus-backed issuer/reporter direction for two-company holding queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.reasoning.holding_report_index import HoldingReportIndex
from app.reasoning.holding_reporter import (
    canonical_reporter_key,
    reporter_matches,
    reporter_surface_spans,
)


#: ``plan.evidence`` key recording that this plan's reporter was produced by
#: corpus-backed two-company role resolution.  Only the validator writes it and
#: only evidence scoping reads it: it is provenance and a gate, never a value
#: store.  The typed ``QueryPlan.company``/``corp_code``/``reporter`` fields
#: remain the single source of truth for who the issuer and the holder are, so
#: nothing here is duplicated and nothing downstream can drift from them.
ROLE_PROVENANCE_KEY = "holding_company_role"
ROLE_PROVENANCE_SOURCE = "corpus_relation"

RESOLVED = "resolved"
NO_INDEX = "no_index"
INCOMPLETE_INDEX = "incomplete_index"
STALE_INDEX = "stale_index"
INVALID_PAIR = "invalid_pair"
NO_DIRECTION = "no_direction"
BIDIRECTIONAL = "bidirectional"
INDEX_ERROR = "index_error"
#: No holder of this issuer answers to the name the question used.
UNKNOWN_FILER = "unknown_filer"
#: More than one holder of this issuer answers to it, and nothing separates
#: them.  Two holders are not one holder written twice.
AMBIGUOUS_FILER = "ambiguous_filer"

#: Which corpus fact proved a role.  Both paths read the same index and both
#: earn the same provenance source; the path says which question the index was
#: asked.  ``two_company_relation`` compared both directed relationships;
#: ``filer_identity`` matched a named holder against this issuer's own filers.
ROLE_PATH_RELATION = "two_company_relation"
ROLE_PATH_FILER = "filer_identity"
#: ``query_grounded`` searched this issuer's own holders for one the question
#: writes out.  The corpus still supplies the identity; the question only
#: says which of the holders it already has is the one being asked about.
ROLE_PATH_QUERY_GROUNDED = "query_grounded"


@dataclass(frozen=True)
class HoldingCompanyRoleResolution:
    """The result of evaluating both directed corpus relationships."""

    status: str
    issuer: str | None = None
    issuer_corp_code: str | None = None
    reporter: str | None = None
    reporter_key: str = ""
    direction_1_report_count: int = 0
    direction_2_report_count: int = 0
    reason: str | None = None

    @property
    def resolved(self) -> bool:
        return (
            self.status == RESOLVED
            and bool(self.issuer)
            and bool(self.issuer_corp_code)
            and bool(self.reporter)
            and bool(self.reporter_key)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "resolved": self.resolved,
            "issuer": self.issuer,
            "issuer_corp_code": self.issuer_corp_code,
            "reporter": self.reporter,
            "reporter_key": self.reporter_key,
            "direction_1_report_count": self.direction_1_report_count,
            "direction_2_report_count": self.direction_2_report_count,
            "reason": self.reason,
        }


class HoldingCompanyRoleResolver:
    """Resolve roles only when exactly one indexed direction exists."""

    def __init__(
        self,
        index: HoldingReportIndex | None,
        *,
        active_corpus_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.active_corpus_identity = (
            dict(active_corpus_identity)
            if active_corpus_identity is not None
            else None
        )

    def resolve(
        self,
        company_a: str,
        corp_code_a: str,
        company_b: str,
        corp_code_b: str,
    ) -> HoldingCompanyRoleResolution:
        """Evaluate ``A issuer/B reporter`` and its reverse independently."""

        name_a = str(company_a or "").strip()
        name_b = str(company_b or "").strip()
        code_a = str(corp_code_a or "").strip()
        code_b = str(corp_code_b or "").strip()
        key_a = canonical_reporter_key(name_a)
        key_b = canonical_reporter_key(name_b)
        if (
            not name_a
            or not name_b
            or not code_a
            or not code_b
            or not key_a
            or not key_b
            or name_a == name_b
            or code_a == code_b
        ):
            return HoldingCompanyRoleResolution(
                INVALID_PAIR, reason="two distinct canonical companies are required"
            )
        if self.index is None:
            return HoldingCompanyRoleResolution(
                NO_INDEX, reason="holding report index is unavailable"
            )
        if not self.index.complete:
            return HoldingCompanyRoleResolution(
                INCOMPLETE_INDEX, reason="holding report index is incomplete"
            )
        if (
            self.active_corpus_identity is not None
            and not self.index.matches_corpus(self.active_corpus_identity)
        ):
            return HoldingCompanyRoleResolution(
                STALE_INDEX, reason="holding report index does not match the active corpus"
            )

        try:
            direction_1 = self.index.enumerate_reports(code_a, name_b)
            direction_2 = self.index.enumerate_reports(code_b, name_a)
        except Exception:  # noqa: BLE001 - corpus lookup must fail closed
            return HoldingCompanyRoleResolution(
                INDEX_ERROR, reason="holding relationship lookup failed"
            )

        count_1 = len(direction_1)
        count_2 = len(direction_2)
        if bool(count_1) == bool(count_2):
            status = BIDIRECTIONAL if count_1 else NO_DIRECTION
            reason = (
                "both directed relationships are present"
                if count_1
                else "neither directed relationship is present"
            )
            return HoldingCompanyRoleResolution(
                status,
                direction_1_report_count=count_1,
                direction_2_report_count=count_2,
                reason=reason,
            )
        if count_1:
            return HoldingCompanyRoleResolution(
                RESOLVED,
                issuer=name_a,
                issuer_corp_code=code_a,
                reporter=name_b,
                reporter_key=key_b,
                direction_1_report_count=count_1,
                direction_2_report_count=count_2,
            )
        return HoldingCompanyRoleResolution(
            RESOLVED,
            issuer=name_b,
            issuer_corp_code=code_b,
            reporter=name_a,
            reporter_key=key_a,
            direction_1_report_count=count_1,
            direction_2_report_count=count_2,
        )

    def resolve_filer(
        self, issuer: str, issuer_corp_code: str, surface: str
    ) -> HoldingCompanyRoleResolution:
        """Prove which of an issuer's own filers a question named.

        The other ``resolve`` compares two canonical companies because both are
        in the company universe.  A holding report's filer usually is not: the
        universe is issuer-scoped, so a question naming a filer hands this
        stage a *surface* and nothing else.  What makes it resolvable anyway is
        that the corpus records every filer of every issuer, so the surface is
        checked against that list instead of against an alias table.

        Matching is the frozen ``reporter_matches`` and nothing wider, and an
        exact canonical key wins outright: a corpus that records both 국민연금
        and 국민연금공단 for one issuer keeps them as the two holders they are.
        Anything else that matches more than one holder is refused, because
        picking either would state one holder's position as the other's.
        """

        name = str(issuer or "").strip()
        code = str(issuer_corp_code or "").strip()
        named = str(surface or "").strip()
        key = canonical_reporter_key(named)
        if not name or not code or not key:
            return HoldingCompanyRoleResolution(
                INVALID_PAIR, reason="an issuer and a named filer are both required"
            )
        if self.index is None:
            return HoldingCompanyRoleResolution(
                NO_INDEX, reason="holding report index is unavailable"
            )
        if not self.index.complete:
            return HoldingCompanyRoleResolution(
                INCOMPLETE_INDEX, reason="holding report index is incomplete"
            )
        if (
            self.active_corpus_identity is not None
            and not self.index.matches_corpus(self.active_corpus_identity)
        ):
            return HoldingCompanyRoleResolution(
                STALE_INDEX,
                reason="holding report index does not match the active corpus",
            )

        try:
            filers = self.index.enumerate_reporters(code)
            matches = [filer for filer in filers if reporter_matches(filer, named)]
            exact = [
                filer for filer in matches if canonical_reporter_key(filer) == key
            ]
            chosen = exact or matches
            reports = {
                canonical_reporter_key(filer): len(
                    self.index.enumerate_reports(code, filer)
                )
                for filer in chosen
            }
        except Exception:  # noqa: BLE001 - corpus lookup must fail closed
            return HoldingCompanyRoleResolution(
                INDEX_ERROR, reason="holding filer lookup failed"
            )

        if not chosen:
            return HoldingCompanyRoleResolution(
                UNKNOWN_FILER,
                reason="no holder of this issuer answers to the name given",
            )
        if len(reports) > 1:
            return HoldingCompanyRoleResolution(
                AMBIGUOUS_FILER,
                direction_1_report_count=sum(reports.values()),
                reason="more than one holder of this issuer answers to the name given",
            )
        filer = chosen[0]
        return HoldingCompanyRoleResolution(
            RESOLVED,
            issuer=name,
            issuer_corp_code=code,
            reporter=filer,
            reporter_key=canonical_reporter_key(filer),
            direction_1_report_count=next(iter(reports.values())),
        )

    def resolve_query_grounded(
        self, issuer: str, issuer_corp_code: str, question: str
    ) -> HoldingCompanyRoleResolution:
        """The one holder of this issuer that this question writes out.

        The question names somebody, and the deterministic parsers could not
        tell which of its words that was -- the holder is two words long, or
        carries no particle, or sits behind wording those parsers decline.  So
        the question is turned around: instead of searching the sentence for a
        name, this asks which of the holders the corpus *already records for
        this issuer* is written in it.  Nothing outside that set can ever be
        produced, which is what makes the search safe without a grammar for it.

        The issuer is excluded by identity before counting, because a company
        is frequently both, and its own name appearing in its own question
        proves nothing.  Exactly one surviving holder binds; none and more than
        one both leave the reporter alone, and the ambiguous case is not
        resolved by preferring longer, earlier or more frequent holders.
        """

        name = str(issuer or "").strip()
        code = str(issuer_corp_code or "").strip()
        text = str(question or "")
        if not name or not code or not text:
            return HoldingCompanyRoleResolution(
                INVALID_PAIR, reason="an issuer and a question are both required"
            )
        if self.index is None:
            return HoldingCompanyRoleResolution(
                NO_INDEX, reason="holding report index is unavailable"
            )
        if not self.index.complete:
            return HoldingCompanyRoleResolution(
                INCOMPLETE_INDEX, reason="holding report index is incomplete"
            )
        if (
            self.active_corpus_identity is not None
            and not self.index.matches_corpus(self.active_corpus_identity)
        ):
            return HoldingCompanyRoleResolution(
                STALE_INDEX,
                reason="holding report index does not match the active corpus",
            )

        issuer_key = canonical_reporter_key(name)
        try:
            written: dict[str, str] = {}
            for filer in self.index.enumerate_reporters(code):
                key = canonical_reporter_key(filer)
                if not key or key == issuer_key:
                    continue
                spans = reporter_surface_spans(text, filer)
                if len(spans) != 1:
                    # Absent, or written more than once with nothing to say
                    # which mention is the holder being asked about.
                    continue
                start, end = spans[0]
                written.setdefault(key, text[start:end])
        except Exception:  # noqa: BLE001 - corpus lookup must fail closed
            return HoldingCompanyRoleResolution(
                INDEX_ERROR, reason="holding filer lookup failed"
            )

        if not written:
            return HoldingCompanyRoleResolution(
                UNKNOWN_FILER,
                reason="the question names no holder this issuer is known to have",
            )
        if len(written) > 1:
            return HoldingCompanyRoleResolution(
                AMBIGUOUS_FILER,
                reason="the question names more than one holder of this issuer",
            )
        # The surface came out of the question; the identity behind it is still
        # decided by the same filer lookup every other holding path uses.
        return self.resolve_filer(name, code, next(iter(written.values())))

    def document_ids(
        self, issuer_corp_code: str, reporter: str
    ) -> frozenset[str] | None:
        """Indexed documents for an exact pair, or ``None`` if not provable.

        This is a post-retrieval scope, not a search.  An unavailable, stale,
        incomplete, or unmatched index leaves older one-company reporter flows
        untouched; a B.2-resolved pair always reaches the non-empty branch.
        """

        code = str(issuer_corp_code or "").strip()
        key = canonical_reporter_key(reporter)
        if not code or not key or self.index is None or not self.index.complete:
            return None
        if (
            self.active_corpus_identity is not None
            and not self.index.matches_corpus(self.active_corpus_identity)
        ):
            return None
        try:
            records = self.index.enumerate_reports(code, reporter)
        except Exception:  # noqa: BLE001 - evidence scoping must fail closed
            return None
        if not records:
            return None
        return frozenset(record.doc_id for record in records)


def role_provenance(
    resolution: HoldingCompanyRoleResolution,
    *,
    path: str = ROLE_PATH_RELATION,
) -> dict[str, Any]:
    """Bounded provenance for one resolved direction.

    Deliberately carries no issuer or reporter value.  Exactly one direction is
    non-empty for a resolved result, so its report count is the informative
    half and the other is zero.  ``path`` records which corpus question proved
    it, so a reader can tell a two-company relation from a named filer without
    either one having to restate the parties.
    """

    return {
        "source": ROLE_PROVENANCE_SOURCE,
        "resolved": True,
        "path": path,
        "direction_report_count": max(
            resolution.direction_1_report_count,
            resolution.direction_2_report_count,
        ),
    }


def has_role_provenance(plan: Any) -> bool:
    """Whether this plan's reporter came from corpus-backed role resolution."""

    evidence = getattr(plan, "evidence", None)
    if not isinstance(evidence, Mapping):
        return False
    marker = evidence.get(ROLE_PROVENANCE_KEY)
    return (
        isinstance(marker, Mapping)
        and marker.get("source") == ROLE_PROVENANCE_SOURCE
        and bool(marker.get("resolved"))
    )


__all__ = [
    "AMBIGUOUS_FILER",
    "BIDIRECTIONAL",
    "HoldingCompanyRoleResolution",
    "HoldingCompanyRoleResolver",
    "ROLE_PATH_FILER",
    "ROLE_PATH_RELATION",
    "ROLE_PROVENANCE_KEY",
    "ROLE_PROVENANCE_SOURCE",
    "has_role_provenance",
    "role_provenance",
    "INCOMPLETE_INDEX",
    "INDEX_ERROR",
    "INVALID_PAIR",
    "NO_DIRECTION",
    "NO_INDEX",
    "RESOLVED",
    "STALE_INDEX",
    "UNKNOWN_FILER",
]
