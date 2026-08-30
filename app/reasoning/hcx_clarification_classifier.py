"""Optional HCX classifier over a deterministic clarification candidate set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.reasoning.clarification_request import ClarificationCandidate
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


HCX_CLARIFICATION_SYSTEM_PROMPT = """You classify a Korean disclosure question.
Return exactly one RFC 8259 JSON object with exactly these keys:
decision, candidate_ids, reason.
decision is one of resolved, clarify. candidate_ids contains only IDs supplied
in candidate_interpretations. Use exactly one ID for resolved and at least two
supplied IDs for clarify. reason is a short string. Do not add facts,
candidates, companies, dates, metrics,
evidence, an answer, prose outside JSON, or a Markdown fence. If the question
does not uniquely select one supplied candidate, return clarify.
"""

_FIELDS = frozenset({"decision", "candidate_ids", "reason"})
_DECISIONS = frozenset({"resolved", "clarify"})


class ClarificationClassifierSchemaError(ValueError):
    def __init__(self, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(code)
        self.code = str(code)
        self.fields = tuple(sorted(dict.fromkeys(str(field) for field in fields)))


@dataclass(frozen=True)
class HcxClarificationResult:
    decision: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class HcxClarificationOutcome:
    result: HcxClarificationResult | None
    status: str
    elapsed_ms: float
    transport_status: str = "not_called"
    parse_status: str = "not_attempted"
    schema_error_code: str | None = None
    schema_error_fields: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status == "success" and self.result is not None

    def diagnostic(self) -> dict[str, Any]:
        return {
            "transport_status": self.transport_status,
            "parse_status": self.parse_status,
            "schema_error_code": self.schema_error_code,
            "schema_error_fields": list(self.schema_error_fields),
        }


class HcxClarificationClassifier:
    """Choose within supplied IDs, or return a failure the resolver can ignore."""

    def __init__(
        self,
        settings: HcxSettings | None = None,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.settings = settings or HcxSettings.from_env()
        self.transport = transport or UrllibJsonTransport()
        self.call_count = 0

    def classify(
        self,
        question: str,
        candidates: Sequence[ClarificationCandidate],
    ) -> HcxClarificationOutcome:
        start = perf_counter()
        supplied = tuple(candidates)
        if not self.settings.enabled:
            return HcxClarificationOutcome(None, "disabled", _elapsed_ms(start))
        if not self.settings.configured:
            return HcxClarificationOutcome(None, "not_configured", _elapsed_ms(start))
        if len(supplied) < 2:
            return HcxClarificationOutcome(None, "not_needed", _elapsed_ms(start))

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": HCX_CLARIFICATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": str(question),
                            "candidate_interpretations": [
                                candidate.to_classifier_dict() for candidate in supplied
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(self.settings.max_tokens, 256),
        }
        self.call_count += 1
        try:
            response = self.transport.post_json(
                self.settings.endpoint,
                headers=self.settings.request_headers(),
                payload=payload,
                timeout_seconds=self.settings.timeout_seconds,
            )
        except (EmbeddingHttpError, TimeoutError):
            return HcxClarificationOutcome(
                None,
                "transport_failure",
                _elapsed_ms(start),
                transport_status="failure",
            )
        except Exception:  # noqa: BLE001 - deterministic clarification remains safe
            return HcxClarificationOutcome(
                None,
                "classifier_error",
                _elapsed_ms(start),
                transport_status="failure",
            )

        content = _response_content(response) if isinstance(response, Mapping) else None
        if content is None:
            return HcxClarificationOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                parse_status="content_missing",
            )
        try:
            result = parse_hcx_clarification_result(
                content,
                allowed_candidate_ids=tuple(candidate.id for candidate in supplied),
            )
        except json.JSONDecodeError:
            return HcxClarificationOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                parse_status="invalid_json",
            )
        except ClarificationClassifierSchemaError as error:
            return HcxClarificationOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                parse_status="schema_invalid",
                schema_error_code=error.code,
                schema_error_fields=error.fields,
            )
        return HcxClarificationOutcome(
            result,
            "success",
            _elapsed_ms(start),
            transport_status="success",
            parse_status="success",
        )


def parse_hcx_clarification_result(
    content: str | Mapping[str, Any],
    *,
    allowed_candidate_ids: Sequence[str],
) -> HcxClarificationResult:
    if isinstance(content, Mapping):
        payload = dict(content)
    else:
        payload = json.loads(str(content).strip())
    if not isinstance(payload, dict):
        raise ClarificationClassifierSchemaError("invalid_type")
    missing = _FIELDS - set(payload)
    unexpected = set(payload) - _FIELDS
    if missing:
        raise ClarificationClassifierSchemaError("missing_required", missing)
    if unexpected:
        raise ClarificationClassifierSchemaError("unexpected_field", unexpected)

    decision = payload.get("decision")
    if not isinstance(decision, str) or decision not in _DECISIONS:
        raise ClarificationClassifierSchemaError("invalid_enum", ("decision",))
    candidate_ids = payload.get("candidate_ids")
    if not isinstance(candidate_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in candidate_ids
    ):
        raise ClarificationClassifierSchemaError("invalid_type", ("candidate_ids",))
    identifiers = tuple(dict.fromkeys(value.strip() for value in candidate_ids))
    if len(identifiers) != len(candidate_ids):
        raise ClarificationClassifierSchemaError("duplicate_candidate", ("candidate_ids",))
    allowed = set(str(value) for value in allowed_candidate_ids)
    if any(identifier not in allowed for identifier in identifiers):
        raise ClarificationClassifierSchemaError("unknown_candidate", ("candidate_ids",))

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
        raise ClarificationClassifierSchemaError("invalid_type", ("reason",))
    if decision == "resolved" and len(identifiers) != 1:
        raise ClarificationClassifierSchemaError("invalid_cardinality", ("candidate_ids",))
    if decision == "clarify" and len(identifiers) < 2:
        raise ClarificationClassifierSchemaError("invalid_cardinality", ("candidate_ids",))
    return HcxClarificationResult(decision=decision, candidate_ids=identifiers)


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


__all__ = [
    "HCX_CLARIFICATION_SYSTEM_PROMPT",
    "ClarificationClassifierSchemaError",
    "HcxClarificationClassifier",
    "HcxClarificationOutcome",
    "HcxClarificationResult",
    "parse_hcx_clarification_result",
]
