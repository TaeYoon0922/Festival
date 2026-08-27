"""One-shot HyperCLOVA X semantic fallback for unresolved query slots.

The model is a constrained classifier, never a retriever or fact judge.  Its
JSON output is accepted only after schema checks here and deterministic slot,
enum, company, time-range, and conflict checks in ``query_validation``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.reasoning.query_validation import (
    ALLOWED_EVENT_FAMILIES,
    ALLOWED_OPERATIONS,
    ALLOWED_TASK_TYPES,
    QueryValidationResult,
)
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


SEMANTIC_QUERY_SYSTEM_PROMPT = """You classify Korean DART disclosure questions.
Return exactly one valid RFC 8259 JSON object and no prose or Markdown fence.
Always include these keys exactly once:
task_type, event_family, operation, set_intent, requested_state,
ambiguity, possible_interpretations.

Use double quotes for every key and string. Use only lowercase JSON literals
true, false, and null. Never use Python literals True, False, or None, single
quotes, comments, or trailing commas. Follow this JSON syntax template:
{"task_type":null,"event_family":null,"operation":null,"set_intent":null,
"requested_state":null,"ambiguity":false,"possible_interpretations":[]}.
Do not add any other top-level key. task_type, event_family, and operation are
either null or one supplied enum value. set_intent is true, false, or null.
requested_state is null or a non-empty string of at most 80 characters.
ambiguity is a boolean. possible_interpretations is an array of at most five
non-empty strings, or objects containing only non-empty string fields named id,
label, task_type, event_family, and operation; each string is at most 80
characters and enum-named fields use only supplied enum values. Use null when
the text does not determine a value.
Do not search for documents. Do not decide whether a company, correction,
event relation, document truth, completeness, or fact exists. Do not answer the
question. Preserve deterministic locked slots exactly.
"""

_SEMANTIC_FIELDS = frozenset(
    {
        "task_type",
        "event_family",
        "operation",
        "set_intent",
        "requested_state",
        "ambiguity",
        "possible_interpretations",
    }
)
_INTERPRETATION_FIELDS = frozenset(
    {"id", "label", "task_type", "event_family", "operation"}
)


class SemanticSchemaError(ValueError):
    """Schema rejection carrying only safe field names and a stable code."""

    def __init__(self, code: str, fields: tuple[str, ...] | list[str]) -> None:
        super().__init__(code)
        self.code = code
        self.fields = tuple(sorted(dict.fromkeys(str(field) for field in fields)))


@dataclass(frozen=True)
class SemanticQueryResult:
    task_type: str | None = None
    event_family: str | None = None
    operation: str | None = None
    set_intent: bool | None = None
    requested_state: str | None = None
    ambiguity: bool = False
    possible_interpretations: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "event_family": self.event_family,
            "operation": self.operation,
            "set_intent": self.set_intent,
            "requested_state": self.requested_state,
            "ambiguity": self.ambiguity,
            "possible_interpretations": list(self.possible_interpretations),
        }


@dataclass(frozen=True)
class SemanticFallbackOutcome:
    result: SemanticQueryResult | None
    status: str
    elapsed_ms: float
    transport_status: str = "not_called"
    response_shape: str | None = None
    content_present: bool = False
    parse_status: str = "not_attempted"
    content_format: str | None = None
    prefix_text_present: bool = False
    suffix_text_present: bool = False
    schema_error_code: str | None = None
    schema_error_fields: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.status == "success" and self.result is not None

    def diagnostic(self) -> dict[str, Any]:
        """Safe transport/parse metadata; never includes model content."""

        return {
            "transport_status": self.transport_status,
            "response_shape": self.response_shape,
            "content_present": self.content_present,
            "parse_status": self.parse_status,
            "content_format": self.content_format,
            "prefix_text_present": self.prefix_text_present,
            "suffix_text_present": self.suffix_text_present,
            "schema_error_code": self.schema_error_code,
            "schema_error_fields": list(self.schema_error_fields),
        }


class HcxSemanticQueryFallback:
    """Call the existing HCX JSON transport at most once per ``interpret``."""

    def __init__(
        self,
        settings: HcxSettings | None = None,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.settings = settings or HcxSettings.from_env()
        self.transport = transport or UrllibJsonTransport()
        self.call_count = 0

    def interpret(
        self,
        question: str,
        validation: QueryValidationResult,
    ) -> SemanticFallbackOutcome:
        start = perf_counter()
        if not self.settings.enabled:
            return SemanticFallbackOutcome(None, "disabled", _elapsed_ms(start))
        if not self.settings.configured:
            return SemanticFallbackOutcome(None, "not_configured", _elapsed_ms(start))
        if not validation.fallback_recommended:
            return SemanticFallbackOutcome(None, "not_needed", _elapsed_ms(start))

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": SEMANTIC_QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "allowed": {
                                "task_type": sorted(ALLOWED_TASK_TYPES),
                                "event_family": sorted(ALLOWED_EVENT_FAMILIES),
                                "operation": sorted(ALLOWED_OPERATIONS),
                            },
                            "locked_slots": {
                                name: slot.value
                                for name, slot in validation.slots.items()
                                if slot.locked
                            },
                            "unresolved_slots": [
                                *validation.missing_slots,
                                *validation.ambiguous_slots,
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(self.settings.max_tokens, 512),
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
            return SemanticFallbackOutcome(
                None,
                "transport_failure",
                _elapsed_ms(start),
                transport_status="failure",
            )
        except Exception:  # noqa: BLE001 - query must fail closed, never fail open
            return SemanticFallbackOutcome(
                None,
                "fallback_error",
                _elapsed_ms(start),
                transport_status="failure",
            )

        response_shape = _safe_response_shape(response)
        content = _response_content(response) if isinstance(response, Mapping) else None
        if content is None:
            return SemanticFallbackOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                response_shape=response_shape,
                parse_status="content_missing",
            )
        content_format, prefix_present, suffix_present = _content_shape(content)
        try:
            result = parse_semantic_query_result(content)
        except json.JSONDecodeError:
            return SemanticFallbackOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                response_shape=response_shape,
                content_present=True,
                parse_status="invalid_json",
                content_format=content_format,
                prefix_text_present=prefix_present,
                suffix_text_present=suffix_present,
            )
        except SemanticSchemaError as error:
            return SemanticFallbackOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                response_shape=response_shape,
                content_present=True,
                parse_status="schema_invalid",
                content_format=content_format,
                prefix_text_present=prefix_present,
                suffix_text_present=suffix_present,
                schema_error_code=error.code,
                schema_error_fields=error.fields,
            )
        except (TypeError, ValueError):
            return SemanticFallbackOutcome(
                None,
                "malformed_response",
                _elapsed_ms(start),
                transport_status="success",
                response_shape=response_shape,
                content_present=True,
                parse_status="schema_invalid",
                content_format=content_format,
                prefix_text_present=prefix_present,
                suffix_text_present=suffix_present,
                schema_error_code="invalid_type",
            )
        return SemanticFallbackOutcome(
            result,
            "success",
            _elapsed_ms(start),
            transport_status="success",
            response_shape=response_shape,
            content_present=True,
            parse_status="success",
            content_format=content_format,
            prefix_text_present=prefix_present,
            suffix_text_present=suffix_present,
        )


def parse_semantic_query_result(content: str | Mapping[str, Any]) -> SemanticQueryResult:
    if isinstance(content, Mapping):
        payload = dict(content)
    else:
        payload = _json_payload(str(content))
    if not isinstance(payload, Mapping):
        raise SemanticSchemaError("invalid_type", ("response",))

    payload_fields = {str(key) for key in payload}
    missing = _SEMANTIC_FIELDS - payload_fields
    if missing:
        raise SemanticSchemaError("missing_required", list(missing))
    unexpected = payload_fields - _SEMANTIC_FIELDS
    if unexpected:
        raise SemanticSchemaError("unexpected_field", list(unexpected))

    task = _enum_or_none(payload.get("task_type"), ALLOWED_TASK_TYPES, "task_type")
    event = _enum_or_none(
        payload.get("event_family"), ALLOWED_EVENT_FAMILIES, "event_family"
    )
    operation = _enum_or_none(
        payload.get("operation"), ALLOWED_OPERATIONS, "operation"
    )
    set_intent = payload.get("set_intent")
    if set_intent is not None and not isinstance(set_intent, bool):
        raise SemanticSchemaError("invalid_type", ("set_intent",))
    ambiguity = payload.get("ambiguity")
    if not isinstance(ambiguity, bool):
        raise SemanticSchemaError("invalid_type", ("ambiguity",))
    requested_state = payload.get("requested_state")
    if requested_state is not None and (
        not isinstance(requested_state, str)
        or not requested_state.strip()
        or len(requested_state) > 80
    ):
        raise SemanticSchemaError("invalid_type", ("requested_state",))
    interpretations = payload.get("possible_interpretations")
    if not isinstance(interpretations, list) or len(interpretations) > 5:
        raise SemanticSchemaError(
            "invalid_interpretation_shape", ("possible_interpretations",)
        )
    safe_interpretations = tuple(_safe_interpretation(item) for item in interpretations)
    return SemanticQueryResult(
        task_type=task,
        event_family=event,
        operation=operation,
        set_intent=set_intent,
        requested_state=requested_state.strip() if requested_state else None,
        ambiguity=ambiguity,
        possible_interpretations=safe_interpretations,
    )


def _enum_or_none(value: Any, allowed: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticSchemaError("invalid_type", (field,))
    if value not in allowed:
        raise SemanticSchemaError("invalid_enum", (field,))
    return value


def _safe_interpretation(value: Any) -> Any:
    if isinstance(value, str):
        if not value.strip() or len(value) > 80:
            raise SemanticSchemaError(
                "invalid_interpretation_shape", ("possible_interpretations",)
            )
        return value
    if not isinstance(value, Mapping):
        raise SemanticSchemaError(
            "invalid_interpretation_shape", ("possible_interpretations",)
        )
    fields = {str(key) for key in value}
    if not fields or fields - _INTERPRETATION_FIELDS:
        raise SemanticSchemaError(
            "invalid_interpretation_shape", ("possible_interpretations",)
        )
    if any(
        not isinstance(item, str) or not item.strip() or len(item) > 80
        for item in value.values()
    ):
        raise SemanticSchemaError(
            "invalid_interpretation_shape", ("possible_interpretations",)
        )
    for field, allowed in (
        ("task_type", ALLOWED_TASK_TYPES),
        ("event_family", ALLOWED_EVENT_FAMILIES),
        ("operation", ALLOWED_OPERATIONS),
    ):
        if field in value and value[field] not in allowed:
            raise SemanticSchemaError("invalid_enum", (field,))
    return {key: value[key] for key in _INTERPRETATION_FIELDS if key in value}


_JSON_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<payload>.*?)\r?\n?```",
    re.DOTALL | re.IGNORECASE,
)


def _json_payload(content: str) -> Any:
    """Decode exact JSON or one explicit JSON fence, never prose inference."""

    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as direct_error:
        fences = list(_JSON_FENCE.finditer(text))
        if len(fences) != 1:
            raise direct_error
        fence = fences[0]
        prefix = text[: fence.start()]
        suffix = text[fence.end() :]
        if "```" in prefix or "```" in suffix:
            raise direct_error
        return json.loads(fence.group("payload").strip())


def _content_shape(content: str) -> tuple[str, bool, bool]:
    text = str(content).strip()
    fences = list(_JSON_FENCE.finditer(text))
    if len(fences) == 1:
        fence = fences[0]
        return (
            "json_fence",
            bool(text[: fence.start()].strip()),
            bool(text[fence.end() :].strip()),
        )
    if text.startswith("{") and text.endswith("}"):
        return "json_object", False, False
    return "text", False, False


def _safe_response_shape(response: Any) -> str:
    """Describe only recognized wrapper paths and value types."""

    if not isinstance(response, Mapping):
        return f"response:{_safe_type(response)}"
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            return (
                "choices[0].message.content:"
                f"{_safe_type(message.get('content'))}"
            )
    result = response.get("result")
    if isinstance(result, Mapping):
        message = result.get("message")
        if isinstance(message, Mapping):
            return f"result.message.content:{_safe_type(message.get('content'))}"
    return "unrecognized"


def _safe_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "other"


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


__all__ = [
    "HcxSemanticQueryFallback",
    "SEMANTIC_QUERY_SYSTEM_PROMPT",
    "SemanticSchemaError",
    "SemanticFallbackOutcome",
    "SemanticQueryResult",
    "parse_semantic_query_result",
]
