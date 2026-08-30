"""HyperCLOVA X constrained verbalizer.

The verbalizer restates an already-verified :class:`GeneratedAnswer` in fluent
Korean.  It never sees retrieved chunks or database metadata, so it has no way
to introduce a fact the deterministic pipeline did not prove.  Every response is
checked by :mod:`app.generation.answer_validator`; anything that fails, errors,
or times out falls back to the deterministic text.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.generation.answer_generator import GeneratedAnswer
from app.generation.compact_claim import CompactClaim, build_compact_claim
from app.generation.answer_validator import ValidationPolicy
from app.generation.lossless_verbalization import (
    CITATION_ATTACHMENT_FAILED,
    REDUNDANT_UNIT_SUFFIX,
    LOSSLESS_VERBALIZER_SYSTEM_PROMPT,
    DetachedClaimInput,
    claim_event_count,
    detach_claim_citations,
    verify_lossless_candidate,
)
from app.generation.protected_literals import (
    PLACEHOLDER_DUPLICATED,
    PLACEHOLDER_MISSING,
    PLACEHOLDER_REORDERED,
    PLACEHOLDER_UNEXPECTED,
)

# The embedding adapter already owns a tested JSON transport with sanitized HTTP
# error semantics.  Reusing it keeps one network contract and leaves that
# frozen module untouched.
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


ENV_PREFIX = "FESTIVAL_HCX_"

#: CLOVA Studio's OpenAI compatibility API. The native /v3/chat-completions
#: route is deliberately not used: this contract keeps one JSON transport and
#: one response shape.
DEFAULT_ENDPOINT = (
    "https://clovastudio.stream.ntruss.com/v1/openai/chat/completions"
)
#: Verbalization restates verified facts, so the reasoning-oriented HCX-007
#: is not needed here.
DEFAULT_MODEL = "HCX-005"
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_TOKENS = 1024

#: A protected literal did not survive the round trip.
PLACEHOLDER_INTEGRITY_FAILED = "fallback_placeholder_integrity_failed"

#: No compact verified claim could be built.  This is a deliberate skip that
#: keeps the grounded deterministic answer, not an error.
SKIPPED_NO_COMPACT_CLAIM = "skipped_no_compact_verified_claim"

#: The claim covers more than one verified event.  Live runs showed the model
#: cannot restate those without reordering or dropping facts, so it is not
#: asked to.  A deliberate skip, not an error.
SKIPPED_MULTI_EVENT_CLAIM = "skipped_multi_event_compact_claim"

#: The answer carries a statement about what the question left open, and a
#: compact claim holds only verified facts -- so the model is never shown that
#: statement and its reply cannot contain it.  Restating the facts would
#: therefore delete it.  A deliberate skip, not an error.
SKIPPED_SEMANTIC_CONTROL_NOTICE = "skipped_semantic_control_notice"

#: Draft flags that mark an answer as carrying such a statement.  Structural,
#: so the text itself stays owned by the generator and is never matched here.
_SEMANTIC_CONTROL_FLAGS = ("under_specified", "exact_multi_match")

#: Citations could not be reattached to the events that own them.
CITATION_ATTACHMENT_FAILED_STATUS = "fallback_citation_attachment_failed"

#: The model appended a unit that the verified value or the claim
#: template already carried.
REDUNDANT_UNIT_STATUS = "fallback_redundant_unit_suffix"


@dataclass(frozen=True)
class HcxSettings:
    """Connection and generation settings for HyperCLOVA X."""

    enabled: bool = True
    endpoint: str = DEFAULT_ENDPOINT
    api_key: str = field(default="", repr=False)
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "HcxSettings":
        values = os.environ if environment is None else environment

        def read(name: str, default: str) -> str:
            return values.get(f"{ENV_PREFIX}{name}", default)

        return cls(
            enabled=_parse_bool(read("ENABLED", "true")),
            endpoint=read("API_URL", DEFAULT_ENDPOINT).strip(),
            api_key=read("API_KEY", "").strip(),
            api_key_header=read("API_KEY_HEADER", "Authorization"),
            api_key_prefix=read("API_KEY_PREFIX", "Bearer"),
            model=read("MODEL", DEFAULT_MODEL),
            timeout_seconds=float(read("TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
            max_tokens=int(read("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(read("TEMPERATURE", "0.0")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.api_key)

    def request_headers(self) -> dict[str, str]:
        value = " ".join(
            part for part in (self.api_key_prefix.strip(), self.api_key) if part
        )
        return {self.api_key_header: value}


@dataclass(frozen=True)
class VerbalizationOutcome:
    """Final answer text plus why it came from HCX or from the fallback."""

    text: str
    status: str
    reason: str | None = None

    @property
    def used_hcx(self) -> bool:
        return self.status == "success"


class HcxVerbalizer:
    """Rewrite a verified answer, or return the deterministic text unchanged."""

    def __init__(
        self,
        settings: HcxSettings | None = None,
        *,
        transport: JsonHttpTransport | None = None,
        policy: ValidationPolicy | None = None,
    ) -> None:
        self.settings = settings or HcxSettings.from_env()
        self.transport = transport or UrllibJsonTransport()
        self.policy = policy or ValidationPolicy()

    def verbalize(
        self,
        generated: GeneratedAnswer,
        *,
        draft: Any = None,
        resolution: Any = None,
        task_type: str | None = None,
        claim: CompactClaim | None = None,
        required_terms: Iterable[str] = (),
    ) -> VerbalizationOutcome:
        """Restate a compact verified claim, or return the deterministic answer.

        Production passes ``draft``/``resolution``/``task_type`` and the claim is
        derived from them.  ``claim`` is an escape hatch for diagnostics that
        already hold one; it never widens what the model is allowed to see.
        """

        deterministic = generated.answer_text

        if not self.settings.enabled:
            return VerbalizationOutcome(deterministic, "disabled")
        if not self.settings.configured:
            return VerbalizationOutcome(deterministic, "not_configured")
        if not generated.answerable:
            # An unsupported answer must never be made to sound confident.
            return VerbalizationOutcome(deterministic, "skipped_not_answerable")
        if claim is None:
            claim = build_compact_claim(draft, resolution, task_type=task_type)
        if claim is None:
            # Grounding beats fluency: without a compact verified claim the
            # model would be handed a whole report and would rewrite it.
            return VerbalizationOutcome(deterministic, SKIPPED_NO_COMPACT_CLAIM)

        # Live evidence, not caution for its own sake: single-event claims came
        # back clean fourteen times out of fourteen, while multi-event claims
        # failed every attempt by reordering fields across events or dropping
        # the company name.  This narrows only when HCX is called; compact-claim
        # eligibility and the caps behind it are untouched.
        if claim_event_count(claim) != 1:
            return VerbalizationOutcome(deterministic, SKIPPED_MULTI_EVENT_CLAIM)

        if _carries_semantic_control_notice(draft):
            # Last gate before the model, so the established skip statuses keep
            # their precedence.  Restating the claim would drop this answer's
            # statement about what the question left open: the claim carries
            # verified facts only, so that statement is not in what the model is
            # shown and cannot be in what it returns.  Whether it survives must
            # not depend on the model, nor on how much evidence retrieval
            # happened to serve.
            return VerbalizationOutcome(
                deterministic, SKIPPED_SEMANTIC_CONTROL_NOTICE
            )

        try:
            detached = detach_claim_citations(claim)
        except ValueError as error:
            return VerbalizationOutcome(
                deterministic, "fallback_error", type(error).__name__
            )

        try:
            masked_candidate = self._request(detached)
        except _VerbalizerFailure as failure:
            return VerbalizationOutcome(deterministic, failure.status, failure.reason)

        result = verify_lossless_candidate(
            masked_candidate, claim=claim, detached=detached
        )
        if not result.valid or result.final_answer is None:
            # Never repair a mangled reply: a guessed literal, or a citation
            # placed by inference, is worse than the deterministic answer.
            return VerbalizationOutcome(
                deterministic, _failure_status(result.reason), result.reason
            )
        return VerbalizationOutcome(result.final_answer, "success")

    def _request(self, detached: DetachedClaimInput) -> str:
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": LOSSLESS_VERBALIZER_SYSTEM_PROMPT},
                {"role": "user", "content": detached.protection.masked},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        try:
            response = self.transport.post_json(
                self.settings.endpoint,
                headers=self.settings.request_headers(),
                payload=payload,
                timeout_seconds=self.settings.timeout_seconds,
            )
        except EmbeddingHttpError as error:
            status = (
                "fallback_timeout"
                if error.status_code is None
                else "fallback_http_error"
            )
            raise _VerbalizerFailure(status, _error_reason(error)) from error
        except TimeoutError as error:
            raise _VerbalizerFailure("fallback_timeout", "request timed out") from error
        except Exception as error:  # noqa: BLE001 - never fail the API request
            raise _VerbalizerFailure(
                "fallback_error", type(error).__name__
            ) from error

        content = _response_content(response)
        if content is None:
            raise _VerbalizerFailure(
                "fallback_invalid_response", "no message content in response"
            )
        return content


#: Placeholder faults keep their own status; everything else a check rejects is
#: a validation fallback, with the specific check named in ``reason``.
_PLACEHOLDER_FAULTS = frozenset(
    {
        PLACEHOLDER_MISSING,
        PLACEHOLDER_DUPLICATED,
        PLACEHOLDER_REORDERED,
        PLACEHOLDER_UNEXPECTED,
    }
)

_CITATION_FAULTS = frozenset(
    {
        CITATION_ATTACHMENT_FAILED,
        "citation_sequence_mismatch",
        "citation_mapping_missing",
        "event_count_mismatch",
        "event_span_not_found",
        "event_field_text_mismatch",
    }
)


def _carries_semantic_control_notice(draft: Any) -> bool:
    """Whether this answer states something the model would silently drop.

    Reads the draft's own flags rather than the rendered text: the wording
    belongs to the generator, and a verbalizer that matched on it would own a
    copy of a sentence it does not write.  A draft without the mapping -- an
    older caller, or a diagnostic passing a claim directly -- reports nothing,
    so existing behaviour is unchanged.
    """

    ambiguity = getattr(draft, "ambiguity", None)
    if not isinstance(ambiguity, Mapping):
        return False
    return any(bool(ambiguity.get(flag)) for flag in _SEMANTIC_CONTROL_FLAGS)


def _failure_status(reason: str | None) -> str:
    if reason in _PLACEHOLDER_FAULTS:
        return PLACEHOLDER_INTEGRITY_FAILED
    if reason == REDUNDANT_UNIT_SUFFIX:
        return REDUNDANT_UNIT_STATUS
    if reason in _CITATION_FAULTS:
        return CITATION_ATTACHMENT_FAILED_STATUS
    return "fallback_validation_failed"


class _VerbalizerFailure(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason



def _response_content(response: Mapping[str, Any]) -> str | None:
    """Read the assistant text from an OpenAI-compatible or native CLOVA reply."""

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content

    result = response.get("result")
    if isinstance(result, Mapping):
        message = result.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _error_reason(error: EmbeddingHttpError) -> str:
    if error.status_code is None:
        return "transport error"
    return f"HTTP {error.status_code}"


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
