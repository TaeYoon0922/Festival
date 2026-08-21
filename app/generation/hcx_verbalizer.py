"""HyperCLOVA X constrained verbalizer.

The verbalizer restates an already-verified :class:`GeneratedAnswer` in fluent
Korean.  It never sees retrieved chunks or database metadata, so it has no way
to introduce a fact the deterministic pipeline did not prove.  Every response is
checked by :mod:`app.generation.answer_validator`; anything that fails, errors,
or times out falls back to the deterministic text.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.generation.answer_generator import GeneratedAnswer
from app.generation.answer_validator import (
    ValidationPolicy,
    validate_verbalized_answer,
)
from app.generation.protected_literals import (
    ProtectedText,
    check_placeholder_integrity,
    contains_placeholder_syntax,
    protect_literals,
    restore_literals,
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

SYSTEM_PROMPT = """당신은 이미 검증된 공시 답변을 자연스러운 한국어로 다듬는 편집자입니다.

반드시 지킬 것:
- 입력에 있는 사실만 사용한다. 새로운 사실을 추가하지 않는다.
- __FESTIVAL_...__ 형태의 토큰은 보호된 값이다. 글자 하나도 바꾸지 말고
  개수와 순서를 입력 그대로 유지한다. 삭제, 추가, 중복, 번역, 분할하지 않는다.
- 보호 토큰 안쪽이나 바로 옆에 다른 문자를 끼워 넣지 않는다.
- 보호 토큰이 무엇을 뜻하는지 추측하거나 숫자, 날짜로 바꿔 쓰지 않는다.
- 외부 지식, 추정, 투자 의견, 전망, 추천을 쓰지 않는다.
- Markdown 서식을 새로 만들지 않는다. 굵게(**), 기울임(*), 제목(#),
  목록(-), 코드 블록을 추가하지 않는다.
- 원문보다 길게 쓰지 않는다.

출력은 다듬어진 답변 본문만 쓴다. 설명이나 머리말을 붙이지 않는다."""


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
        required_terms: Iterable[str] = (),
    ) -> VerbalizationOutcome:
        reference = generated.answer_text

        if not self.settings.enabled:
            return VerbalizationOutcome(reference, "disabled")
        if not self.settings.configured:
            return VerbalizationOutcome(reference, "not_configured")
        if not generated.answerable:
            # An unsupported answer must never be made to sound confident.
            return VerbalizationOutcome(reference, "skipped_not_answerable")

        if contains_placeholder_syntax(reference):
            # Protection would be ambiguous, so the model is not consulted.
            return VerbalizationOutcome(
                reference,
                PLACEHOLDER_INTEGRITY_FAILED,
                "reference already contains placeholder syntax",
            )

        protection = protect_literals(reference)
        try:
            masked_candidate = self._request(generated, protection)
        except _VerbalizerFailure as failure:
            return VerbalizationOutcome(reference, failure.status, failure.reason)

        integrity = check_placeholder_integrity(masked_candidate, protection)
        if not integrity.valid:
            # Never repair a mangled placeholder: a guessed literal is worse
            # than the deterministic answer.
            return VerbalizationOutcome(
                reference, PLACEHOLDER_INTEGRITY_FAILED, integrity.reason
            )

        candidate = restore_literals(masked_candidate, protection)
        result = validate_verbalized_answer(
            candidate,
            reference=reference,
            required_terms=required_terms,
            policy=self.policy,
        )
        if not result.valid:
            return VerbalizationOutcome(
                reference, "fallback_validation_failed", result.reason
            )
        return VerbalizationOutcome(candidate.strip(), "success")

    def _request(self, generated: GeneratedAnswer, protection: ProtectedText) -> str:
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _verified_facts(generated, protection)},
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


class _VerbalizerFailure(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _verified_facts(generated: GeneratedAnswer, protection: ProtectedText) -> str:
    """Serialize the masked answer and nothing that would unmask it.

    ``answer_text`` is rendered from the sections, so sending the sections too
    would hand the model a second, unprotected copy of every number, date, and
    citation marker — exactly what the placeholders exist to prevent.  Only the
    section titles survive, as structural hints that carry no literals.
    """

    facts = {
        "question": generated.question,
        "answer_to_rewrite": protection.masked,
        "section_titles": [section.title for section in generated.sections],
        "protected_tokens": list(protection.placeholders),
    }
    return json.dumps(facts, ensure_ascii=False, indent=2)


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
