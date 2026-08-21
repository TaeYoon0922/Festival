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

# The embedding adapter already owns a tested JSON transport with sanitized HTTP
# error semantics.  Reusing it keeps one network contract and leaves that
# frozen module untouched.
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


ENV_PREFIX = "FESTIVAL_HCX_"

DEFAULT_MODEL = "HCX-005"
DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_TOKENS = 1024

SYSTEM_PROMPT = """당신은 이미 검증된 공시 답변을 자연스러운 한국어로 다듬는 편집자입니다.

반드시 지킬 것:
- 입력에 있는 사실만 사용한다. 새로운 사실을 추가하지 않는다.
- 숫자, 날짜, 기업명은 입력에 적힌 표기를 문자 그대로 유지한다.
  예: "2,967,759주"를 "약 297만 주"로 바꾸지 않는다.
- 인용 표기([1], [2] 등)를 모두 그대로 유지한다. 추가하거나 삭제하지 않는다.
- 외부 지식, 추정, 투자 의견, 전망, 추천을 쓰지 않는다.
- 원문보다 길게 쓰지 않는다.

출력은 다듬어진 답변 본문만 쓴다. 설명이나 머리말을 붙이지 않는다."""


@dataclass(frozen=True)
class HcxSettings:
    """Connection and generation settings for HyperCLOVA X."""

    enabled: bool = True
    endpoint: str = ""
    api_key: str = field(default="", repr=False)
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    top_p: float = 0.1

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "HcxSettings":
        values = os.environ if environment is None else environment

        def read(name: str, default: str) -> str:
            return values.get(f"{ENV_PREFIX}{name}", default)

        return cls(
            enabled=_parse_bool(read("ENABLED", "true")),
            endpoint=read("API_URL", "").strip(),
            api_key=read("API_KEY", "").strip(),
            api_key_header=read("API_KEY_HEADER", "Authorization"),
            api_key_prefix=read("API_KEY_PREFIX", "Bearer"),
            model=read("MODEL", DEFAULT_MODEL),
            timeout_seconds=float(read("TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
            max_tokens=int(read("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
            temperature=float(read("TEMPERATURE", "0.0")),
            top_p=float(read("TOP_P", "0.1")),
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
        return self.status == "applied"


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

        try:
            candidate = self._request(generated)
        except _VerbalizerFailure as failure:
            return VerbalizationOutcome(reference, failure.status, failure.reason)

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
        return VerbalizationOutcome(candidate.strip(), "applied")

    def _request(self, generated: GeneratedAnswer) -> str:
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _verified_facts(generated)},
            ],
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
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


def _verified_facts(generated: GeneratedAnswer) -> str:
    """Serialize only what the deterministic pipeline already proved."""

    facts = {
        "question": generated.question,
        "deterministic_answer": generated.answer_text,
        "sections": [
            {
                "title": section.title,
                "content": section.content,
                "citations": list(section.citations),
                "metadata": list(section.metadata),
            }
            for section in generated.sections
        ],
        "citation_ids": [citation.citation_id for citation in generated.citations],
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
