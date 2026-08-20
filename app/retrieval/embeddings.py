"""Provider-neutral embedding contracts and deterministic local test support."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


CLOVA_CONTEXT_LENGTH_ERROR_CODE = "40003"
CLOVA_DEFAULT_SEGMENT_MAX_CHARS = 1800


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "hash"
    model: str = "festival-hash-embedding"
    version: str = "v1"
    dimensions: int = 1024
    batch_size: int = 32
    max_length: int = 8192
    device: str = "cpu"
    cuda_oom_retry: bool = True
    min_batch_size: int = 1

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.version.strip():
            raise ValueError("embedding provider, model, and version must not be empty")
        if not 1 <= self.dimensions <= 2000:
            raise ValueError("embedding dimensions must be between 1 and 2000")
        if self.batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        if self.max_length <= 0:
            raise ValueError("embedding max length must be positive")
        if not 1 <= self.min_batch_size <= self.batch_size:
            raise ValueError("embedding min batch size must be between 1 and batch size")
        if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", self.device.strip().casefold()):
            raise ValueError("embedding device must be cpu, cuda, or cuda:<index>")

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "EmbeddingConfig":
        values = os.environ if environment is None else environment
        return cls(
            provider=values.get("FESTIVAL_EMBEDDING_PROVIDER", "hash"),
            model=values.get("FESTIVAL_EMBEDDING_MODEL", "festival-hash-embedding"),
            version=values.get("FESTIVAL_EMBEDDING_VERSION", "v1"),
            dimensions=int(values.get("FESTIVAL_EMBEDDING_DIMENSIONS", "1024")),
            batch_size=int(values.get("FESTIVAL_EMBEDDING_BATCH_SIZE", "32")),
            max_length=int(values.get("FESTIVAL_EMBEDDING_MAX_LENGTH", "8192")),
            device=values.get("FESTIVAL_EMBEDDING_DEVICE", "cpu"),
            cuda_oom_retry=_parse_bool(
                values.get("FESTIVAL_EMBEDDING_CUDA_OOM_RETRY", "true")
            ),
            min_batch_size=int(
                values.get("FESTIVAL_EMBEDDING_MIN_BATCH_SIZE", "1")
            ),
        )


class EmbeddingProvider(Protocol):
    config: EmbeddingConfig

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingDimensionMismatch(ValueError):
    """Raised when provider output cannot match the configured vector index."""


class EmbeddingHttpError(RuntimeError):
    """Sanitized HTTP failure with enough metadata for bounded retry decisions."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None,
        transient: bool,
        retry_after_seconds: float | None = None,
        response_error_code: str | None = None,
        response_error_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient
        self.retry_after_seconds = retry_after_seconds
        self.response_error_code = response_error_code
        self.response_error_message = response_error_message


class JsonHttpTransport(Protocol):
    """Injectable JSON transport so provider tests never need the network."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    """Small standard-library transport used only by an explicitly configured provider."""

    def __init__(self, *, opener: Any | None = None) -> None:
        self._opener = opener or urlopen

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **dict(headers)},
            method="POST",
        )
        try:
            with self._opener(  # nosec B310
                request, timeout=timeout_seconds
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            status_code = int(error.code)
            retry_after = _parse_retry_after(
                error.headers.get("Retry-After") if error.headers else None
            )
            response_error_code, response_error_message = _http_error_details(error)
            transient = status_code == 429 or 500 <= status_code <= 599
            code_suffix = (
                f" (error code {response_error_code})"
                if response_error_code is not None
                else ""
            )
            raise EmbeddingHttpError(
                f"embedding endpoint returned HTTP {status_code}{code_suffix}",
                status_code=status_code,
                transient=transient,
                retry_after_seconds=retry_after,
                response_error_code=response_error_code,
                response_error_message=response_error_message,
            ) from error
        except (URLError, TimeoutError) as error:
            raise EmbeddingHttpError(
                "embedding endpoint request failed before a response was received",
                status_code=None,
                transient=True,
            ) from error
        if not isinstance(value, Mapping):
            raise ValueError("embedding endpoint must return a JSON object")
        return value


@dataclass(frozen=True)
class HttpEmbeddingSettings:
    endpoint: str
    api_key: str = field(repr=False)
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("embedding API endpoint must not be empty")
        if urlparse(self.endpoint).scheme not in {"http", "https"}:
            raise ValueError("embedding API endpoint must use HTTP or HTTPS")
        if not self.api_key.strip():
            raise ValueError("embedding API key must not be empty")
        if not self.api_key_header.strip():
            raise ValueError("embedding API key header must not be empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("embedding API timeout must be positive")

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None
    ) -> "HttpEmbeddingSettings":
        values = os.environ if environment is None else environment
        return cls(
            endpoint=values.get("FESTIVAL_EMBEDDING_API_URL", ""),
            api_key=values.get("FESTIVAL_EMBEDDING_API_KEY", ""),
            api_key_header=values.get(
                "FESTIVAL_EMBEDDING_API_KEY_HEADER", "Authorization"
            ),
            api_key_prefix=values.get("FESTIVAL_EMBEDDING_API_KEY_PREFIX", "Bearer"),
            timeout_seconds=float(
                values.get("FESTIVAL_EMBEDDING_TIMEOUT_SECONDS", "60")
            ),
        )

    def request_headers(self) -> dict[str, str]:
        value = " ".join(
            part for part in (self.api_key_prefix.strip(), self.api_key) if part
        )
        return {self.api_key_header: value}


@dataclass(frozen=True)
class OpenAIEmbeddingRequestOptions:
    """Request-shape options for explicitly selected OpenAI-compatible APIs."""

    input_mode: str = "batch"
    encoding_format: str | None = None
    include_dimensions: bool = False
    long_text_fallback: bool = False
    segment_max_chars: int = CLOVA_DEFAULT_SEGMENT_MAX_CHARS

    def __post_init__(self) -> None:
        if self.input_mode not in {"batch", "sequential"}:
            raise ValueError("OpenAI embedding input mode must be batch or sequential")
        if self.encoding_format is not None and not self.encoding_format.strip():
            raise ValueError("embedding encoding format must not be empty")
        if self.segment_max_chars <= 0:
            raise ValueError("embedding segment max chars must be positive")

    @classmethod
    def clova_studio(
        cls, *, segment_max_chars: int = CLOVA_DEFAULT_SEGMENT_MAX_CHARS
    ) -> "OpenAIEmbeddingRequestOptions":
        return cls(
            input_mode="sequential",
            encoding_format="float",
            include_dimensions=True,
            long_text_fallback=True,
            segment_max_chars=segment_max_chars,
        )


class OpenAICompatibleEmbeddingProvider:
    """Production-ready adapter for the common ``/embeddings`` JSON contract."""

    def __init__(
        self,
        config: EmbeddingConfig,
        settings: HttpEmbeddingSettings,
        *,
        transport: JsonHttpTransport | None = None,
        request_options: OpenAIEmbeddingRequestOptions | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.transport = transport or UrllibJsonTransport()
        self.request_options = request_options or OpenAIEmbeddingRequestOptions()
        self._long_text_fallbacks = 0
        self._long_text_segments = 0

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.request_options.input_mode == "sequential":
            vectors: list[list[float]] = []
            for text in texts:
                try:
                    vectors.extend(self._embed_input(text, expected_count=1))
                except EmbeddingHttpError as error:
                    if not self._is_context_length_error(error):
                        raise
                    vectors.append(self._embed_long_text(text))
            return vectors
        return self._embed_input(list(texts), expected_count=len(texts))

    def embedding_statistics(self) -> dict[str, int]:
        """Return counters only; input text and credentials are never retained."""

        return {
            "long_text_fallbacks": self._long_text_fallbacks,
            "long_text_segments": self._long_text_segments,
        }

    def _is_context_length_error(self, error: EmbeddingHttpError) -> bool:
        return (
            self.request_options.long_text_fallback
            and error.status_code == 400
            and error.response_error_code == CLOVA_CONTEXT_LENGTH_ERROR_CODE
        )

    def _embed_long_text(self, text: str) -> list[float]:
        self._long_text_fallbacks += 1
        vectors = self._embed_safe_segments(text)
        self._long_text_segments += len(vectors)
        return _mean_pool_and_normalize(vectors, self.config.dimensions)

    def _embed_safe_segments(self, text: str) -> list[list[float]]:
        segment_limit = self.request_options.segment_max_chars
        if len(text) <= segment_limit:
            segment_limit = max(1, len(text) // 2)
        segments = _split_embedding_text(text, max_chars=segment_limit)
        if len(segments) <= 1:
            raise ValueError("CLOVA long-text fallback could not split the input")

        vectors: list[list[float]] = []
        for segment in segments:
            try:
                vectors.extend(self._embed_input(segment, expected_count=1))
            except EmbeddingHttpError as error:
                if not self._is_context_length_error(error) or len(segment) <= 1:
                    raise
                vectors.extend(self._embed_safe_segments(segment))
        return vectors

    def _embed_input(
        self, input_value: str | list[str], *, expected_count: int
    ) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": input_value,
        }
        if self.request_options.encoding_format is not None:
            payload["encoding_format"] = self.request_options.encoding_format
        if self.request_options.include_dimensions:
            payload["dimensions"] = self.config.dimensions
        response = self.transport.post_json(
            self.settings.endpoint,
            headers=self.settings.request_headers(),
            payload=payload,
            timeout_seconds=self.settings.timeout_seconds,
        )
        return _parse_openai_embedding_response(
            response,
            expected_count=expected_count,
            dimensions=self.config.dimensions,
        )


def _parse_openai_embedding_response(
    response: Mapping[str, Any], *, expected_count: int, dimensions: int
) -> list[list[float]]:
    """Validate and restore input order for an OpenAI-compatible response."""

    data = response.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError("embedding response must contain a data array")
    ordered: list[tuple[int, list[float]]] = []
    for fallback_index, item in enumerate(data):
        embedding = item.get("embedding") if isinstance(item, Mapping) else None
        if not isinstance(item, Mapping) or not isinstance(
            embedding, Sequence
        ) or isinstance(embedding, (str, bytes)):
            raise ValueError("embedding response item is malformed")
        index = int(item.get("index", fallback_index))
        vector = [float(value) for value in item["embedding"]]
        _validate_embedding(vector, dimensions)
        ordered.append((index, vector))
    ordered.sort(key=lambda item: item[0])
    if len(ordered) != expected_count:
        raise ValueError("embedding response count does not match input count")
    if [index for index, _ in ordered] != list(range(expected_count)):
        raise ValueError("embedding response indexes do not match input order")
    return [vector for _, vector in ordered]


class DeterministicHashEmbedder:
    """Dependency-free local/mock embedder; not intended as a semantic model."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.config.dimensions
        tokens = re.findall(r"[0-9A-Za-z가-힣]+", text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.config.dimensions
            vector[index] += 1.0 if digest[8] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def chunk_embedding_text(chunk: Mapping[str, object]) -> str:
    """Return the frozen retrieval text used for document embeddings."""

    value = str(chunk.get("retrieval_text") or "").strip()
    if not value:
        raise ValueError("chunk retrieval_text must not be empty")
    return value


def create_embedding_provider(
    config: EmbeddingConfig,
    *,
    environment: Mapping[str, str] | None = None,
    transport: JsonHttpTransport | None = None,
    bge_encoder: Any | None = None,
    bge_encoder_factory: Any | None = None,
) -> EmbeddingProvider:
    """Create a configured adapter without embedding any text."""

    provider = config.provider.strip().casefold().replace("-", "_")
    if provider == "hash":
        return DeterministicHashEmbedder(config)
    if provider in {"openai", "openai_compatible", "http"}:
        return OpenAICompatibleEmbeddingProvider(
            config,
            HttpEmbeddingSettings.from_env(environment),
            transport=transport,
        )
    if provider in {"clova", "clova_studio", "clova_openai_compatible"}:
        values = os.environ if environment is None else environment
        return OpenAICompatibleEmbeddingProvider(
            config,
            HttpEmbeddingSettings.from_env(environment),
            transport=transport,
            request_options=OpenAIEmbeddingRequestOptions.clova_studio(
                segment_max_chars=int(
                    values.get(
                        "FESTIVAL_EMBEDDING_LONG_TEXT_SEGMENT_CHARS",
                        str(CLOVA_DEFAULT_SEGMENT_MAX_CHARS),
                    )
                )
            ),
        )
    if provider in {"bge_m3_local", "bgem3_local"}:
        from app.retrieval.bge_m3 import BgeM3LocalEmbeddingProvider

        return BgeM3LocalEmbeddingProvider(
            config,
            encoder=bge_encoder,
            encoder_factory=bge_encoder_factory,
        )
    if provider in {"bge_m3_http", "bgem3_http"}:
        from app.retrieval.bge_m3 import BgeM3HttpEmbeddingProvider

        return BgeM3HttpEmbeddingProvider(
            config,
            HttpEmbeddingSettings.from_env(environment),
            transport=transport,
        )
    raise ValueError(f"unsupported embedding provider: {config.provider}")


def _validate_embedding(vector: Sequence[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise EmbeddingDimensionMismatch(
            f"embedding dimension mismatch: expected {dimensions}, got {len(vector)}"
        )
    if any(not math.isfinite(float(value)) for value in vector):
        raise EmbeddingDimensionMismatch("embedding values must be finite")


def _split_embedding_text(text: str, *, max_chars: int) -> list[str]:
    """Split without truncation, preferring paragraph, line, sentence, then space."""

    if max_chars <= 0:
        raise ValueError("embedding segment max chars must be positive")
    if not text:
        return []
    segments: list[str] = []
    start = 0
    while len(text) - start > max_chars:
        hard_end = start + max_chars
        minimum = start + max(1, max_chars // 2)
        split_at = _preferred_split_point(text, minimum, hard_end)
        if split_at <= start:
            split_at = hard_end
        segments.append(text[start:split_at])
        start = split_at
    if start < len(text):
        segments.append(text[start:])
    return segments


def _preferred_split_point(text: str, minimum: int, hard_end: int) -> int:
    for marker in ("\n\n", "\n", ". ", "\u3002", "! ", "? ", " "):
        position = text.rfind(marker, minimum, hard_end)
        if position >= minimum:
            return position + len(marker)
    return hard_end


def _mean_pool_and_normalize(
    vectors: Sequence[Sequence[float]], dimensions: int
) -> list[float]:
    if not vectors:
        raise EmbeddingDimensionMismatch("cannot pool an empty embedding collection")
    totals = [0.0] * dimensions
    for vector in vectors:
        _validate_embedding(vector, dimensions)
        for index, value in enumerate(vector):
            totals[index] += float(value)
    mean = [value / len(vectors) for value in totals]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm <= 0.0:
        raise EmbeddingDimensionMismatch("mean pooled embedding is a zero vector")
    normalized = [value / norm for value in mean]
    _validate_embedding(normalized, dimensions)
    return normalized


def should_retry_embedding_error(error: BaseException) -> bool:
    """Unknown provider errors retain legacy retry behavior; explicit permanent errors do not."""

    return getattr(error, "transient", None) is not False


def embedding_retry_delay(
    error: BaseException,
    configured_delay: float,
    max_delay: float | None = None,
) -> float:
    retry_after = getattr(error, "retry_after_seconds", None)
    delay = (
        configured_delay
        if retry_after is None
        else max(configured_delay, float(retry_after))
    )
    return min(delay, max_delay) if max_delay is not None else delay


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean embedding setting: {value!r}")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max((target - datetime.now(timezone.utc)).total_seconds(), 0.0)
        except (TypeError, ValueError, OverflowError):
            return None


def _http_error_details(error: HTTPError) -> tuple[str | None, str | None]:
    try:
        raw = error.read(65_536)
        value = json.loads(raw.decode("utf-8"))
    except (
        AttributeError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
    ):
        return None, None
    if not isinstance(value, Mapping):
        return None, None
    detail = value.get("error", value)
    if not isinstance(detail, Mapping):
        return None, None
    code = _sanitize_error_field(detail.get("code"), max_length=64, code=True)
    message = _sanitize_error_field(detail.get("message"), max_length=256)
    return code, message


def _sanitize_error_field(
    value: object, *, max_length: int, code: bool = False
) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if code:
        text = re.sub(r"[^0-9A-Za-z_.-]", "", text)
    return text[:max_length] or None
