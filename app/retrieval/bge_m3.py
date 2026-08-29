"""Dense-only BGE-M3 adapters for local FlagEmbedding and HTTP inference."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingDimensionMismatch,
    HttpEmbeddingSettings,
    JsonHttpTransport,
    UrllibJsonTransport,
)


BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIMENSIONS = 1024
BGE_M3_MAX_LENGTH = 8192

#: A commit SHA is the only revision that cannot move.  Every vector is stored
#: under ``embedding_version``, and retrieval compares vectors by that column, so
#: a branch or tag name would let one label describe two different models.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

#: What a dense BGE-M3 load actually consumes, established by loading the pinned
#: snapshot with files withheld one at a time.  README, ``.gitattributes`` and
#: the sparse/ColBERT heads are absent from a working load and are not required.
_REQUIRED_SNAPSHOT_FILES = ("config.json", "model.safetensors")

#: Any one of these is enough for the tokenizer.  One is still required: a
#: snapshot without them loads, but only by sourcing a tokenizer from elsewhere,
#: which is the identity drift this module exists to prevent.
_TOKENIZER_SNAPSHOT_FILES = ("tokenizer.json", "sentencepiece.bpe.model")

#: Fetch only what dense inference needs.  Asking for the whole repository would
#: reject a cache that is complete for our purposes but partial for the repo.
_SNAPSHOT_ALLOW_PATTERNS = ("*.json", "*.model", "model.safetensors", "1_Pooling/*")


class BgeM3IdentityError(RuntimeError):
    """The model that loaded is not provably the model that was configured."""


@dataclass(frozen=True)
class ResolvedSnapshot:
    """A local checkout proven to be the configured commit."""

    model: str
    revision: str
    local_path: str


class BgeM3Encoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


EncoderFactory = Callable[[EmbeddingConfig], BgeM3Encoder]
OomCleanup = Callable[[], None]


class BgeM3LocalEmbeddingProvider:
    """Lazy local FlagEmbedding adapter; dense vectors only."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        encoder: BgeM3Encoder | None = None,
        encoder_factory: EncoderFactory | None = None,
        clock: Callable[[], float] = time.perf_counter,
        oom_cleanup: OomCleanup | None = None,
    ) -> None:
        _validate_bge_config(config)
        self.config = config
        self._encoder = encoder
        self._encoder_factory = encoder_factory or _load_flag_embedding_encoder
        self._clock = clock
        self._oom_cleanup = oom_cleanup or _clear_cuda_cache
        self.load_time_seconds = 0.0
        self.effective_batch_size = config.batch_size
        self.oom_retries = 0
        #: Set once an encoder is built. ``None`` until then, and for injected
        #: encoders whose provenance this provider cannot vouch for.
        self.resolved_snapshot: ResolvedSnapshot | None = getattr(
            encoder, "festival_resolved_snapshot", None)

    def load(self) -> None:
        self._ensure_encoder()

    def identity(self) -> dict[str, Any]:
        """What was configured beside what actually loaded.

        The two are reported separately on purpose: a diagnostic that echoed the
        configured revision back as the resolved one would have hidden the very
        defect this module was written for.
        """

        snapshot = self.resolved_snapshot
        return {
            "configured_model": self.config.model,
            "configured_revision": self.config.version,
            "resolved_model": snapshot.model if snapshot else None,
            "resolved_revision": snapshot.revision if snapshot else None,
            "resolved_snapshot_path": snapshot.local_path if snapshot else None,
            "requested_device": self.config.device,
            "verified": bool(snapshot and snapshot.revision == self.config.version),
        }

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def _ensure_encoder(self) -> BgeM3Encoder:
        if self._encoder is None:
            started = self._clock()
            self._encoder = self._encoder_factory(self.config)
            self.load_time_seconds = self._clock() - started
            self.resolved_snapshot = getattr(
                self._encoder, "festival_resolved_snapshot", None)
        return self._encoder

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = self._ensure_encoder()
        while True:
            try:
                output = encoder.encode(
                    list(texts),
                    batch_size=self.effective_batch_size,
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                break
            except Exception as error:
                if not self._can_retry_cuda_oom(error):
                    raise
                self.effective_batch_size = max(
                    self.config.min_batch_size, self.effective_batch_size // 2
                )
                self.oom_retries += 1
                self._oom_cleanup()
        return _normalized_dense_vectors(output, self.config.dimensions, len(texts))

    def _can_retry_cuda_oom(self, error: BaseException) -> bool:
        return (
            self.config.cuda_oom_retry
            and self.config.device.casefold().startswith("cuda")
            and self.effective_batch_size > self.config.min_batch_size
            and _is_cuda_out_of_memory(error)
        )

    def peak_device_memory_bytes(self) -> int | None:
        """Return CUDA allocator peak when PyTorch exposes it; otherwise ``None``."""

        if not self.config.device.casefold().startswith("cuda"):
            return None
        try:
            import torch

            return int(torch.cuda.max_memory_allocated(self.config.device))
        except (ImportError, RuntimeError, TypeError, ValueError):
            return None


class BgeM3HttpEmbeddingProvider:
    """HTTP adapter for a dense-only service wrapping BGE-M3 inference."""

    def __init__(
        self,
        config: EmbeddingConfig,
        settings: HttpEmbeddingSettings,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        _validate_bge_config(config)
        self.config = config
        self.settings = settings
        self.transport = transport or UrllibJsonTransport()
        self.load_time_seconds = 0.0

    def load(self) -> None:
        """The remote service owns model loading; no request is made here."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.transport.post_json(
            self.settings.endpoint,
            headers=self.settings.request_headers(),
            payload={
                "model": self.config.model,
                "version": self.config.version,
                "texts": list(texts),
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "normalize_embeddings": True,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            },
            timeout_seconds=self.settings.timeout_seconds,
        )
        return _normalized_dense_vectors(response, self.config.dimensions, len(texts))


def _resolve_verified_bge_snapshot(
    config: EmbeddingConfig,
    *,
    downloader: Callable[..., str] | None = None,
) -> ResolvedSnapshot:
    """Resolve the configured commit ourselves and prove what we got.

    ``FlagEmbedding`` accepts a ``revision`` argument and does not honour it: asked
    for a pinned commit it resolved ``main`` instead.  Nothing downstream noticed,
    because every stored row is stamped with the *configured* revision.  So the
    revision is resolved here, the resolved commit is checked against the one that
    was asked for, and only the verified local directory is handed onward.

    The cache is consulted before the network, so a warm, already-verified pin
    loads offline; the network is used only when the pin is genuinely absent.
    """

    model = str(config.model or "").strip()
    revision = str(config.version or "").strip()
    if not _COMMIT_SHA.fullmatch(revision):
        raise BgeM3IdentityError(
            f"bge_m3_local requires an immutable 40-character commit SHA as the "
            f"embedding version; got {revision!r}. A branch or tag name would let "
            f"one stored embedding_version describe two different models."
        )

    download = downloader or _snapshot_downloader()
    patterns = list(_SNAPSHOT_ALLOW_PATTERNS)
    try:
        path = download(repo_id=model, revision=revision, allow_patterns=patterns,
                        local_files_only=True)
    except Exception:  # noqa: BLE001 - absent or partial cache; try the network
        path = download(repo_id=model, revision=revision, allow_patterns=patterns)

    local = Path(str(path))
    resolved = local.name
    if resolved != revision:
        raise BgeM3IdentityError(
            f"resolved snapshot does not match the configured revision: "
            f"model={model!r} configured={revision!r} resolved={resolved!r}"
        )

    missing = [name for name in _REQUIRED_SNAPSHOT_FILES
               if not (local / name).is_file()]
    if not any((local / name).is_file() for name in _TOKENIZER_SNAPSHOT_FILES):
        missing.append(" or ".join(_TOKENIZER_SNAPSHOT_FILES))
    if missing:
        raise BgeM3IdentityError(
            f"snapshot {resolved!r} of {model!r} is missing files required for "
            f"dense inference: {', '.join(missing)}"
        )
    return ResolvedSnapshot(model=model, revision=resolved, local_path=str(local))


def _snapshot_downloader() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "bge_m3_local requires the optional requirements-embedding.txt dependencies"
        ) from error
    return snapshot_download


def _load_flag_embedding_encoder(config: EmbeddingConfig) -> BgeM3Encoder:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        raise RuntimeError(
            "bge_m3_local requires the optional requirements-embedding.txt dependencies"
        ) from error
    snapshot = _resolve_verified_bge_snapshot(config)
    # The verified directory, never the repository id: given a path there is no
    # second resolution for FlagEmbedding to perform, so ``main`` cannot be
    # substituted. Passing ``revision`` here would be meaningless and misleading.
    encoder = BGEM3FlagModel(
        snapshot.local_path,
        normalize_embeddings=True,
        use_fp16=config.device.casefold().startswith("cuda"),
        devices=config.device,
    )
    # Carried on the encoder so the provider can report what actually loaded
    # rather than echoing what was requested.
    try:
        encoder.festival_resolved_snapshot = snapshot
    except AttributeError:  # pragma: no cover - exotic encoder objects
        pass
    return encoder


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    return name == "outofmemoryerror" or any(
        marker in message
        for marker in ("cuda out of memory", "cuda error: out of memory")
    )


def _clear_cuda_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _validate_bge_config(config: EmbeddingConfig) -> None:
    if config.dimensions != BGE_M3_DIMENSIONS:
        raise EmbeddingDimensionMismatch(
            f"BGE-M3 dense embeddings require {BGE_M3_DIMENSIONS} dimensions"
        )
    if not 1 <= config.max_length <= BGE_M3_MAX_LENGTH:
        raise ValueError(
            f"BGE-M3 max_length must be between 1 and {BGE_M3_MAX_LENGTH}"
        )


def _normalized_dense_vectors(
    output: Any, dimensions: int, expected_count: int
) -> list[list[float]]:
    value = output
    if isinstance(output, Mapping):
        for key in ("dense_vecs", "embeddings", "vectors"):
            if key in output:
                value = output[key]
                break
        else:
            data = output.get("data")
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                ordered: list[tuple[int, Any]] = []
                for fallback_index, item in enumerate(data):
                    if not isinstance(item, Mapping) or "embedding" not in item:
                        raise ValueError("BGE-M3 response data item is malformed")
                    ordered.append(
                        (int(item.get("index", fallback_index)), item["embedding"])
                    )
                ordered.sort(key=lambda item: item[0])
                if [index for index, _ in ordered] != list(range(len(ordered))):
                    raise ValueError("BGE-M3 response indexes do not match input order")
                value = [vector for _, vector in ordered]
            else:
                raise ValueError("BGE-M3 response does not contain dense vectors")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("BGE-M3 dense vectors must be an array")
    if len(value) != expected_count:
        raise ValueError(
            f"BGE-M3 response count mismatch: expected {expected_count}, got {len(value)}"
        )
    vectors: list[list[float]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise ValueError("BGE-M3 dense vector is malformed")
        vector = [float(component) for component in item]
        if len(vector) != dimensions:
            raise EmbeddingDimensionMismatch(
                f"BGE-M3 dimension mismatch: expected {dimensions}, got {len(vector)}"
            )
        if any(not math.isfinite(component) for component in vector):
            raise EmbeddingDimensionMismatch("BGE-M3 vector contains a non-finite value")
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 0.0:
            raise EmbeddingDimensionMismatch("BGE-M3 returned a zero dense vector")
        vectors.append([component / norm for component in vector])
    return vectors
