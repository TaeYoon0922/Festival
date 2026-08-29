"""Refuse to run a BGE-M3 evaluation that is not actually a BGE-M3 evaluation.

Two things can quietly turn this benchmark into a lie.  The provider can fall
back to the deterministic hash embedder, which has the *same 1024 dimensions*, so
no shape check catches it.  Or the corpus can hold vectors from a different model
or revision, in which case the vector SQL -- which filters on
``embedding_model``/``embedding_version`` -- matches nothing, returns no rows, and
the hybrid pipeline silently degrades to lexical-only while still reporting a
number.

Both are failures, not fallbacks.  Every check here raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: The identity this evaluation is defined against.  Anything else is a different
#: experiment and must not be reported under the same name.
PINNED_MODEL = "BAAI/bge-m3"
PINNED_REVISION = "6892b95fed65c899a30896eb40d619ae284d0455"
PINNED_DIMENSIONS = 1024
ALLOWED_PROVIDERS = frozenset({"bge_m3_local", "bgem3_local"})
FORBIDDEN_PROVIDERS = frozenset({"hash"})


class PreflightError(RuntimeError):
    """Raised instead of producing a result that would misstate its own basis."""


def _normalized(provider: object) -> str:
    return str(provider or "").strip().casefold().replace("-", "_")


def assert_embedding_identity(config: Any, *, require_device: str | None = "cuda") -> None:
    """The provider, model, revision, dimension and device must all be the pinned ones."""

    provider = _normalized(getattr(config, "provider", None))
    if provider in FORBIDDEN_PROVIDERS:
        raise PreflightError(
            "hash embeddings are forbidden for a BGE-M3 evaluation: "
            f"provider={provider!r}"
        )
    if provider not in ALLOWED_PROVIDERS:
        raise PreflightError(
            f"provider must be one of {sorted(ALLOWED_PROVIDERS)}, got {provider!r}"
        )
    model = str(getattr(config, "model", "") or "").strip()
    if model != PINNED_MODEL:
        raise PreflightError(f"model must be {PINNED_MODEL!r}, got {model!r}")
    revision = str(getattr(config, "version", "") or "").strip()
    if revision != PINNED_REVISION:
        raise PreflightError(
            f"model revision must be {PINNED_REVISION!r}, got {revision!r}"
        )
    dimensions = int(getattr(config, "dimensions", 0) or 0)
    if dimensions != PINNED_DIMENSIONS:
        raise PreflightError(
            f"dimensions must be {PINNED_DIMENSIONS}, got {dimensions}"
        )
    if require_device is not None:
        device = str(getattr(config, "device", "") or "").strip().casefold()
        if not device.startswith(require_device):
            raise PreflightError(
                f"this baseline is defined on {require_device!r}; refusing "
                f"device={device!r} rather than silently benchmarking elsewhere"
            )


def assert_stored_identity(rows: Iterable[Mapping[str, Any]]) -> None:
    """Every stored embedding row consulted must carry the pinned identity."""

    seen = {
        (str(r.get("embedding_model")), str(r.get("embedding_version")),
         int(r.get("embedding_dimensions") or 0))
        for r in rows
    }
    expected = (PINNED_MODEL, PINNED_REVISION, PINNED_DIMENSIONS)
    unexpected = sorted(s for s in seen if s != expected)
    if unexpected:
        raise PreflightError(
            f"stored embeddings carry a foreign identity: {unexpected}"
        )


def assert_vector_coverage(coverage: Mapping[str, Mapping[str, int]]) -> None:
    """No accepted question may reach the vector lane with nothing to score.

    ``coverage`` maps question_id -> {"eligible": n, "embedded": n}.  A question
    with eligible candidates but no matching BGE rows would still return a
    lexical-only answer, and that answer must never be labelled BGE-M3.
    """

    starved = sorted(
        qid for qid, counts in coverage.items()
        if int(counts.get("eligible", 0)) > 0 and int(counts.get("embedded", 0)) == 0
    )
    if starved:
        raise PreflightError(
            "questions have vector-eligible candidates but zero BGE-M3 rows, "
            f"which would silently degrade to lexical-only: {starved}"
        )
    incomplete = {
        qid: (int(c.get("eligible", 0)), int(c.get("embedded", 0)))
        for qid, c in coverage.items()
        if int(c.get("embedded", 0)) < int(c.get("eligible", 0))
    }
    if incomplete:
        raise PreflightError(
            f"incomplete BGE-M3 coverage (question: eligible/embedded): {incomplete}"
        )


def assert_manifest_current(manifest: Mapping[str, Any],
                            expected: Mapping[str, Any]) -> None:
    """A candidate union is only valid for the state it was collected against."""

    drifted = {
        key: (manifest.get(key), expected.get(key))
        for key in ("corpus_snapshot", "question_set", "candidate_union_hash",
                    "embedding_model", "embedding_revision", "retrieval_config_hash")
        if key in expected and manifest.get(key) != expected.get(key)
    }
    if drifted:
        raise PreflightError(f"stale evaluation manifest (field: stored/expected): {drifted}")


def pinned_encoder_factory(config: Any) -> Any:
    """Load BGE-M3 from the pinned commit, proven by path, not by a kwarg.

    ``FlagEmbedding`` accepts a ``revision`` argument and does not honour it -- it
    resolves ``main`` regardless -- so trusting that kwarg would silently
    benchmark whatever the Hub happens to serve today.  Resolving the snapshot
    ourselves and asserting the directory is named for the pinned commit turns
    model identity into something checked rather than requested.
    """

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        PINNED_MODEL,
        revision=PINNED_REVISION,
        allow_patterns=["*.json", "*.model", "model.safetensors", "1_Pooling/*"],
    )
    resolved = Path(path)
    if resolved.name != PINNED_REVISION:
        raise PreflightError(
            f"resolved snapshot {resolved.name!r} is not the pinned revision "
            f"{PINNED_REVISION!r}"
        )
    if not (resolved / "model.safetensors").exists():
        raise PreflightError(f"pinned snapshot {resolved} has no model.safetensors")

    from FlagEmbedding import BGEM3FlagModel

    device = str(getattr(config, "device", "cpu"))
    return BGEM3FlagModel(
        str(resolved),
        normalize_embeddings=True,
        use_fp16=device.casefold().startswith("cuda"),
        devices=device,
    )


def describe(config: Any) -> dict[str, Any]:
    """The identity fields a result file must carry to be reproducible."""

    return {
        "embedding_provider": getattr(config, "provider", None),
        "embedding_model": getattr(config, "model", None),
        "embedding_revision": getattr(config, "version", None),
        "embedding_dimensions": getattr(config, "dimensions", None),
        "embedding_device": getattr(config, "device", None),
    }
