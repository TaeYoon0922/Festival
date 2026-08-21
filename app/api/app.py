"""FastAPI application exposing the frozen disclosure agent as ``GET /answer``.

The pipeline is built on first use rather than at startup: ``PostgresBackend``
opens no connection until a query runs, so the server stays up when the database
is down and reports the outage per request instead of refusing to boot.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import AfterValidator

from app.api.pipeline import (
    INTERNAL_ERROR,
    AnswerPipeline,
    AnswerPipelineError,
)
from app.api.schemas import AnswerResponse, ErrorResponse


SERVICE_UNAVAILABLE = 503

INTERNAL_ERROR_MESSAGE = "The answer pipeline failed to complete."


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, AfterValidator(_not_blank)]


class _PipelineHolder:
    """Build the pipeline once, on the first request that needs it."""

    def __init__(self, factory: Callable[[], AnswerPipeline]) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._pipeline: AnswerPipeline | None = None

    def get(self) -> AnswerPipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._factory()
            return self._pipeline


def create_app(
    *, pipeline_factory: Callable[[], AnswerPipeline] | None = None
) -> FastAPI:
    holder = _PipelineHolder(pipeline_factory or AnswerPipeline.from_env)

    app = FastAPI(
        title="Disclosure Agent Answer API",
        description=(
            "Answers Korean disclosure questions from the frozen DART corpus "
            "using hybrid retrieval, deterministic reasoning, and a constrained "
            "HyperCLOVA X verbalizer."
        ),
        version="1.0.0",
    )
    app.state.pipeline_holder = holder

    @app.get(
        "/answer",
        response_model=AnswerResponse,
        responses={SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
    )
    def answer(  # noqa: D401 - blocking pipeline runs in the threadpool
        question_id: Annotated[NonBlank, Query(min_length=1)],
        question: Annotated[NonBlank, Query(min_length=1)],
    ) -> AnswerResponse:
        payload = holder.get().answer(question_id, question)
        return AnswerResponse.model_validate(payload)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(AnswerPipelineError)
    async def pipeline_unavailable(
        request: Request, error: AnswerPipelineError
    ) -> JSONResponse:
        del request
        return _error(error.reason, error.public_message)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, error: Exception) -> JSONResponse:
        """Never leak a traceback, DSN, or credential to a client."""

        del request, error
        return _error(INTERNAL_ERROR, INTERNAL_ERROR_MESSAGE)

    return app


def _error(reason: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=SERVICE_UNAVAILABLE,
        content={"reason": reason, "message": message},
    )


app = create_app()
