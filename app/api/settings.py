"""Retrieval settings for the answer API.

The defaults mirror ``scripts/evaluate_postgres_agent_gold60.py`` exactly so the
served pipeline and the frozen Gold60 evaluation share one configuration.
``tests/test_api_settings.py`` asserts that parity, which is why the numbers are
written out here instead of being imported from the evaluation script.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.retrieval.hybrid import HybridRetrievalConfig, RRFConfig


ENV_PREFIX = "FESTIVAL_API_"


@dataclass(frozen=True)
class ApiSettings:
    """Frozen Gold60 retrieval parameters, overridable for local diagnostics."""

    top_k: int = 10
    lexical_top_n: int = 50
    vector_top_n: int = 50
    rrf_k: int = 60
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    fusion_weight: float = 0.60
    deterministic_weight: float = 0.40
    rerank_mode: str = "legacy"
    rerank_window_size: int = 2
    diagnostic_top_n: int | None = None
    #: Bounds how long a request waits on an unreachable database.
    db_connect_timeout_seconds: int = 10

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "ApiSettings":
        values = os.environ if environment is None else environment

        def read(name: str, cast: Any, default: Any) -> Any:
            raw = values.get(f"{ENV_PREFIX}{name.upper()}")
            return default if raw is None or not raw.strip() else cast(raw)

        diagnostic_raw = values.get(f"{ENV_PREFIX}DIAGNOSTIC_TOP_N")
        return cls(
            top_k=read("top_k", int, cls.top_k),
            lexical_top_n=read("lexical_top_n", int, cls.lexical_top_n),
            vector_top_n=read("vector_top_n", int, cls.vector_top_n),
            rrf_k=read("rrf_k", int, cls.rrf_k),
            lexical_weight=read("lexical_weight", float, cls.lexical_weight),
            vector_weight=read("vector_weight", float, cls.vector_weight),
            fusion_weight=read("fusion_weight", float, cls.fusion_weight),
            deterministic_weight=read(
                "deterministic_weight", float, cls.deterministic_weight
            ),
            rerank_mode=read("rerank_mode", str, cls.rerank_mode),
            rerank_window_size=read(
                "rerank_window_size", int, cls.rerank_window_size
            ),
            diagnostic_top_n=(
                None
                if diagnostic_raw is None or not diagnostic_raw.strip()
                else int(diagnostic_raw)
            ),
            db_connect_timeout_seconds=read(
                "db_connect_timeout_seconds", int, cls.db_connect_timeout_seconds
            ),
        )

    def retrieval_config(self) -> HybridRetrievalConfig:
        return HybridRetrievalConfig(
            lexical_top_n=self.lexical_top_n,
            vector_top_n=self.vector_top_n,
            final_top_k=self.top_k,
            fusion_weight=self.fusion_weight,
            deterministic_weight=self.deterministic_weight,
            rerank_mode=self.rerank_mode,
            rerank_window_size=self.rerank_window_size,
            diagnostic_top_n=self.diagnostic_top_n,
            rrf=RRFConfig(
                k=self.rrf_k,
                lexical_weight=self.lexical_weight,
                vector_weight=self.vector_weight,
            ),
        )
