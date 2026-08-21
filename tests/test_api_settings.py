from __future__ import annotations

import ast
import unittest
from pathlib import Path

from app.api.settings import ApiSettings


EVALUATION_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "evaluate_postgres_agent_gold60.py"
)

#: Parameters that must be identical in the API and in the frozen evaluation.
SHARED_PARAMETERS = (
    "top_k",
    "lexical_top_n",
    "vector_top_n",
    "rrf_k",
    "lexical_weight",
    "vector_weight",
    "fusion_weight",
    "deterministic_weight",
    "rerank_mode",
    "rerank_window_size",
    "diagnostic_top_n",
)


def _evaluation_defaults() -> dict[str, object]:
    """Read argparse defaults out of the evaluation script without running it."""

    tree = ast.parse(EVALUATION_SCRIPT.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        name = next(
            (
                argument.value[2:].replace("-", "_")
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ),
            None,
        )
        if name is None:
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            try:
                defaults[name] = ast.literal_eval(keyword.value)
            except ValueError:
                # Computed defaults such as --output-dir are not shared settings.
                continue
    return defaults


class ApiSettingsParityTests(unittest.TestCase):
    def test_defaults_match_the_frozen_evaluation_script(self) -> None:
        expected = _evaluation_defaults()
        settings = ApiSettings()

        for name in SHARED_PARAMETERS:
            self.assertIn(name, expected, f"{name} is not declared by the evaluator")
            self.assertEqual(
                getattr(settings, name),
                expected[name],
                f"{name} drifted from the frozen Gold60 configuration",
            )

    def test_retrieval_config_carries_the_defaults(self) -> None:
        config = ApiSettings().retrieval_config()

        self.assertEqual(config.final_top_k, 10)
        self.assertEqual(config.lexical_top_n, 50)
        self.assertEqual(config.vector_top_n, 50)
        self.assertEqual(config.rerank_mode, "legacy")
        self.assertEqual(config.rerank_window_size, 2)
        self.assertEqual(config.rrf.k, 60)
        self.assertEqual(config.rrf.lexical_weight, 1.0)
        self.assertEqual(config.rrf.vector_weight, 1.0)


class ApiSettingsEnvironmentTests(unittest.TestCase):
    def test_empty_environment_keeps_the_frozen_defaults(self) -> None:
        self.assertEqual(ApiSettings.from_env({}), ApiSettings())

    def test_blank_values_are_ignored(self) -> None:
        settings = ApiSettings.from_env(
            {"FESTIVAL_API_TOP_K": "  ", "FESTIVAL_API_RERANK_MODE": ""}
        )

        self.assertEqual(settings.top_k, 10)
        self.assertEqual(settings.rerank_mode, "legacy")

    def test_database_connect_timeout_is_bounded_by_default(self) -> None:
        self.assertEqual(ApiSettings().db_connect_timeout_seconds, 10)
        self.assertEqual(
            ApiSettings.from_env(
                {"FESTIVAL_API_DB_CONNECT_TIMEOUT_SECONDS": "3"}
            ).db_connect_timeout_seconds,
            3,
        )

    def test_overrides_are_applied(self) -> None:
        settings = ApiSettings.from_env(
            {
                "FESTIVAL_API_TOP_K": "5",
                "FESTIVAL_API_FUSION_WEIGHT": "0.75",
                "FESTIVAL_API_RERANK_MODE": "bounded",
                "FESTIVAL_API_DIAGNOSTIC_TOP_N": "20",
            }
        )

        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.fusion_weight, 0.75)
        self.assertEqual(settings.rerank_mode, "bounded")
        self.assertEqual(settings.diagnostic_top_n, 20)


if __name__ == "__main__":
    unittest.main()
