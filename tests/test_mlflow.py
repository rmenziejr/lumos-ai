from __future__ import annotations

from lumosai.mlflow import resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.settings import Settings


def test_resolve_experiment_name_prefers_argument() -> None:
    loaded = Settings()

    assert resolve_experiment_name("explicit", loaded) == "explicit"


def test_resolve_experiment_name_uses_default_setting() -> None:
    loaded = Settings()
    loaded.mlflow.default_experiment_name = "default"

    assert resolve_experiment_name(None, loaded) == "default"


def test_log_result_without_experiment_returns_original_result() -> None:
    from lumosai.mlflow import log_result

    result = LumosResult(metrics={"performance/f1": 1.0})

    assert log_result(result, experiment_name=None) is result
