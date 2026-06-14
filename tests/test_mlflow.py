from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import lumosai.mlflow as mlflow_adapter
from lumosai.mlflow import log_artifact_paths, log_result, resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.settings import Settings


class FakeMlflow:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.dicts: dict[str, dict[str, Any]] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.experiment_name: str | None = None

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def active_run(self) -> Any | None:
        return SimpleNamespace(info=SimpleNamespace(run_id="active-run"))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dicts[artifact_file] = payload

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path or ""))


def test_resolve_experiment_name_prefers_argument() -> None:
    loaded = Settings()

    assert resolve_experiment_name("explicit", loaded) == "explicit"


def test_resolve_experiment_name_uses_default_setting() -> None:
    loaded = Settings()
    loaded.mlflow.default_experiment_name = "default"

    assert resolve_experiment_name(None, loaded) == "default"


def test_log_result_without_experiment_returns_original_result() -> None:
    result = LumosResult(metrics={"performance/f1": 1.0})

    assert log_result(result, experiment_name=None) is result


def test_log_result_logs_metadata_before_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mlflow_adapter, "require_mlflow", lambda: fake_mlflow)
    result = LumosResult(metrics={"performance/f1": 1.0})

    logged = log_result(result, experiment_name="experiment")

    assert logged.metadata["logged_to_mlflow"] is True
    assert logged.metadata["mlflow_run_id"] == "active-run"
    assert fake_mlflow.dicts["lumosai_result.json"]["metadata"] == {
        "logged_to_mlflow": True,
        "mlflow_run_id": "active-run",
    }


def test_log_artifact_paths_honors_disabled_artifact_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mlflow_adapter, "require_mlflow", lambda: fake_mlflow)
    loaded = Settings()
    loaded.mlflow.log_artifacts = False
    path = tmp_path / "report.html"
    path.write_text("ok")

    result = log_artifact_paths(
        {"report": path},
        experiment_name="experiment",
        loaded_settings=loaded,
    )

    assert result == {"report": str(path)}
    assert fake_mlflow.artifacts == []
