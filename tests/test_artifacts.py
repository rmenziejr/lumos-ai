from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import lumosai.artifacts as artifacts_adapter
from lumosai.artifacts import (
    artifact_workspace,
    html_artifact_metadata,
    log_result_with_html_artifact,
)
from lumosai.results import LumosResult
from lumosai.settings import Settings


class FakeMlflow:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.metric_step: int | None = None
        self.dicts: list[tuple[dict[str, Any], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.experiment_name: str | None = None

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def active_run(self) -> Any | None:
        return SimpleNamespace(info=SimpleNamespace(run_id="active-run"))

    def start_run(self) -> Any:
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="started-run")))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(metrics)
        self.metric_step = step

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dicts.append((payload, artifact_file))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


def test_artifact_workspace_uses_temp_dir_when_not_keeping_local() -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = False

    with artifact_workspace(loaded_settings=loaded) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")
        assert path.exists()

    assert not path.exists()


def test_artifact_workspace_keeps_configured_local_dir(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = True
    loaded.artifacts.local_dir = tmp_path

    with artifact_workspace(loaded_settings=loaded) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")

    assert path.exists()


def test_artifact_workspace_can_force_local_retention(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = False
    loaded.artifacts.local_dir = tmp_path

    with artifact_workspace(loaded_settings=loaded, keep_local=True) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")

    assert path.exists()


def test_html_artifact_metadata_caches_mlflow_html_for_display(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.mlflow.default_experiment_name = "experiment"
    loaded.mlflow.log_artifacts = True
    loaded.artifacts.cache_mlflow_html = True
    loaded.artifacts.display_cache_dir = tmp_path / "display-cache"
    html_path = tmp_path / "source.html"
    html_path.write_text("<html>report</html>", encoding="utf-8")

    artifacts, keep_local = html_artifact_metadata(
        html_path,
        artifact_path="performance",
        experiment_name=None,
        loaded_settings=loaded,
    )

    assert keep_local is False
    assert artifacts["html"]["mlflow_artifact_path"] == "performance/source.html"
    cached_path = Path(artifacts["html"]["local_path"])
    assert cached_path.exists()
    assert cached_path.read_text(encoding="utf-8") == "<html>report</html>"
    assert cached_path.parent == loaded.artifacts.display_cache_dir / "performance"


def test_log_result_with_html_artifact_passes_mlflow_step_and_honors_log_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(artifacts_adapter, "require_mlflow", lambda: fake_mlflow)
    loaded = Settings()
    html_path = tmp_path / "report.html"
    html_path.write_text("<html>report</html>", encoding="utf-8")
    result = LumosResult(metrics={"performance/f1": 1.0})

    logged = log_result_with_html_artifact(
        result,
        html_path=html_path,
        artifact_path="performance",
        experiment_name="experiment",
        loaded_settings=loaded,
        mlflow_step=4,
        log_dict=False,
    )

    assert logged.metadata["logged_to_mlflow"] is True
    assert logged.metadata["mlflow_run_id"] == "active-run"
    assert logged.metadata["mlflow_step"] == 4
    assert fake_mlflow.metrics == {"performance/f1": 1.0}
    assert fake_mlflow.metric_step == 4
    assert fake_mlflow.dicts == []
