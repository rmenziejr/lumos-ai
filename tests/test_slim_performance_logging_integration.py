from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

import lumosai
import lumosai.mlflow as mlflow_adapter
from lumosai.artifacts import log_result_with_html_artifact
from lumosai.mlflow import log_result
from lumosai.model import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_PROBABILITY_METRICS,
    PERFORMANCE_METRICS,
    REGRESSION_METRICS,
    ClassificationMetric,
    MetricPreset,
    PerformanceMetric,
    RegressionMetric,
)
from lumosai.results import LumosResult
from lumosai.settings import Settings


class FakeMlflow:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.metric_step: int | None = None
        self.dicts: list[tuple[dict[str, Any], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(metrics)
        self.metric_step = step

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dicts.append((payload, artifact_file))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


def _patch_mlflow_run(monkeypatch: pytest.MonkeyPatch, fake: FakeMlflow) -> None:
    @contextmanager
    def fake_run(*args: Any, **kwargs: Any) -> Iterator[tuple[FakeMlflow, str]]:
        yield fake, "active-run"

    monkeypatch.setattr(mlflow_adapter, "mlflow_run", fake_run)


def test_log_result_passes_step_and_can_suppress_result_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeMlflow()
    _patch_mlflow_run(monkeypatch, fake)
    result = LumosResult(metrics={"performance/f1": 0.8})

    logged = log_result(
        result,
        experiment_name="folds",
        mlflow_step=3,
        log_dict=False,
    )

    assert logged.metadata["mlflow_step"] == 3
    assert fake.metric_step == 3
    assert fake.metrics == {"performance/f1": 0.8}
    assert fake.dicts == []


def test_html_result_logging_passes_step_and_can_suppress_result_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = FakeMlflow()
    _patch_mlflow_run(monkeypatch, fake)
    html_path = tmp_path / "report.html"
    html_path.write_text("<html>report</html>", encoding="utf-8")
    result = LumosResult(metrics={"performance/f1": 0.8})

    logged = log_result_with_html_artifact(
        result,
        html_path=html_path,
        artifact_path="performance",
        experiment_name="folds",
        mlflow_step=4,
        log_dict=False,
    )

    assert logged.metadata["mlflow_step"] == 4
    assert fake.metric_step == 4
    assert fake.metrics == {"performance/f1": 0.8}
    assert fake.dicts == []
    assert fake.artifacts == [(str(html_path), "performance")]


def test_metric_types_and_constants_are_public() -> None:
    assert lumosai.CLASSIFICATION_METRICS is CLASSIFICATION_METRICS
    assert lumosai.CLASSIFICATION_PROBABILITY_METRICS is CLASSIFICATION_PROBABILITY_METRICS
    assert lumosai.REGRESSION_METRICS is REGRESSION_METRICS
    assert lumosai.PERFORMANCE_METRICS is PERFORMANCE_METRICS
    assert lumosai.ClassificationMetric == ClassificationMetric
    assert lumosai.RegressionMetric == RegressionMetric
    assert lumosai.PerformanceMetric == PerformanceMetric
    assert lumosai.MetricPreset == MetricPreset


def test_settings_include_log_loss_as_default_probability_metric() -> None:
    loaded = Settings()
    assert loaded.model.classification_probability_metrics == ["roc_auc", "pr_auc", "log_loss"]
    assert loaded.model.metric_thresholds["log_loss"].greater_is_better is False
