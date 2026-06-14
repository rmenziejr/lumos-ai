from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import lumosai.mlflow as mlflow_adapter
from lumosai.mlflow import log_artifact_paths, log_result, log_sample, resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.settings import Settings


class FakeMlflow:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.dicts: list[tuple[dict[str, Any], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.experiment_name: str | None = None

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def active_run(self) -> Any | None:
        return SimpleNamespace(info=SimpleNamespace(run_id="active-run"))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dicts.append((payload, artifact_file))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


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
    assert fake_mlflow.dicts == [
        (
            {
                "metrics": {"performance/f1": 1.0},
                "summary": {},
                "flagged": [],
                "artifacts": {},
                "metadata": {
                    "logged_to_mlflow": True,
                    "mlflow_run_id": "active-run",
                },
            },
            "lumosai_result.json",
        )
    ]


def test_log_sample_logs_metadata_without_raw_artifact(monkeypatch, tmp_path) -> None:
    fake = FakeMlflow()
    result = LumosResult(
        summary={"role": "train_benchmark", "sample_rows": 2},
        artifacts={"sample": pd.DataFrame({"x": [1, 2]})},
        metadata={"report_type": "sample"},
    )

    monkeypatch.setattr(
        "lumosai.mlflow.mlflow_run",
        lambda *_args, **_kwargs: nullcontext((fake, "run-1")),
    )

    logged = log_sample(
        result,
        artifact_path=tmp_path / "sample.csv",
        experiment_name="monitoring",
        log_metadata=True,
        log_artifact=False,
    )

    assert fake.dicts == [
        ({"role": "train_benchmark", "sample_rows": 2}, "lumosai_sample_metadata.json")
    ]
    assert fake.artifacts == []
    assert logged.metadata["sample_artifact_path"].endswith("sample.csv")


def test_log_sample_raw_artifact_is_explicit_opt_in(monkeypatch, tmp_path) -> None:
    fake = FakeMlflow()
    result = LumosResult(
        summary={"role": "holdout", "sample_rows": 2},
        artifacts={"sample": pd.DataFrame({"x": [1, 2]})},
        metadata={"report_type": "sample"},
    )

    monkeypatch.setattr(
        "lumosai.mlflow.mlflow_run",
        lambda *_args, **_kwargs: nullcontext((fake, "run-1")),
    )

    log_sample(
        result,
        artifact_path=tmp_path / "sample.csv",
        experiment_name="monitoring",
        log_metadata=True,
        log_artifact=True,
    )

    assert fake.artifacts == [(str(tmp_path / "sample.csv"), "samples")]


def test_log_sample_does_not_log_raw_artifact_without_dataframe(monkeypatch, tmp_path) -> None:
    fake = FakeMlflow()
    result = LumosResult(
        summary={"role": "holdout", "sample_rows": 0},
        artifacts={},
        metadata={"report_type": "sample"},
    )

    monkeypatch.setattr(
        "lumosai.mlflow.mlflow_run",
        lambda *_args, **_kwargs: nullcontext((fake, "run-1")),
    )

    log_sample(
        result,
        artifact_path=tmp_path / "missing.csv",
        experiment_name="monitoring",
        log_metadata=False,
        log_artifact=True,
    )

    assert fake.artifacts == []
    assert not (tmp_path / "missing.csv").exists()


def test_log_sample_uses_sample_artifact_format_for_suffixless_path(tmp_path) -> None:
    loaded = Settings()
    loaded.data.sample_artifact_format = "csv"
    result = LumosResult(
        summary={"role": "holdout", "sample_rows": 2},
        artifacts={"sample": pd.DataFrame({"x": [1, 2]})},
        metadata={"report_type": "sample"},
    )

    logged = log_sample(
        result,
        artifact_path=tmp_path / "sample",
        loaded_settings=loaded,
        log_metadata=False,
        log_artifact=False,
    )

    expected = tmp_path / "sample.csv"
    assert expected.exists()
    assert logged.metadata["sample_artifact_path"] == str(expected)


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
