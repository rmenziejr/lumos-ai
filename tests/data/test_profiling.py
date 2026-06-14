from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from lumosai.data.profiling import profile, temporal_sample
from lumosai.exceptions import LumosValidationError
from lumosai.results import LumosResult
from lumosai.settings import settings


def test_temporal_sample_includes_each_period() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-03-01"]),
            "value": [1, 2, 3, 4],
        }
    )

    sampled = temporal_sample(frame, time_column="event_date", freq="M", sample_size=1)

    assert sampled["event_date"].dt.to_period("M").nunique() == 3


def test_temporal_sample_uses_min_per_period() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
            "value": [1, 2, 3, 4],
        }
    )

    sampled = temporal_sample(
        frame,
        time_column="event_date",
        freq="M",
        sample_size=1,
        min_per_period=2,
    )

    assert len(sampled) == 4


def test_temporal_sample_rejects_invalid_sample_size() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01"]),
            "value": [1],
        }
    )

    with pytest.raises(LumosValidationError, match="sample_size"):
        temporal_sample(frame, time_column="event_date", sample_size=0)


def test_temporal_sample_rejects_invalid_min_per_period() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01"]),
            "value": [1],
        }
    )

    with pytest.raises(LumosValidationError, match="min_per_period"):
        temporal_sample(frame, time_column="event_date", min_per_period=0)


def test_temporal_sample_rejects_null_or_invalid_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "event_date": ["2026-01-01", None, "not-a-date"],
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(LumosValidationError, match="null or invalid"):
        temporal_sample(frame, time_column="event_date")


def test_profile_returns_lumos_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    class FakeProfileReport:
        def __init__(self, df: pd.DataFrame, minimal: bool) -> None:
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file: Path) -> None:
            output_file.write_text("<html>profile</html>")

    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)

    result = profile(frame)

    assert isinstance(result, LumosResult)
    assert result.metadata["report_type"] == "profile"
    assert result.summary["rows"] == 3
    assert cast(Any, result.report).minimal is True
    assert "html" in result.artifacts


def test_profile_uses_temporal_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]),
            "value": [1, 2, 3, 4],
        }
    )

    class FakeProfileReport:
        def __init__(self, df: pd.DataFrame, minimal: bool) -> None:
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file: Path) -> None:
            output_file.write_text("<html>profile</html>")

    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)

    result = profile(frame, time_column="event_date", sample_size=1)

    assert result.summary["rows"] == 2
    assert result.summary["sampling"]["mode"] == "temporal"
    assert result.metadata["minimal"] is False


def test_profile_logs_mlflow_artifact_and_result_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})
    started_runs: list[str] = []
    artifact_runs: list[str] = []
    dict_runs: list[str] = []

    class FakeRun:
        def __enter__(self) -> object:
            started_runs.append("run-1")
            fake_mlflow.current_run_id = "run-1"
            return type("Run", (), {"info": type("Info", (), {"run_id": "run-1"})()})()

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            fake_mlflow.current_run_id = None

    class FakeMlflow:
        def __init__(self) -> None:
            self.current_run_id: str | None = None

        def set_experiment(self, experiment_name: str) -> None:
            self.experiment_name = experiment_name

        def active_run(self) -> object | None:
            return None

        def start_run(self) -> FakeRun:
            return FakeRun()

        def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
            artifact_runs.append(self.current_run_id or "")

        def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
            dict_runs.append(self.current_run_id or "")

    class FakeProfileReport:
        def __init__(self, df: pd.DataFrame, minimal: bool) -> None:
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file: Path) -> None:
            output_file.write_text("<html>profile</html>")

    fake_mlflow = FakeMlflow()
    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: fake_mlflow)

    result = profile(frame, experiment_name="experiment")

    assert started_runs == ["run-1"]
    assert artifact_runs == ["run-1"]
    assert dict_runs == ["run-1"]
    assert result.artifacts["html"] == {"mlflow_artifact_path": "profile/profile.html"}


def test_profile_keeps_local_artifact_when_mlflow_artifacts_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    class FakeRun:
        def __enter__(self) -> object:
            return type("Run", (), {"info": type("Info", (), {"run_id": "run-1"})()})()

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

    class FakeMlflow:
        def set_experiment(self, experiment_name: str) -> None:
            self.experiment_name = experiment_name

        def active_run(self) -> object | None:
            return None

        def start_run(self) -> FakeRun:
            return FakeRun()

        def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
            self.payload = payload

    class FakeProfileReport:
        def __init__(self, df: pd.DataFrame, minimal: bool) -> None:
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file: Path) -> None:
            output_file.write_text("<html>profile</html>")

    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: FakeMlflow())
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    monkeypatch.setattr(settings.mlflow, "log_artifacts", False)

    result = profile(frame, experiment_name="experiment")

    artifact_path = Path(cast(str, result.artifacts["html"]))
    assert artifact_path.exists()
    assert artifact_path.parent == tmp_path
