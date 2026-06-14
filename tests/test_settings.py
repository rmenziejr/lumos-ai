from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from lumosai.settings import MetricThreshold, Settings


def test_settings_parse_nested_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUMOSAI_MLFLOW__TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("LUMOSAI_MLFLOW__RUN_MODE", "require_active")
    monkeypatch.setenv("LUMOSAI_ARTIFACTS__KEEP_LOCAL", "true")
    monkeypatch.setenv("LUMOSAI_ARTIFACTS__LOCAL_DIR", "./artifacts")
    monkeypatch.setenv("LUMOSAI_MODEL__SHAP_SAMPLE_SIZE", "250")

    loaded = Settings()

    assert loaded.mlflow.tracking_uri == "http://localhost:5000"
    assert loaded.mlflow.run_mode == "require_active"
    assert loaded.artifacts.keep_local is True
    assert loaded.artifacts.local_dir == Path("artifacts")
    assert loaded.model.shap_sample_size == 250


def test_metric_threshold_environment_override_preserves_defaults(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUMOSAI_MODEL__METRIC_THRESHOLDS__RMSE__VALUE", "2.0")

    loaded = Settings()

    assert loaded.model.metric_thresholds["rmse"] == MetricThreshold(
        mode="relative",
        value=2.0,
        greater_is_better=False,
    )
    assert loaded.model.metric_thresholds["f1"] == MetricThreshold(
        mode="relative",
        value=0.8,
        greater_is_better=True,
    )


def test_metric_threshold_defaults_include_metric_direction() -> None:
    loaded = Settings()

    assert loaded.model.metric_thresholds["f1"] == MetricThreshold(
        mode="relative",
        value=0.8,
        greater_is_better=True,
    )
    assert loaded.model.metric_thresholds["rmse"] == MetricThreshold(
        mode="relative",
        value=1.25,
        greater_is_better=False,
    )


def test_data_drift_share_threshold_must_be_share(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUMOSAI_DATA__DRIFT_SHARE_THRESHOLD", "1.5")

    with pytest.raises(ValidationError):
        Settings()
