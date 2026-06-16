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


def test_model_importance_settings_defaults() -> None:
    loaded = Settings()

    assert loaded.model.feature_importance_method == "both"
    assert loaded.model.include_feature_importance_plots is True


def test_model_importance_settings_env_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUMOSAI_MODEL__FEATURE_IMPORTANCE_METHOD", "permutation")
    monkeypatch.setenv("LUMOSAI_MODEL__INCLUDE_FEATURE_IMPORTANCE_PLOTS", "false")

    loaded = Settings()

    assert loaded.model.feature_importance_method == "permutation"
    assert loaded.model.include_feature_importance_plots is False


def test_data_drift_share_threshold_must_be_share(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUMOSAI_DATA__DRIFT_SHARE_THRESHOLD", "1.5")

    with pytest.raises(ValidationError):
        Settings()


def test_data_sample_settings_defaults() -> None:
    loaded = Settings()

    assert loaded.data.default_sample_size == 10000
    assert loaded.data.sample_artifact_format == "parquet"
    assert loaded.data.log_sample_metadata is True
    assert loaded.data.log_sample_artifacts is False


def test_data_sample_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LUMOSAI_DATA__DEFAULT_SAMPLE_SIZE", "2500")
    monkeypatch.setenv("LUMOSAI_DATA__SAMPLE_ARTIFACT_FORMAT", "csv")
    monkeypatch.setenv("LUMOSAI_DATA__LOG_SAMPLE_ARTIFACTS", "true")

    loaded = Settings()

    assert loaded.data.default_sample_size == 2500
    assert loaded.data.sample_artifact_format == "csv"
    assert loaded.data.log_sample_artifacts is True


def test_important_drift_settings_defaults() -> None:
    loaded = Settings()

    assert loaded.data.important_drift_top_n == 10
    assert loaded.data.alert_on_important_feature_drift is True


def test_important_drift_settings_env_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUMOSAI_DATA__IMPORTANT_DRIFT_TOP_N", "3")
    monkeypatch.setenv("LUMOSAI_DATA__ALERT_ON_IMPORTANT_FEATURE_DRIFT", "false")

    loaded = Settings()

    assert loaded.data.important_drift_top_n == 3
    assert loaded.data.alert_on_important_feature_drift is False


def test_bundle_settings_defaults() -> None:
    loaded = Settings()

    assert loaded.bundles.include_profile_in_training is False
    assert loaded.bundles.include_feature_importance_in_training is True
    assert loaded.bundles.include_previous_window_drift is True
    assert loaded.bundles.fail_fast is True


def test_bundle_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LUMOSAI_BUNDLES__INCLUDE_PREVIOUS_WINDOW_DRIFT", "false")
    monkeypatch.setenv("LUMOSAI_BUNDLES__FAIL_FAST", "false")

    loaded = Settings()

    assert loaded.bundles.include_previous_window_drift is False
    assert loaded.bundles.fail_fast is False
