from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pandas as pd
import pytest

import lumosai.bundles as bundles
from lumosai.bundles import monitoring_report, training_report
from lumosai.exceptions import LumosValidationError
from lumosai.results import LumosResult
from lumosai.settings import Settings, settings


def make_monitoring_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "amount": [10, 12, 14, 16, 18, 20],
            "age": [30, 31, 32, 33, 34, 35],
            "target": [0, 1, 0, 1, 0, 1],
            "prediction": [0, 1, 0, 0, 0, 1],
            "region": ["a", "a", "b", "b", "a", "b"],
        }
    )


def make_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "amount": [10, 12, 14, 16, 18, 20],
            "age": [30, 31, 32, 33, 34, 35],
            "target": [0, 1, 0, 1, 0, 1],
            "prediction": [0, 1, 0, 0, 0, 1],
            "region": ["a", "a", "b", "b", "a", "b"],
        }
    )


def test_monitoring_report_requires_temporal_features_for_drift() -> None:
    with pytest.raises(LumosValidationError, match="temporal_features"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            feature_columns=["amount", "age"],
        )


def test_training_report_runs_default_training_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_child_report(*_args: Any, **kwargs: Any) -> LumosResult:
        report_type = kwargs.pop("_report_type")
        calls.append((report_type, kwargs))
        return LumosResult(
            metrics={f"{report_type}/metric": 1.0},
            metadata={"report_type": report_type},
        )

    monkeypatch.setattr(
        bundles,
        "build_sample",
        lambda *_args, **kwargs: fake_child_report(
            **kwargs,
            _report_type=f"sample_{kwargs['role']}",
        ),
    )
    monkeypatch.setattr(
        bundles,
        "performance_report",
        lambda *_args, **kwargs: fake_child_report(**kwargs, _report_type="performance"),
    )
    monkeypatch.setattr(
        bundles,
        "feature_importance",
        lambda *_args, **kwargs: fake_child_report(**kwargs, _report_type="feature_importance"),
    )

    result = training_report(
        make_training_frame(),
        make_training_frame(),
        target="target",
        prediction="prediction",
        model=object(),
        feature_columns=["amount", "age"],
        sample_size=3,
        report_name="training",
    )

    assert result.run_type == "training"
    assert set(result.results) == {
        "train_sample",
        "holdout_sample",
        "performance",
        "feature_importance",
    }
    assert result.metadata["report_name"] == "training"
    assert result.metadata["skipped_reports"]["profile"] == "include_profile not enabled"
    assert [call[0] for call in calls] == [
        "sample_train_benchmark",
        "sample_holdout",
        "performance",
        "feature_importance",
    ]
    assert calls[0][1]["role"] == "train_benchmark"
    assert calls[1][1]["role"] == "holdout"
    assert calls[3][1]["method"] == "both"
    assert calls[3][1]["include_plots"] is True


def test_training_report_runs_optional_profile_and_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_report_types: list[str] = []

    def fake_result(report_type: str) -> LumosResult:
        called_report_types.append(report_type)
        return LumosResult(
            metrics={f"{report_type}/metric": 1.0},
            metadata={"report_type": report_type},
        )

    monkeypatch.setattr(bundles, "build_sample", lambda *_args, **_kwargs: fake_result("sample"))
    monkeypatch.setattr(bundles, "profile", lambda *_args, **_kwargs: fake_result("profile"))
    monkeypatch.setattr(
        bundles,
        "performance_report",
        lambda *_args, **_kwargs: fake_result("performance"),
    )
    monkeypatch.setattr(bundles, "bias_report", lambda *_args, **_kwargs: fake_result("bias"))

    result = training_report(
        make_training_frame(),
        make_training_frame(),
        target="target",
        prediction="prediction",
        protected_attribute="region",
        feature_columns=["amount", "age"],
        include_profile=True,
        include_feature_importance=False,
    )

    assert set(result.results) == {
        "train_sample",
        "holdout_sample",
        "profile",
        "performance",
        "bias",
    }
    assert called_report_types == ["sample", "sample", "profile", "performance", "bias"]


def test_training_report_requires_prediction_when_performance_enabled() -> None:
    with pytest.raises(LumosValidationError, match="prediction"):
        training_report(
            make_training_frame(),
            make_training_frame(),
            target="target",
            include_performance=True,
            feature_columns=["amount", "age"],
        )


def test_training_report_requires_model_when_feature_importance_enabled() -> None:
    with pytest.raises(LumosValidationError, match="model"):
        training_report(
            make_training_frame(),
            make_training_frame(),
            target="target",
            feature_columns=["amount", "age"],
            include_feature_importance=True,
        )


def test_training_report_honors_feature_importance_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = Settings()
    loaded.bundles.include_feature_importance_in_training = False
    called_report_types: list[str] = []

    def fake_result(report_type: str) -> LumosResult:
        called_report_types.append(report_type)
        return LumosResult(
            metrics={f"{report_type}/metric": 1.0},
            metadata={"report_type": report_type},
        )

    monkeypatch.setattr(bundles, "build_sample", lambda *_args, **_kwargs: fake_result("sample"))

    result = training_report(
        make_training_frame(),
        make_training_frame(),
        target="target",
        prediction="prediction",
        model=object(),
        feature_columns=["amount", "age"],
        loaded_settings=loaded,
    )

    assert "feature_importance" not in result.results
    assert (
        result.metadata["skipped_reports"]["feature_importance"]
        == "feature importance disabled by settings"
    )


def test_monitoring_report_requires_prediction_when_performance_enabled() -> None:
    with pytest.raises(LumosValidationError, match="prediction"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            target="target",
            include_performance=True,
            feature_columns=["amount", "age"],
        )


def test_monitoring_report_requires_protected_attribute_when_bias_enabled() -> None:
    with pytest.raises(LumosValidationError, match="protected_attribute"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            target="target",
            prediction="prediction",
            include_bias=True,
            feature_columns=["amount", "age"],
        )


def test_monitoring_report_rejects_temporal_features_in_feature_columns() -> None:
    with pytest.raises(LumosValidationError, match="temporal_features"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            feature_columns=["event_date", "amount"],
        )


def test_monitoring_report_runs_sample_drift_and_performance() -> None:
    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        target="target",
        prediction="prediction",
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
        sample_size=3,
        report_name="daily-monitoring",
    )

    assert result.run_type == "monitoring"
    assert set(result.results) == {
        "monitoring_window",
        "drift_benchmark",
        "performance",
    }
    assert result.results["monitoring_window"].metadata["sample_role"] == "monitoring_window"
    assert result.results["drift_benchmark"].metadata["comparison"] == "benchmark"
    assert result.results["performance"].metadata["report_type"] == "performance"
    assert result.metadata["report_name"] == "daily-monitoring"


def test_monitoring_report_runs_previous_window_drift_when_provided() -> None:
    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        previous_window=make_monitoring_frame(),
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
    )

    assert "drift_previous_window" in result.results
    assert result.results["drift_previous_window"].metadata["comparison"] == "previous_window"


def test_monitoring_report_honors_previous_window_drift_setting() -> None:
    loaded = Settings()
    loaded.bundles.include_previous_window_drift = False

    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        previous_window=make_monitoring_frame(),
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
        loaded_settings=loaded,
    )

    assert "drift_previous_window" not in result.results
    assert (
        result.metadata["skipped_reports"]["drift_previous_window"]
        == "previous_window drift disabled by settings"
    )


def test_monitoring_report_rejects_disabled_fail_fast_setting() -> None:
    loaded = Settings()
    loaded.bundles.fail_fast = False

    with pytest.raises(LumosValidationError, match="fail_fast"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            feature_columns=["amount", "age"],
            loaded_settings=loaded,
        )


def test_monitoring_report_runs_bias_when_protected_attribute_provided() -> None:
    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        target="target",
        prediction="prediction",
        protected_attribute="region",
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
    )

    assert "bias" in result.results
    assert result.results["bias"].metadata["report_type"] == "bias"


def test_monitoring_report_suppresses_child_result_dict_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = Settings()
    loaded.mlflow.default_experiment_name = "monitoring"
    monkeypatch.setattr(settings.mlflow, "log_dicts", True)
    child_log_dict_settings: list[bool] = []
    final_log_dict_settings: list[bool] = []

    def fake_child_report(*_args: Any, **_kwargs: Any) -> LumosResult:
        child_log_dict_settings.append(settings.mlflow.log_dicts)
        return LumosResult(metrics={"child/metric": 1.0})

    def fake_log_run(run: Any, **_kwargs: Any) -> Any:
        final_log_dict_settings.append(settings.mlflow.log_dicts)
        return run

    monkeypatch.setattr(
        bundles,
        "mlflow_run",
        lambda *_args, **_kwargs: nullcontext((object(), "run-1")),
    )
    monkeypatch.setattr(bundles, "build_sample", fake_child_report)
    monkeypatch.setattr(bundles, "drift_report", fake_child_report)
    monkeypatch.setattr(bundles, "log_run", fake_log_run)

    monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
        loaded_settings=loaded,
    )

    assert child_log_dict_settings == [False, False]
    assert final_log_dict_settings == [True]
    assert settings.mlflow.log_dicts is True
