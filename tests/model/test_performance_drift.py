from __future__ import annotations

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.performance_drift import performance_drift_report
from lumosai.settings import Settings, settings


def test_performance_drift_settings_defaults_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = Settings()
    assert loaded.model.performance_drift_psi_threshold == 0.2

    monkeypatch.setenv("LUMOSAI_MODEL__PERFORMANCE_DRIFT_PSI_THRESHOLD", "0.35")
    loaded = Settings()
    assert loaded.model.performance_drift_psi_threshold == 0.35


def test_prediction_only_score_psi_flags_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.model, "performance_drift_psi_threshold", 0.01)
    baseline = pd.DataFrame({"score": [0.01, 0.02, 0.03, 0.04, 0.05]})
    current = pd.DataFrame({"score": [0.95, 0.96, 0.97, 0.98, 0.99]})

    result = performance_drift_report(
        baseline,
        current,
        prediction_score="score",
        include_plots=False,
    )

    assert result.metadata["mode"] == "prediction_only"
    assert result.metrics["performance_drift/baseline/score_psi"] > 0.01
    assert result.summary["score"]["columns"] == ["score"]
    assert result.flagged == [
        {
            "comparison": "baseline",
            "metric": "score_psi",
            "value": result.metrics["performance_drift/baseline/score_psi"],
            "threshold": 0.01,
        }
    ]


def test_performance_drift_requires_signal() -> None:
    frame = pd.DataFrame({"score": [0.1, 0.2]})

    with pytest.raises(LumosValidationError, match="requires prediction_score or target"):
        performance_drift_report(frame, frame, include_plots=False)


def test_labeled_classification_adds_metric_and_residual_drift() -> None:
    baseline = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1, 1, 0],
            "prediction": [0, 0, 1, 1, 1, 0],
            "score": [0.05, 0.1, 0.8, 0.85, 0.9, 0.2],
        }
    )
    current = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1, 1, 0],
            "prediction": [1, 1, 0, 0, 1, 1],
            "score": [0.75, 0.8, 0.25, 0.3, 0.65, 0.9],
        }
    )

    result = performance_drift_report(
        baseline,
        current,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        include_plots=False,
    )

    assert result.metadata["mode"] == "labeled"
    assert result.metrics["performance_drift/baseline/baseline/accuracy"] == 1.0
    assert result.metrics["performance_drift/baseline/current/accuracy"] < 1.0
    assert "performance_drift/baseline/residual_psi" in result.metrics
    assert result.summary["residual"]["kind"] == "classification_probability"
    assert any(flag["metric"] == "metric_drift" for flag in result.flagged)


def test_labeled_regression_adds_metric_and_residual_drift() -> None:
    baseline = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [2.0, 3.0, 4.0]})

    result = performance_drift_report(
        baseline,
        current,
        target="actual",
        prediction="prediction",
        task_type="regression",
        include_plots=False,
    )

    assert result.metadata["mode"] == "labeled"
    assert result.metrics["performance_drift/baseline/current/rmse"] > 0.0
    assert "performance_drift/baseline/residual_psi" in result.metrics
    assert result.summary["residual"]["kind"] == "regression"
