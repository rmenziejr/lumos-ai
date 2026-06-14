from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lumosai.mlflow as mlflow_adapter
from lumosai.exceptions import LumosOptionalDependencyError
from lumosai.model.performance import performance_report
from lumosai.results import LumosResult


def test_performance_report_returns_namespaced_metrics() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "prediction_score": [0.1, 0.9, 0.8, 0.2],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
    )

    assert isinstance(result, LumosResult)
    assert result.metrics["performance/accuracy"] == 1.0
    assert result.metrics["performance/roc_auc"] == 1.0
    assert result.metadata["report_type"] == "performance"
    assert result.metadata["task_type"] == "classification"


def test_performance_report_handles_multiclass_score_vectors() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 2, 1, 2, 0],
            "prediction": [0, 1, 2, 1, 2, 0],
            "prediction_score": [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.15, 0.8],
                [0.1, 0.85, 0.05],
                [0.05, 0.1, 0.85],
                [0.8, 0.1, 0.1],
            ],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
    )

    assert result.metrics["performance/roc_auc"] == 1.0


def test_performance_report_handles_multiclass_numpy_score_vectors() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 2, 1, 2, 0],
            "prediction": [0, 1, 2, 1, 2, 0],
            "prediction_score": [
                np.array([0.9, 0.05, 0.05]),
                np.array([0.1, 0.8, 0.1]),
                np.array([0.05, 0.15, 0.8]),
                np.array([0.1, 0.85, 0.05]),
                np.array([0.05, 0.1, 0.85]),
                np.array([0.8, 0.1, 0.1]),
            ],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
    )

    assert result.metrics["performance/roc_auc"] == 1.0


def test_performance_report_uses_empty_string_score_column_name() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "": [0.1, 0.9, 0.8, 0.2],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="",
        task_type="classification",
    )

    assert result.metrics["performance/roc_auc"] == 1.0


def test_performance_report_requires_mlflow_when_logging_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.5, 2.5]})
    expected = LumosOptionalDependencyError("missing")
    monkeypatch.setattr(mlflow_adapter, "require_mlflow", lambda: (_ for _ in ()).throw(expected))

    with pytest.raises(LumosOptionalDependencyError):
        performance_report(
            frame,
            target="actual",
            prediction="prediction",
            task_type="regression",
            experiment_name="requested",
        )


def test_performance_report_to_dict_is_json_safe() -> None:
    frame = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.5, 2.5]})

    result = performance_report(
        frame, target="actual", prediction="prediction", task_type="regression"
    )

    assert result.to_dict()["metrics"]["performance/mae"] > 0
