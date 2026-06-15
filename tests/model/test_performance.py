from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lumosai.mlflow as mlflow_adapter
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.model.performance import performance_report
from lumosai.results import LumosResult
from lumosai.settings import settings


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


def test_performance_report_creates_default_classification_html_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0, 1, 0],
            "prediction": [0, 1, 1, 0, 0, 0],
            "prediction_score": [0.1, 0.9, 0.8, 0.2, 0.45, 0.3],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
        report_name="Holdout Performance",
    )

    html_path = Path(result.artifacts["html"])
    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert "Holdout Performance" in html
    assert "ROC Curve" in html
    assert "Precision-Recall Curve" in html
    assert "Lift by Decile" in html
    assert "Confusion Matrix" in html


def test_performance_report_creates_default_regression_html_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 3.0, 4.0],
            "prediction": [1.1, 1.9, 3.2, 3.7],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        task_type="regression",
        report_name="Holdout Regression",
    )

    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "Holdout Regression" in html
    assert "Predicted vs Actual" in html
    assert "Residuals vs Prediction" in html
    assert "Residual Distribution" in html


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


def test_performance_report_includes_report_name_and_schema_metadata() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "day_of_week": [1, 2, 3, 4],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        feature_columns=["day_of_week"],
        categorical_columns=["day_of_week"],
        report_name="Holdout Performance",
        task_type="classification",
    )

    assert result.metadata["report_name"] == "Holdout Performance"
    assert result.metadata["feature_columns"] == ["day_of_week"]
    assert result.metadata["categorical_columns"] == ["day_of_week"]


def test_performance_report_rejects_categorical_outside_features() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1],
            "prediction": [0, 1],
            "feature": [1, 2],
            "day_of_week": [1, 2],
        }
    )

    with pytest.raises(LumosValidationError, match="categorical_columns"):
        performance_report(
            frame,
            target="actual",
            prediction="prediction",
            feature_columns=["feature"],
            categorical_columns=["day_of_week"],
            task_type="classification",
        )


def test_performance_report_records_explicit_score_labels_and_log_loss() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold", "gold"],
            "prediction": ["bronze", "silver", "silver", "gold"],
            "score": [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.1, 0.8],
            ],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        score_labels=["bronze", "silver", "gold"],
        task_type="classification",
    )

    assert "performance/log_loss" in result.metrics
    assert result.metadata["score_labels"] == ["bronze", "silver", "gold"]
    assert result.metadata["score_labels_inferred"] is False


def test_performance_report_infers_score_labels_with_warning_metadata() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        task_type="classification",
    )

    assert result.metadata["score_labels"] == ["bronze", "gold", "silver"]
    assert result.metadata["score_labels_inferred"] is True
    assert "score_label_warning" in result.metadata


def test_performance_report_accepts_probability_mapping() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.2, 0.8],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
        task_type="classification",
    )

    assert "performance/log_loss" in result.metrics
    assert result.metadata["score_labels"] == ["bronze", "gold"]


def test_performance_report_adds_lift_when_enabled() -> None:
    df = pd.DataFrame(
        {
            "actual": [1] * 5 + [0] * 15,
            "prediction": [1] * 5 + [0] * 15,
            "score": [
                0.99,
                0.95,
                0.93,
                0.91,
                0.89,
                0.7,
                0.65,
                0.6,
                0.55,
                0.5,
                0.45,
                0.4,
                0.35,
                0.3,
                0.25,
                0.2,
                0.15,
                0.1,
                0.05,
                0.01,
            ],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        include_lift=True,
        task_type="classification",
    )

    assert result.metrics["performance/lift/positive/top_decile"] == pytest.approx(4.0)
    assert "lift" in result.summary


def test_performance_report_requires_scores_for_lift() -> None:
    df = pd.DataFrame({"actual": [0, 1], "prediction": [0, 1]})

    with pytest.raises(LumosValidationError, match="requires classification prediction_score"):
        performance_report(
            df,
            target="actual",
            prediction="prediction",
            include_lift=True,
            task_type="classification",
        )
