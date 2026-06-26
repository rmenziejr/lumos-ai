from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, log_loss

from lumosai.exceptions import LumosValidationError
from lumosai.model.metrics import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_PROBABILITY_METRICS,
    PERFORMANCE_METRICS,
    REGRESSION_METRICS,
    compare_metric,
    detect_task_type,
    get_metrics,
)
from lumosai.model.validation import validate_prediction_frame
from lumosai.settings import MetricThreshold


def test_metric_constants_list_supported_names() -> None:
    assert CLASSIFICATION_METRICS == ("accuracy", "precision", "recall", "f1")
    assert CLASSIFICATION_PROBABILITY_METRICS == ("roc_auc", "pr_auc", "log_loss")
    assert REGRESSION_METRICS == ("mae", "rmse", "r2")
    assert "f1" in PERFORMANCE_METRICS
    assert "rmse" in PERFORMANCE_METRICS


def test_detect_task_type_classification_for_low_cardinality_labels() -> None:
    assert detect_task_type(pd.Series([0, 1, 1, 0]), pd.Series([0, 1, 0, 0])) == "classification"


def test_detect_task_type_regression_for_float_continuous_values() -> None:
    assert (
        detect_task_type(pd.Series([1.2, 2.5, 3.7, 4.1]), pd.Series([1.0, 2.0, 4.0, 4.5]))
        == "regression"
    )


def test_get_metrics_classification_uses_scores_for_roc_auc() -> None:
    metrics = get_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 0],
        y_score=[0.1, 0.9, 0.8, 0.2],
        task_type="classification",
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_get_metrics_filters_classification_metrics() -> None:
    metrics = get_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        y_score=[0.1, 0.9, 0.4, 0.2],
        task_type="classification",
        metrics=["f1", "roc_auc"],
    )

    assert set(metrics) == {"f1", "roc_auc"}
    assert metrics["f1"] < 1.0
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_get_metrics_all_includes_supported_classification_metrics() -> None:
    metrics = get_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        y_score=[0.1, 0.9, 0.4, 0.2],
        task_type="classification",
        metrics="all",
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
    }


def test_get_metrics_classification_handles_string_labels() -> None:
    metrics = get_metrics(
        y_true=["no", "yes", "yes", "no"],
        y_pred=["no", "yes", "no", "no"],
        task_type="classification",
        metrics=["accuracy", "precision", "recall", "f1"],
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == pytest.approx(0.8333333333)
    assert metrics["recall"] == 0.75
    assert metrics["f1"] == pytest.approx(0.7333333333)


def test_get_metrics_multiclass_classification_uses_probability_matrix_for_roc_auc() -> None:
    metrics = get_metrics(
        y_true=[0, 1, 2, 1, 2, 0],
        y_pred=[0, 1, 2, 1, 2, 0],
        y_score=[
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
            [0.1, 0.85, 0.05],
            [0.05, 0.1, 0.85],
            [0.8, 0.1, 0.1],
        ],
        task_type="classification",
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_get_metrics_multiclass_roc_auc_without_score_labels_uses_sorted_score_order() -> None:
    metrics = get_metrics(
        y_true=[2, 0, 1, 2, 1, 0],
        y_pred=[2, 0, 1, 2, 1, 0],
        y_score=[
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
        ],
        task_type="classification",
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_get_metrics_binary_roc_auc_uses_reversed_score_labels_for_2d_scores() -> None:
    y_true = pd.Series(["no", "yes", "yes", "no"])
    y_score = np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])

    metrics = get_metrics(
        y_true,
        y_pred=pd.Series(["no", "yes", "yes", "no"]),
        y_score=y_score,
        score_labels=["yes", "no"],
        task_type="classification",
        metrics=["roc_auc", "pr_auc", "log_loss"],
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["log_loss"] == pytest.approx(
        log_loss(y_true, y_score[:, [1, 0]], labels=["no", "yes"])
    )


def test_get_metrics_binary_roc_auc_uses_score_labels_for_1d_scores() -> None:
    y_true = pd.Series(["no", "yes", "yes", "no"])

    metrics = get_metrics(
        y_true,
        y_pred=pd.Series(["no", "yes", "yes", "no"]),
        y_score=[0.9, 0.1, 0.2, 0.8],
        score_labels=["yes", "no"],
        task_type="classification",
        metrics=["roc_auc", "pr_auc", "log_loss"],
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["log_loss"] == pytest.approx(
        log_loss(
            y_true,
            np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]]),
            labels=["no", "yes"],
        )
    )


def test_get_metrics_skips_log_loss_for_1d_decision_scores_outside_probability_range() -> None:
    metrics = get_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 0],
        y_score=[-2.0, 3.0, 2.0, -1.0],
        task_type="classification",
        metrics="all",
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert "log_loss" not in metrics


def test_get_metrics_mixed_explicit_score_labels_do_not_raise_raw_type_error() -> None:
    metrics = get_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 0],
        y_score=np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]]),
        score_labels=[0, "one"],
        task_type="classification",
        metrics="all",
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert "log_loss" not in metrics


def test_get_metrics_adds_log_loss_for_binary_probability_matrix() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]])

    metrics = get_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        score_labels=[0, 1],
        task_type="classification",
        metrics=["log_loss"],
    )

    assert metrics["log_loss"] == pytest.approx(log_loss(y_true, y_score, labels=[0, 1]))


def test_get_metrics_adds_log_loss_for_multiclass_probability_matrix() -> None:
    y_true = pd.Series(["bronze", "silver", "gold", "gold"])
    y_pred = pd.Series(["bronze", "silver", "silver", "gold"])
    y_score = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8],
        ]
    )

    metrics = get_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        score_labels=["bronze", "silver", "gold"],
        task_type="classification",
        metrics=["roc_auc", "pr_auc", "log_loss"],
    )

    score_in_sklearn_order = y_score[:, [0, 2, 1]]
    assert metrics["log_loss"] == pytest.approx(
        log_loss(y_true, score_in_sklearn_order, labels=["bronze", "gold", "silver"])
    )
    expected_pr_auc = average_precision_score(
        pd.get_dummies(y_true)[["bronze", "gold", "silver"]],
        score_in_sklearn_order,
        average="weighted",
    )
    assert metrics["pr_auc"] == pytest.approx(expected_pr_auc)
    assert "roc_auc" in metrics


def test_get_metrics_regression() -> None:
    metrics = get_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.5, 2.5],
        task_type="regression",
    )

    assert metrics["mae"] == pytest.approx(0.3333333333)
    assert metrics["rmse"] == pytest.approx(0.4082482904)
    assert metrics["r2"] == pytest.approx(0.75)


def test_get_metrics_filters_regression_metrics() -> None:
    metrics = get_metrics(
        [1.0, 2.0, 3.0],
        [1.1, 1.8, 2.9],
        task_type="regression",
        metrics=["rmse"],
    )

    assert set(metrics) == {"rmse"}
    assert metrics["rmse"] > 0


def test_get_metrics_rejects_unknown_metric() -> None:
    with pytest.raises(LumosValidationError, match="Unsupported metrics: banana"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["banana"])


def test_get_metrics_rejects_task_mismatched_metric() -> None:
    with pytest.raises(LumosValidationError, match="not valid for classification"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["rmse"])


def test_get_metrics_rejects_score_metric_without_scores() -> None:
    with pytest.raises(LumosValidationError, match="require prediction scores"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["roc_auc"])


def test_get_metrics_accepts_empty_builtin_metrics_with_custom_metric() -> None:
    metrics = get_metrics(
        [0, 1],
        [0, 1],
        task_type="classification",
        metrics=[],
        custom_metrics=[("business_value", lambda y_true, y_pred: 42.0)],
    )

    assert metrics == {"business_value": 42.0}


def test_get_metrics_rejects_custom_metric_collisions() -> None:
    with pytest.raises(LumosValidationError, match="Custom metric names collide"):
        get_metrics(
            [0, 1],
            [0, 1],
            task_type="classification",
            metrics=["f1"],
            custom_metrics=[("f1", lambda y_true, y_pred: 1.0)],
        )

    with pytest.raises(LumosValidationError, match="Duplicate custom metric names"):
        get_metrics(
            [0, 1],
            [0, 1],
            task_type="classification",
            metrics=[],
            custom_metrics=[
                ("business_value", lambda y_true, y_pred: 1.0),
                ("business_value", lambda y_true, y_pred: 2.0),
            ],
        )


def test_compare_metric_higher_is_better_relative_flag() -> None:
    threshold = MetricThreshold(mode="relative", value=0.8, greater_is_better=True)

    comparison = compare_metric("f1", group_value=0.70, best_value=1.0, threshold=threshold)

    assert comparison["flagged"] is True
    assert comparison["ratio"] == pytest.approx(0.7)


def test_compare_metric_lower_is_better_relative_flag() -> None:
    threshold = MetricThreshold(mode="relative", value=1.25, greater_is_better=False)

    comparison = compare_metric("rmse", group_value=1.4, best_value=1.0, threshold=threshold)

    assert comparison["flagged"] is True
    assert comparison["ratio"] == pytest.approx(1.4)


def test_compare_metric_relative_uses_absolute_diff_when_baseline_non_positive() -> None:
    threshold = MetricThreshold(mode="relative", value=0.8, greater_is_better=True)

    comparison = compare_metric("r2", group_value=-0.4, best_value=-0.2, threshold=threshold)

    assert comparison["flagged"] is True
    assert pd.isna(comparison["ratio"])
    assert comparison["comparison_mode"] == "absolute_fallback"


def test_validate_prediction_frame_accepts_required_columns() -> None:
    frame = pd.DataFrame({"actual": [1], "prediction": [1], "score": [0.9]})

    validate_prediction_frame(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
    )
