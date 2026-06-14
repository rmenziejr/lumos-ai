from __future__ import annotations

import pandas as pd
import pytest

from lumosai.model.metrics import compare_metric, detect_task_type, get_metrics
from lumosai.settings import MetricThreshold


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


def test_get_metrics_regression() -> None:
    metrics = get_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.5, 2.5],
        task_type="regression",
    )

    assert metrics["mae"] == pytest.approx(0.3333333333)
    assert metrics["rmse"] == pytest.approx(0.4082482904)
    assert metrics["r2"] == pytest.approx(0.75)


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
