from __future__ import annotations

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.bias import bias_report
from lumosai.settings import MetricThreshold, settings


def test_bias_report_flags_group_disparity_for_classification() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 1, 1, 1, 1, 1, 1],
            "prediction": [1, 1, 1, 1, 1, 1, 0, 0],
            "segment": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="classification",
    )

    assert result.metadata["report_type"] == "bias"
    assert result.metrics["bias/flags_count"] >= 1
    assert any(flag["protected_attribute"] == "segment" for flag in result.flagged)


def test_bias_report_bins_continuous_protected_attribute() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 0, 0],
            "prediction": [1, 0, 0, 0],
            "age": [22, 35, 62, 70],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute={"age": [0, 40, 120]},
        task_type="classification",
    )

    assert "age" in result.summary["by_attribute"]
    assert len(result.summary["by_attribute"]["age"]["by_group"]) == 2


def test_bias_report_handles_string_positive_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["yes", "yes", "yes", "yes", "no", "no"],
            "prediction": ["yes", "yes", "no", "no", "no", "no"],
            "segment": ["a", "a", "b", "b", "b", "b"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="classification",
    )

    groups = {
        row["group"]: row["positive_prediction_rate"]
        for row in result.summary["by_attribute"]["segment"]["by_group"]
    }
    assert groups == {"a": 1.0, "b": 0.0}
    assert any(flag["metric"] == "positive_prediction_rate" for flag in result.flagged)


def test_bias_report_handles_non_zero_one_numeric_positive_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [2, 2, 2, 2, 1, 1],
            "prediction": [2, 2, 1, 1, 1, 1],
            "segment": ["a", "a", "b", "b", "b", "b"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="classification",
    )

    groups = {
        row["group"]: row["positive_prediction_rate"]
        for row in result.summary["by_attribute"]["segment"]["by_group"]
    }
    assert groups == {"a": 1.0, "b": 0.0}


def test_bias_report_keeps_out_of_bin_and_missing_groups() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 0, 0, 1],
            "prediction": [1, 1, 0, 0, 0],
            "age": [-5, 22, 62, 130, None],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute={"age": [0, 40, 120]},
        task_type="classification",
    )

    groups = {row["group"] for row in result.summary["by_attribute"]["age"]["by_group"]}
    assert "__out_of_bin__" in groups
    assert "__missing__" in groups


def test_bias_report_uses_absolute_parity_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 1, 1],
            "prediction": [1, 1, 1, 0],
            "segment": ["a", "a", "b", "b"],
        }
    )
    monkeypatch.setitem(
        settings.model.metric_thresholds,
        "positive_prediction_rate",
        MetricThreshold(mode="absolute", value=0.25, greater_is_better=True),
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="classification",
    )

    comparisons = result.summary["by_attribute"]["segment"]["comparisons"]
    parity = [
        comparison
        for comparison in comparisons
        if comparison["metric"] == "positive_prediction_rate"
    ]
    assert parity
    assert all(comparison["comparison_mode"] == "absolute_parity" for comparison in parity)
    assert any(comparison["flagged"] for comparison in parity)


def test_bias_report_compares_metrics_missing_from_first_group() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 0, 1, 0, 1],
            "prediction": [1, 1, 0, 1, 1, 0],
            "score": [0.9, 0.8, 0.1, 0.7, 0.4, 0.6],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        protected_attribute=["segment"],
        task_type="classification",
    )

    comparisons = result.summary["by_attribute"]["segment"]["comparisons"]
    assert any(comparison["metric"] == "roc_auc" for comparison in comparisons)


def test_bias_report_treats_regression_error_as_lower_is_better() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 1.0, 2.0],
            "prediction": [1.0, 2.0, 8.0, 9.0],
            "segment": ["a", "a", "b", "b"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="regression",
    )

    flags = [
        flag
        for flag in result.flagged
        if flag["metric"] in {"mae", "rmse", "abs_mean_residual", "mean_absolute_residual"}
    ]
    assert flags
    assert all(flag["group"] == "b" for flag in flags)


def test_bias_report_includes_report_name_and_schema_metadata() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 0, 0],
            "prediction": [1, 0, 0, 0],
            "segment": ["a", "a", "b", "b"],
            "day_of_week": [1, 2, 3, 4],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        feature_columns=["day_of_week"],
        categorical_columns=["day_of_week"],
        report_name="Holdout Bias",
        task_type="classification",
    )

    assert result.metadata["report_name"] == "Holdout Bias"
    assert result.metadata["feature_columns"] == ["day_of_week"]
    assert result.metadata["categorical_columns"] == ["day_of_week"]


def test_bias_report_rejects_categorical_outside_features() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 0],
            "prediction": [1, 0],
            "segment": ["a", "b"],
            "feature": [10, 20],
            "day_of_week": [1, 2],
        }
    )

    with pytest.raises(LumosValidationError, match="categorical_columns"):
        bias_report(
            frame,
            target="actual",
            prediction="prediction",
            protected_attribute=["segment"],
            feature_columns=["feature"],
            categorical_columns=["day_of_week"],
            task_type="classification",
        )
