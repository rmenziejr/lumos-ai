from __future__ import annotations

import pytest

from lumosai.model.metrics import get_metrics


def test_binary_metrics_use_default_positive_label_one() -> None:
    metrics = get_metrics(
        y_true=[0, 0, 0, 1],
        y_pred=[0, 0, 1, 0],
        task_type="classification",
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_binary_metrics_support_explicit_string_positive_label() -> None:
    metrics = get_metrics(
        y_true=["no", "yes", "yes", "no"],
        y_pred=["no", "yes", "no", "no"],
        task_type="classification",
        positive_label="yes",
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_binary_metrics_reject_missing_positive_label() -> None:
    with pytest.raises(ValueError, match="positive_label"):
        get_metrics(
            y_true=["no", "yes"],
            y_pred=["no", "yes"],
            task_type="classification",
        )


def test_multiclass_metrics_report_macro_and_weighted_averages() -> None:
    metrics = get_metrics(
        y_true=[0, 0, 0, 1, 1, 2],
        y_pred=[0, 0, 1, 1, 2, 2],
        task_type="classification",
    )

    assert "precision" not in metrics
    assert "recall" not in metrics
    assert "f1" not in metrics
    assert metrics["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
    assert metrics["macro_recall"] == pytest.approx(((2 / 3) + 0.5 + 1.0) / 3)
    assert metrics["weighted_precision"] == pytest.approx((3 * 1.0 + 2 * 0.5 + 1 * 0.5) / 6)
    assert metrics["weighted_recall"] == pytest.approx((3 * (2 / 3) + 2 * 0.5 + 1 * 1.0) / 6)
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics


def test_binary_probability_metrics_follow_explicit_positive_label() -> None:
    metrics = get_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 0],
        y_score=[
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        score_labels=[0, 1],
        task_type="classification",
        positive_label=0,
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
