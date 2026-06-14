from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.lift import lift_metrics
from lumosai.model.scores import ClassificationScores


def test_binary_lift_deciles_returns_top_decile_metric_and_table() -> None:
    y_true = pd.Series([1] * 5 + [0] * 15)
    positive_probabilities = np.array(
        [
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
        ]
    )
    scores = ClassificationScores(
        values=np.column_stack([1 - positive_probabilities, positive_probabilities]),
        labels=[0, 1],
        labels_inferred=True,
        positive_label=1,
        source="column",
    )

    metrics, summary = lift_metrics(y_true, scores)

    assert metrics["lift/positive/top_decile"] == pytest.approx(4.0)
    assert metrics["lift/positive/decile_1"] == pytest.approx(4.0)
    assert len(summary["classes"]["positive"]) == 10
    assert summary["classes"]["positive"][0]["rows"] == 2


def test_multiclass_lift_runs_one_vs_rest_per_class() -> None:
    y_true = pd.Series(["bronze", "silver", "gold", "gold", "silver", "bronze"])
    scores = ClassificationScores(
        values=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.2, 0.2, 0.6],
                [0.1, 0.7, 0.2],
                [0.6, 0.3, 0.1],
            ]
        ),
        labels=["bronze", "silver", "gold"],
        labels_inferred=False,
        positive_label=None,
        source="array",
    )

    metrics, summary = lift_metrics(y_true, scores)

    assert "lift/bronze/top_decile" in metrics
    assert "lift/silver/top_decile" in metrics
    assert "lift/gold/top_decile" in metrics
    assert set(summary["classes"]) == {"bronze", "silver", "gold"}


def test_lift_skips_class_with_no_positive_examples() -> None:
    y_true = pd.Series(["bronze", "bronze", "silver", "silver"])
    scores = ClassificationScores(
        values=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
            ]
        ),
        labels=["bronze", "silver", "gold"],
        labels_inferred=False,
        positive_label=None,
        source="array",
    )

    metrics, summary = lift_metrics(y_true, scores)

    assert not any(key.startswith("lift/gold/") for key in metrics)
    assert summary["warnings"] == ["Skipping lift for class 'gold' because it has no positives."]


def test_lift_rejects_y_true_and_score_row_count_mismatch() -> None:
    y_true = pd.Series([0, 1, 1])
    scores = ClassificationScores(
        values=np.array([[0.9, 0.1], [0.2, 0.8]]),
        labels=[0, 1],
        labels_inferred=True,
        positive_label=1,
        source="column",
    )

    with pytest.raises(LumosValidationError, match="row count"):
        lift_metrics(y_true, scores)


def test_lift_tiny_sample_marks_empty_deciles_without_metrics() -> None:
    y_true = pd.Series([1, 0, 1])
    positive_probabilities = np.array([0.9, 0.8, 0.1])
    scores = ClassificationScores(
        values=np.column_stack([1 - positive_probabilities, positive_probabilities]),
        labels=[0, 1],
        labels_inferred=True,
        positive_label=1,
        source="column",
    )

    metrics, summary = lift_metrics(y_true, scores)

    table = summary["classes"]["positive"]
    assert len(table) == 10
    assert table[0]["rows"] == 1
    assert metrics["lift/positive/top_decile"] == pytest.approx(1.5)
    assert metrics["lift/positive/decile_1"] == pytest.approx(1.5)

    empty_rows = table[3:]
    assert len(empty_rows) == 7
    for row in empty_rows:
        assert row["rows"] == 0
        assert row["event_count"] == 0
        assert row["event_rate"] is None
        assert row["baseline_event_rate"] == pytest.approx(2 / 3)
        assert row["lift"] is None
        assert f"lift/positive/decile_{row['decile']}" not in metrics


def test_lift_rejects_safe_label_metric_key_collisions() -> None:
    y_true = pd.Series(["A/B", "A B", "other"])
    scores = ClassificationScores(
        values=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        ),
        labels=["A/B", "A B", "other"],
        labels_inferred=False,
        positive_label=None,
        source="array",
    )

    with pytest.raises(LumosValidationError, match="metric key collision"):
        lift_metrics(y_true, scores)
