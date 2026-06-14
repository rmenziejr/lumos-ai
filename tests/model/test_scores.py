from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.scores import normalize_classification_scores, safe_label


def test_binary_1d_scores_infer_positive_label_from_sorted_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "score": [0.1, 0.9, 0.4, 0.2],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
    )

    assert scores.labels == [0, 1]
    assert scores.labels_inferred is True
    assert scores.positive_label == 1
    assert scores.source == "column"
    np.testing.assert_allclose(
        scores.values,
        np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]]),
    )
    assert scores.metadata()["score_labels_inferred"] is True
    assert "score_label_warning" not in scores.metadata()


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_binary_1d_scores_reject_values_outside_probability_bounds(
    score: float,
) -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1],
            "prediction": [0, 1],
            "score": [0.2, score],
        }
    )

    with pytest.raises(LumosValidationError, match="finite probabilities in \\[0, 1\\]"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
        )


@pytest.mark.parametrize("score", [np.nan, np.inf])
def test_binary_1d_scores_reject_non_finite_values(score: float) -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1],
            "prediction": [0, 1],
            "score": [0.2, score],
        }
    )

    with pytest.raises(LumosValidationError, match="finite probabilities in \\[0, 1\\]"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
        )


def test_multiclass_array_scores_use_explicit_score_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        score_labels=["bronze", "silver", "gold"],
    )

    assert scores.labels == ["bronze", "silver", "gold"]
    assert scores.labels_inferred is False
    assert scores.positive_label is None
    assert scores.source == "array"
    np.testing.assert_allclose(scores.values[0], np.array([0.2, 0.7, 0.1]))


def test_multiclass_array_scores_infer_sorted_labels_with_warning_metadata() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
    )

    assert scores.labels == ["bronze", "gold", "silver"]
    assert scores.labels_inferred is True
    assert scores.metadata()["score_label_warning"] == (
        "Multiclass score_labels were inferred by sorted labels; "
        "pass score_labels to match model.classes_."
    )


def test_multiclass_array_scores_require_matching_width() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["a", "b", "c"],
            "prediction": ["a", "b", "c"],
            "score": [[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
        }
    )

    with pytest.raises(LumosValidationError, match="score width"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
            score_labels=["a", "b", "c"],
        )


def test_multiclass_array_scores_require_rows_to_sum_to_one() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.7, 0.3], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        }
    )

    with pytest.raises(LumosValidationError, match="sum to 1"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
            score_labels=["bronze", "silver", "gold"],
        )


def test_unsortable_labels_require_score_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, "two"],
            "prediction": [1, "two"],
            "score": [[0.8, 0.2], [0.1, 0.9]],
        }
    )

    with pytest.raises(LumosValidationError, match="pass score_labels"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
        )


def test_probability_mapping_uses_mapping_keys_as_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.2, 0.8],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
    )

    assert scores.labels == ["bronze", "gold"]
    assert scores.labels_inferred is False
    assert scores.source == "mapping"
    np.testing.assert_allclose(scores.values, np.array([[0.8, 0.2], [0.2, 0.8]]))


def test_probability_mapping_rejects_mismatched_score_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.2, 0.8],
        }
    )

    with pytest.raises(LumosValidationError, match="must match prediction_score mapping"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
            score_labels=["gold", "bronze"],
        )


def test_probability_mapping_scores_require_rows_to_sum_to_one() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.3, 0.8],
        }
    )

    with pytest.raises(LumosValidationError, match="sum to 1"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
        )


@pytest.mark.parametrize("prediction_score", [{}, {"gold": "p_gold"}])
def test_probability_mapping_requires_at_least_two_labels(
    prediction_score: dict[str, str],
) -> None:
    frame = pd.DataFrame(
        {
            "actual": ["gold"],
            "prediction": ["gold"],
            "p_gold": [1.0],
        }
    )

    with pytest.raises(LumosValidationError, match="at least two labels"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score=prediction_score,
        )


def test_score_labels_reject_null_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["no", "yes"],
            "prediction": ["no", "yes"],
            "score": [0.2, 0.8],
        }
    )

    with pytest.raises(LumosValidationError, match="must not contain null labels"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
            score_labels=[None, "yes"],
        )


def test_score_labels_reject_duplicate_equivalent_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1.0],
            "prediction": [1, 1.0],
            "score": [0.2, 0.8],
        }
    )

    with pytest.raises(LumosValidationError, match="must not contain duplicates"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
            score_labels=[1, 1.0],
        )


def test_safe_label_normalizes_metric_path_component() -> None:
    assert safe_label("Gold Plan") == "gold_plan"
    assert safe_label("A/B") == "a_b"
    assert safe_label("") == "empty"
