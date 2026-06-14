from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError

ScoreInput = str | dict[Any, str]

_INFERRED_LABEL_WARNING = (
    "Multiclass score_labels were inferred by sorted labels; "
    "pass score_labels to match model.classes_."
)
_ROW_SUM_ATOL = 1e-8


@dataclass(slots=True)
class ClassificationScores:
    values: np.ndarray
    labels: list[Any]
    labels_inferred: bool
    positive_label: Any | None
    source: Literal["column", "array", "mapping"]
    warning: str | None = None

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "score_labels": list(self.labels),
            "score_labels_inferred": self.labels_inferred,
        }
        if self.positive_label is not None:
            metadata["positive_label"] = self.positive_label
        if self.warning is not None:
            metadata["score_label_warning"] = self.warning
        return metadata

    def label_index(self, label: Any) -> int:
        for index, candidate in enumerate(self.labels):
            if candidate == label:
                return index
        msg = f"label is not present in score_labels: {label!r}"
        raise LumosValidationError(msg)


def safe_label(label: Any) -> str:
    text = str(label).strip().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or "empty"


def _infer_sorted_labels(frame: pd.DataFrame, *, target: str, prediction: str) -> list[Any]:
    values = pd.concat([frame[target], frame[prediction]], ignore_index=True).dropna().unique()
    try:
        return sorted(values.tolist())
    except TypeError as exc:
        msg = (
            "could not infer sorted score_labels for mixed or unsortable labels; pass score_labels"
        )
        raise LumosValidationError(msg) from exc


def _as_score_matrix(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if len(values) == 0:
        msg = "prediction_score must contain at least one row"
        raise LumosValidationError(msg)
    if series.map(lambda value: isinstance(value, list | tuple | np.ndarray)).all():
        try:
            matrix = np.asarray(series.tolist(), dtype=float)
        except (TypeError, ValueError) as exc:
            msg = "prediction_score arrays must contain numeric probabilities"
            raise LumosValidationError(msg) from exc
        if matrix.ndim != 2:
            msg = "prediction_score array values must form a two-dimensional matrix"
            raise LumosValidationError(msg)
        return matrix
    try:
        return np.asarray(values, dtype=float).reshape(-1, 1)
    except (TypeError, ValueError) as exc:
        msg = "prediction_score column must contain numeric probabilities or arrays"
        raise LumosValidationError(msg) from exc


def _is_null_label(label: Any) -> bool:
    try:
        result = pd.isna(label)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool | np.bool_):
        return bool(result)
    return False


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool | np.bool_):
        return bool(result)
    try:
        return bool(np.asarray(result).all())
    except (TypeError, ValueError):
        return False


def _labels_from_score_labels(score_labels: list[Any] | None) -> list[Any] | None:
    if score_labels is None:
        return None
    labels = list(score_labels)
    if len(labels) < 2:
        msg = "score_labels must contain at least two labels"
        raise LumosValidationError(msg)
    for index, label in enumerate(labels):
        if _is_null_label(label):
            msg = "score_labels must not contain null labels"
            raise LumosValidationError(msg)
        if any(_labels_equal(label, existing) for existing in labels[:index]):
            msg = "score_labels must not contain duplicates"
            raise LumosValidationError(msg)
    return labels


def _validate_probability_values(values: np.ndarray) -> None:
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        msg = "prediction_score values must be finite probabilities in [0, 1]"
        raise LumosValidationError(msg)


def _validate_probability_rows_sum_to_one(values: np.ndarray) -> None:
    if not np.allclose(values.sum(axis=1), 1.0, atol=_ROW_SUM_ATOL):
        msg = "prediction_score probability rows must sum to 1"
        raise LumosValidationError(msg)


def _normalize_mapping_scores(
    frame: pd.DataFrame,
    *,
    prediction_score: dict[Any, str],
    score_labels: list[Any] | None,
) -> ClassificationScores:
    labels = list(prediction_score)
    if len(labels) < 2:
        msg = "prediction_score mapping must contain at least two labels"
        raise LumosValidationError(msg)
    explicit_labels = _labels_from_score_labels(score_labels)
    if explicit_labels is not None and explicit_labels != labels:
        msg = "score_labels must match prediction_score mapping keys and order"
        raise LumosValidationError(msg)
    columns = list(prediction_score.values())
    require_columns(frame, columns)
    try:
        values = frame[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        msg = "prediction_score mapping columns must contain numeric probabilities"
        raise LumosValidationError(msg) from exc
    _validate_probability_values(values)
    _validate_probability_rows_sum_to_one(values)
    positive_label = labels[-1] if len(labels) == 2 else None
    return ClassificationScores(
        values=values,
        labels=labels,
        labels_inferred=False,
        positive_label=positive_label,
        source="mapping",
    )


def normalize_classification_scores(
    frame: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    prediction_score: ScoreInput,
    score_labels: list[Any] | None = None,
) -> ClassificationScores:
    require_columns(frame, [target, prediction])
    if isinstance(prediction_score, dict):
        return _normalize_mapping_scores(
            frame,
            prediction_score=prediction_score,
            score_labels=score_labels,
        )

    require_columns(frame, [prediction_score])
    raw_values = _as_score_matrix(frame[prediction_score])
    explicit_labels = _labels_from_score_labels(score_labels)

    if raw_values.shape[1] == 1:
        _validate_probability_values(raw_values)
        labels = explicit_labels or _infer_sorted_labels(
            frame, target=target, prediction=prediction
        )
        if len(labels) != 2:
            msg = "binary 1D prediction_score requires exactly two resolved score labels"
            raise LumosValidationError(msg)
        positive = labels[-1]
        positive_scores = raw_values[:, 0]
        values = np.column_stack([1.0 - positive_scores, positive_scores])
        return ClassificationScores(
            values=values,
            labels=labels,
            labels_inferred=explicit_labels is None,
            positive_label=positive,
            source="column",
        )

    labels_inferred = explicit_labels is None
    labels = explicit_labels or _infer_sorted_labels(frame, target=target, prediction=prediction)
    if raw_values.shape[1] != len(labels):
        msg = (
            "prediction_score score width must match resolved score_labels: "
            f"width={raw_values.shape[1]}, labels={len(labels)}"
        )
        raise LumosValidationError(msg)
    _validate_probability_values(raw_values)
    _validate_probability_rows_sum_to_one(raw_values)
    return ClassificationScores(
        values=raw_values,
        labels=labels,
        labels_inferred=labels_inferred,
        positive_label=labels[-1] if len(labels) == 2 else None,
        source="array",
        warning=_INFERRED_LABEL_WARNING if labels_inferred and len(labels) > 2 else None,
    )
