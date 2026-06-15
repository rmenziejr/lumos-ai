from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from lumosai.settings import MetricThreshold, settings

TaskType = Literal["classification", "regression"]


def detect_task_type(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
) -> TaskType:
    """Infer whether observed and predicted values look categorical or numeric."""

    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    combined = pd.concat([y_true_series, y_pred_series], ignore_index=True).dropna()
    unique_count = combined.nunique()
    is_float = pd.api.types.is_float_dtype(combined)
    if is_float and unique_count > min(20, max(2, len(combined) // 10)):
        return "regression"
    return "classification"


def get_metrics(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series | None = None,
    score_labels: Sequence[Any] | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
) -> dict[str, float]:
    """Compute standard classification or regression metrics."""

    resolved_task = task_type or detect_task_type(y_true, y_pred)
    metrics: dict[str, float] = {}

    if resolved_task == "classification":
        average = "weighted"
        zero_division = 0
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=zero_division)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=zero_division)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=zero_division)
        )
        if y_score is not None:
            metrics["roc_auc"] = _roc_auc(y_true, y_score, score_labels)
            log_loss_value = _log_loss(y_true, y_score, score_labels)
            if log_loss_value is not None:
                metrics["log_loss"] = log_loss_value
    else:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    for name, metric_func in custom_metrics or []:
        metrics[name] = float(metric_func(y_true, y_pred))

    return metrics


def _roc_auc(
    y_true: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series,
    score_labels: Sequence[Any] | None = None,
) -> float:
    labels = list(score_labels) if score_labels is not None else _unique_true_labels(y_true)
    score_array = np.asarray(y_score)
    if len(labels) <= 2:
        if score_array.ndim == 2:
            if score_array.shape[1] != 2:
                msg = "binary y_score must be one-dimensional or have two probability columns"
                raise ValueError(msg)
            score_array = _binary_roc_auc_scores(score_array, score_labels)
        elif score_array.ndim == 1:
            score_array = _binary_roc_auc_1d_scores(score_array, labels, score_labels)
        return float(roc_auc_score(y_true, score_array))
    if score_labels is None:
        return float(
            roc_auc_score(
                y_true,
                score_array,
                multi_class="ovr",
                average="weighted",
            )
        )
    ordered_labels, ordered_scores, _ = _scores_in_sklearn_label_order(score_array, labels)
    return float(
        roc_auc_score(
            y_true,
            ordered_scores,
            labels=ordered_labels,
            multi_class="ovr",
            average="weighted",
        )
    )


def _log_loss(
    y_true: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series,
    score_labels: Sequence[Any] | None,
) -> float | None:
    score_array = _probability_score_array(y_score)
    if score_array is None:
        return None
    labels = list(score_labels) if score_labels is not None else None
    labels_are_sortable = True
    if labels is not None and score_array.ndim == 1 and len(labels) == 2:
        labels, score_array, labels_are_sortable = _binary_1d_scores_in_sklearn_label_order(
            score_array, labels
        )
    if labels is not None and score_array.ndim == 2:
        labels, score_array, labels_are_sortable = _scores_in_sklearn_label_order(
            score_array, labels
        )
    try:
        return float(log_loss(y_true, score_array, labels=labels))
    except (TypeError, ValueError):
        if labels_are_sortable:
            raise
        return None


def _unique_true_labels(y_true: Sequence[Any] | pd.Series) -> list[Any]:
    return pd.Series(y_true).dropna().unique().tolist()


def _binary_roc_auc_scores(
    score_array: np.ndarray,
    score_labels: Sequence[Any] | None,
) -> np.ndarray:
    if score_labels is None:
        return score_array[:, 1]
    labels = list(score_labels)
    if len(labels) != 2:
        return score_array[:, 1]
    ordered_labels, labels_are_sortable = _sklearn_label_order(labels)
    positive_label = ordered_labels[-1] if labels_are_sortable else labels[-1]
    return score_array[:, labels.index(positive_label)]


def _binary_roc_auc_1d_scores(
    score_array: np.ndarray,
    labels: Sequence[Any],
    score_labels: Sequence[Any] | None,
) -> np.ndarray:
    if score_labels is None:
        return score_array
    score_label_list = list(score_labels)
    if len(score_label_list) != 2:
        return score_array
    ordered_labels, labels_are_sortable = _sklearn_label_order(labels)
    sklearn_positive_label = ordered_labels[-1] if labels_are_sortable else score_label_list[-1]
    score_positive_label = score_label_list[-1]
    if sklearn_positive_label == score_positive_label:
        return score_array
    return 1 - score_array


def _probability_score_array(y_score: Sequence[Any] | pd.Series) -> np.ndarray | None:
    try:
        score_array = np.asarray(y_score, dtype=float)
    except (TypeError, ValueError):
        return None

    if not np.all(np.isfinite(score_array)):
        return None
    if not np.all((0 <= score_array) & (score_array <= 1)):
        return None
    if score_array.ndim == 1:
        return score_array
    if score_array.ndim == 2 and np.allclose(score_array.sum(axis=1), 1.0):
        return score_array
    return None


def _scores_in_sklearn_label_order(
    score_array: np.ndarray,
    labels: Sequence[Any],
) -> tuple[list[Any], np.ndarray, bool]:
    original_labels = list(labels)
    ordered_labels, labels_are_sortable = _sklearn_label_order(original_labels)
    if not labels_are_sortable:
        return original_labels, score_array, False
    if original_labels == ordered_labels:
        return ordered_labels, score_array, True
    column_order = [original_labels.index(label) for label in ordered_labels]
    return ordered_labels, score_array[:, column_order], True


def _binary_1d_scores_in_sklearn_label_order(
    score_array: np.ndarray,
    labels: Sequence[Any],
) -> tuple[list[Any], np.ndarray, bool]:
    original_labels = list(labels)
    ordered_labels, labels_are_sortable = _sklearn_label_order(original_labels)
    if not labels_are_sortable:
        return original_labels, score_array, False
    score_positive_label = original_labels[-1]
    probabilities = np.column_stack(
        [
            score_array if label == score_positive_label else 1 - score_array
            for label in ordered_labels
        ]
    )
    return ordered_labels, probabilities, True


def _sklearn_label_order(labels: Sequence[Any]) -> tuple[list[Any], bool]:
    try:
        return sorted(labels), True
    except TypeError:
        return list(labels), False


def compare_metric(
    metric: str,
    *,
    group_value: float,
    best_value: float,
    threshold: MetricThreshold | None = None,
) -> dict[str, float | bool | str]:
    resolved = threshold or settings.model.metric_thresholds.get(metric)
    if resolved is None:
        resolved = MetricThreshold(mode="relative", value=0.8, greater_is_better=True)

    if resolved.mode == "absolute":
        diff = group_value - best_value
        flagged = diff < -resolved.value if resolved.greater_is_better else diff > resolved.value
        ratio = np.nan
        comparison_mode = "absolute"
    else:
        if best_value <= 0:
            ratio = np.nan
            diff = group_value - best_value
            comparison_mode = "absolute_fallback"
            flagged = diff < 0 if resolved.greater_is_better else diff > 0
        else:
            ratio = group_value / best_value
            comparison_mode = "relative"
            flagged = (
                ratio < resolved.value if resolved.greater_is_better else ratio > resolved.value
            )
            diff = group_value - best_value

    return {
        "metric": metric,
        "comparison_mode": comparison_mode,
        "group_value": float(group_value),
        "best_value": float(best_value),
        "diff": float(diff),
        "ratio": float(ratio),
        "threshold": float(resolved.value),
        "flagged": bool(flagged),
    }
