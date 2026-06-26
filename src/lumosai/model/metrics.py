from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from lumosai.exceptions import LumosValidationError
from lumosai.settings import MetricThreshold, settings

TaskType = Literal["classification", "regression"]
ClassificationMetric: TypeAlias = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "log_loss",
]
RegressionMetric: TypeAlias = Literal["mae", "rmse", "r2"]
PerformanceMetric: TypeAlias = ClassificationMetric | RegressionMetric
MetricPreset: TypeAlias = Literal["default", "all"]

CLASSIFICATION_METRICS: tuple[ClassificationMetric, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
)
CLASSIFICATION_PROBABILITY_METRICS: tuple[ClassificationMetric, ...] = (
    "roc_auc",
    "pr_auc",
    "log_loss",
)
REGRESSION_METRICS: tuple[RegressionMetric, ...] = ("mae", "rmse", "r2")
PERFORMANCE_METRICS: tuple[PerformanceMetric, ...] = (
    *CLASSIFICATION_METRICS,
    *CLASSIFICATION_PROBABILITY_METRICS,
    *REGRESSION_METRICS,
)
_SCORE_REQUIRED_METRICS = frozenset(CLASSIFICATION_PROBABILITY_METRICS)


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


def _settings_default_metrics(task_type: TaskType) -> list[str]:
    if task_type == "classification":
        return [
            *settings.model.classification_metrics,
            *settings.model.classification_probability_metrics,
        ]
    return list(settings.model.regression_metrics)


def _all_metrics(task_type: TaskType) -> list[str]:
    if task_type == "classification":
        return [*CLASSIFICATION_METRICS, *CLASSIFICATION_PROBABILITY_METRICS]
    return list(REGRESSION_METRICS)


def _resolve_metric_names(
    *,
    metrics: MetricPreset | list[PerformanceMetric],
    task_type: TaskType,
    has_scores: bool,
) -> list[str]:
    if metrics == "default":
        requested = _settings_default_metrics(task_type)
    elif metrics == "all":
        requested = _all_metrics(task_type)
    else:
        requested = list(metrics)

    supported = set(PERFORMANCE_METRICS)
    unknown = sorted(set(requested).difference(supported))
    if unknown:
        msg = "Unsupported metrics: " + ", ".join(unknown)
        raise LumosValidationError(msg)

    valid_for_task = set(_all_metrics(task_type))
    mismatched = sorted(set(requested).difference(valid_for_task))
    if mismatched:
        msg = "Metrics are not valid for "
        msg += f"{task_type}: " + ", ".join(mismatched)
        raise LumosValidationError(msg)

    score_required = sorted(set(requested).intersection(_SCORE_REQUIRED_METRICS))
    if score_required and not has_scores:
        msg = "Metrics require prediction scores: " + ", ".join(score_required)
        raise LumosValidationError(msg)

    return requested


def _validate_custom_metrics(
    *,
    requested_metrics: list[str],
    custom_metrics: list[tuple[str, Callable[..., float]]] | None,
) -> None:
    custom_names = [name for name, _metric_func in custom_metrics or []]
    duplicate_custom = sorted(
        name for name in set(custom_names) if custom_names.count(name) > 1
    )
    if duplicate_custom:
        msg = "Duplicate custom metric names: " + ", ".join(duplicate_custom)
        raise LumosValidationError(msg)

    built_in_collisions = sorted(set(custom_names).intersection(PERFORMANCE_METRICS))
    requested_collisions = sorted(set(custom_names).intersection(requested_metrics))
    collisions = sorted(set(built_in_collisions + requested_collisions))
    if collisions:
        msg = "Custom metric names collide with built-in metrics: "
        msg += ", ".join(collisions)
        raise LumosValidationError(msg)


def get_metrics(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series | None = None,
    score_labels: Sequence[Any] | None = None,
    task_type: TaskType | None = None,
    metrics: MetricPreset | list[PerformanceMetric] = "default",
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
) -> dict[str, float]:
    """Compute standard classification or regression metrics."""

    resolved_task = task_type or detect_task_type(y_true, y_pred)
    requested_metrics = _resolve_metric_names(
        metrics=metrics,
        task_type=resolved_task,
        has_scores=y_score is not None,
    )
    _validate_custom_metrics(
        requested_metrics=requested_metrics,
        custom_metrics=custom_metrics,
    )
    computed: dict[str, float] = {}

    if resolved_task == "classification":
        average = "weighted"
        zero_division = 0
        if "accuracy" in requested_metrics:
            computed["accuracy"] = float(accuracy_score(y_true, y_pred))
        if "precision" in requested_metrics:
            computed["precision"] = float(
                precision_score(y_true, y_pred, average=average, zero_division=zero_division)
            )
        if "recall" in requested_metrics:
            computed["recall"] = float(
                recall_score(y_true, y_pred, average=average, zero_division=zero_division)
            )
        if "f1" in requested_metrics:
            computed["f1"] = float(
                f1_score(y_true, y_pred, average=average, zero_division=zero_division)
            )
        if y_score is not None and "roc_auc" in requested_metrics:
            computed["roc_auc"] = _roc_auc(y_true, y_score, score_labels)
        if y_score is not None and "pr_auc" in requested_metrics:
            computed["pr_auc"] = _pr_auc(y_true, y_score, score_labels)
        if y_score is not None and "log_loss" in requested_metrics:
            log_loss_value = _log_loss(y_true, y_score, score_labels)
            if log_loss_value is not None:
                computed["log_loss"] = log_loss_value
    else:
        if "mae" in requested_metrics:
            computed["mae"] = float(mean_absolute_error(y_true, y_pred))
        if "rmse" in requested_metrics:
            computed["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        if "r2" in requested_metrics:
            computed["r2"] = float(r2_score(y_true, y_pred))

    for name, metric_func in custom_metrics or []:
        computed[name] = float(metric_func(y_true, y_pred))

    return computed


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


def _pr_auc(
    y_true: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series,
    score_labels: Sequence[Any] | None = None,
) -> float:
    true_labels = _unique_true_labels(y_true)
    score_labels_list = list(score_labels) if score_labels is not None else None
    labels = score_labels_list or true_labels
    score_array = np.asarray(y_score)
    if len(labels) <= 2:
        if score_array.ndim == 2:
            if score_array.shape[1] != 2:
                msg = "binary y_score must be one-dimensional or have two probability columns"
                raise ValueError(msg)
            score_array = _binary_roc_auc_scores(score_array, score_labels)
        elif score_array.ndim == 1:
            score_array = _binary_roc_auc_1d_scores(score_array, true_labels, score_labels)
        positive_label = _positive_label(true_labels)
        events = (pd.Series(y_true) == positive_label).to_numpy(dtype=int)
        return float(average_precision_score(events, score_array))

    if score_labels is None:
        ordered_labels, labels_are_sortable = _sklearn_label_order(labels)
        if not labels_are_sortable:
            ordered_labels = labels
        ordered_scores = score_array
    else:
        ordered_labels, ordered_scores, _ = _scores_in_sklearn_label_order(score_array, labels)
    events = pd.get_dummies(pd.Series(y_true)).reindex(columns=ordered_labels, fill_value=0)
    return float(average_precision_score(events, ordered_scores, average="weighted"))


def _positive_label(
    labels: Sequence[Any],
) -> Any:
    label_list = list(labels)
    ordered_labels, labels_are_sortable = _sklearn_label_order(label_list)
    return ordered_labels[-1] if labels_are_sortable else label_list[-1]


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
