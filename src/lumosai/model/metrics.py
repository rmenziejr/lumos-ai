from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
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
    y_score: Sequence[float] | pd.Series | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
) -> dict[str, float]:
    resolved_task = task_type or detect_task_type(y_true, y_pred)
    metrics: dict[str, float] = {}

    if resolved_task == "classification":
        average = "binary" if pd.Series(y_true).nunique(dropna=True) <= 2 else "weighted"
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
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    for name, metric_func in custom_metrics or []:
        metrics[name] = float(metric_func(y_true, y_pred))

    return metrics


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
    else:
        if best_value == 0:
            ratio = np.inf if group_value != 0 else 1.0
        else:
            ratio = group_value / best_value
        flagged = ratio < resolved.value if resolved.greater_is_better else ratio > resolved.value
        diff = group_value - best_value

    return {
        "metric": metric,
        "group_value": float(group_value),
        "best_value": float(best_value),
        "diff": float(diff),
        "ratio": float(ratio),
        "threshold": float(resolved.value),
        "flagged": bool(flagged),
    }
