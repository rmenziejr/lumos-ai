from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.mlflow import log_result
from lumosai.model.metrics import TaskType, compare_metric, detect_task_type, get_metrics
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult
from lumosai.schema import validate_categorical_columns
from lumosai.settings import MetricThreshold, settings

ProtectedAttribute = list[str] | dict[str, list[float] | None]


def _normalize_protected_attribute(
    protected_attribute: ProtectedAttribute,
) -> dict[str, list[float] | None]:
    if isinstance(protected_attribute, list):
        return {column: None for column in protected_attribute}
    return protected_attribute


def _group_series(df: pd.DataFrame, column: str, bins: list[float] | None) -> pd.Series:
    if bins is None:
        return df[column].astype("object").where(df[column].notna(), "__missing__")
    cut = pd.cut(df[column], bins=bins, include_lowest=True)
    grouped = cut.astype("object")
    grouped = grouped.where(~df[column].isna(), "__missing__")
    return grouped.where(cut.notna() | df[column].isna(), "__out_of_bin__")


def _best_value(values: list[float], *, greater_is_better: bool) -> float:
    return max(values) if greater_is_better else min(values)


def _fallback_threshold(metric_name: str) -> MetricThreshold | None:
    threshold = settings.model.metric_thresholds.get(metric_name)
    if threshold is not None:
        return threshold
    if metric_name in {"abs_mean_residual", "mean_absolute_residual"}:
        return MetricThreshold(mode="relative", value=1.25, greater_is_better=False)
    return None


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _binary_favorable_label(y_true: pd.Series, y_pred: pd.Series) -> Any | None:
    labels = pd.concat([y_true, y_pred], ignore_index=True).dropna().unique().tolist()
    if len(labels) != 2:
        return None

    if all(isinstance(label, bool | np.bool_) for label in labels):
        return True

    numeric_labels: list[float] = []
    for label in labels:
        try:
            numeric_labels.append(float(label))
        except (TypeError, ValueError):
            break
    else:
        if set(numeric_labels) == {0.0, 1.0}:
            return labels[numeric_labels.index(1.0)]
        return labels[numeric_labels.index(max(numeric_labels))]

    favorable_names = {"yes", "true", "positive", "pos", "approved", "approve"}
    for label in labels:
        if str(label).casefold() in favorable_names:
            return label

    return sorted(labels, key=lambda value: str(value))[-1]


def _group_y_score(
    group_df: pd.DataFrame,
    target: str,
    prediction_score: str | None,
    resolved_task: TaskType,
) -> pd.Series | None:
    if prediction_score is None:
        return None
    if resolved_task == "classification" and group_df[target].nunique(dropna=True) < 2:
        return None
    return group_df[prediction_score]


def _finite_metric_rows(
    by_group: list[dict[str, Any]],
    metric_name: str,
) -> list[tuple[dict[str, Any], float]]:
    return [
        (row, float(row[metric_name]))
        for row in by_group
        if metric_name in row and _is_finite(row[metric_name])
    ]


def _parity_comparisons(
    metric_name: str,
    by_group: list[dict[str, Any]],
    *,
    threshold: MetricThreshold | None,
    protected_attribute: str,
) -> list[dict[str, Any]]:
    finite_rows = _finite_metric_rows(by_group, metric_name)
    if len(finite_rows) < 2:
        return []

    values = [value for _, value in finite_rows]
    max_value = max(values)
    min_value = min(values)
    resolved = threshold or MetricThreshold(mode="relative", value=0.8, greater_is_better=True)
    diff = max_value - min_value
    if resolved.mode == "absolute":
        comparison_mode = "absolute_parity"
        ratio = np.nan
        flagged = diff > resolved.value
    else:
        comparison_mode = "relative_parity"
        ratio = 1.0 if max_value == 0 else min_value / max_value
        flagged = ratio < resolved.value

    return [
        {
            "metric": metric_name,
            "comparison_mode": comparison_mode,
            "group": row["group"],
            "protected_attribute": protected_attribute,
            "group_value": value,
            "max_value": float(max_value),
            "min_value": float(min_value),
            "diff": float(diff),
            "ratio": float(ratio),
            "threshold": float(resolved.value),
            "flagged": bool(flagged),
        }
        for row, value in finite_rows
    ]


def bias_report(
    current: Any,
    target: str,
    prediction: str,
    protected_attribute: ProtectedAttribute,
    prediction_score: str | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    report_name: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    """Evaluate model performance parity across protected attribute groups.

    MLflow logging is enabled when `experiment_name` is provided or
    `settings.mlflow.default_experiment_name` is set.
    """

    current_pd = to_pandas(current)
    validate_prediction_frame(
        current_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
    )
    if feature_columns is not None:
        require_columns(current_pd, feature_columns)
    selected_categorical_columns = validate_categorical_columns(
        current_pd,
        categorical_columns=categorical_columns,
        analysis_columns=feature_columns,
    )
    normalized = _normalize_protected_attribute(protected_attribute)
    require_columns(current_pd, normalized.keys())
    resolved_task = task_type or detect_task_type(current_pd[target], current_pd[prediction])

    summary: dict[str, Any] = {"by_attribute": {}}
    flagged: list[dict[str, Any]] = []

    for attribute, bins in normalized.items():
        groups = _group_series(current_pd, attribute, bins)
        working = current_pd.assign(_lumos_group=groups)
        by_group: list[dict[str, Any]] = []
        favorable_label = (
            _binary_favorable_label(current_pd[target], current_pd[prediction])
            if resolved_task == "classification"
            else None
        )

        for group_name, group_df in working.groupby("_lumos_group", observed=False):
            metric_values = get_metrics(
                group_df[target],
                group_df[prediction],
                y_score=_group_y_score(group_df, target, prediction_score, resolved_task),
                task_type=resolved_task,
                custom_metrics=custom_metrics,
            )
            if resolved_task == "classification":
                if favorable_label is not None:
                    metric_values["positive_prediction_rate"] = float(
                        (group_df[prediction] == favorable_label).mean()
                    )
            else:
                residual = group_df[prediction] - group_df[target]
                metric_values["mean_residual"] = float(residual.mean())
                metric_values["abs_mean_residual"] = float(abs(residual.mean()))
                metric_values["mean_absolute_residual"] = float(residual.abs().mean())

            by_group.append(
                {"group": str(group_name), "count": int(len(group_df)), **metric_values}
            )

        comparisons: list[dict[str, Any]] = []
        metric_names = sorted(
            {key for row in by_group for key in row if key not in {"group", "count"}}
        )
        for metric_name in metric_names:
            if metric_name == "mean_residual":
                continue
            threshold = _fallback_threshold(metric_name)
            if metric_name == "positive_prediction_rate":
                parity_comparisons = _parity_comparisons(
                    metric_name,
                    by_group,
                    threshold=threshold,
                    protected_attribute=attribute,
                )
                comparisons.extend(parity_comparisons)
                flagged.extend(
                    comparison for comparison in parity_comparisons if comparison["flagged"]
                )
                continue

            greater_is_better = threshold.greater_is_better if threshold is not None else True
            finite_rows = _finite_metric_rows(by_group, metric_name)
            if len(finite_rows) < 2:
                continue
            values = [value for _, value in finite_rows]
            best = _best_value(values, greater_is_better=greater_is_better)
            for row, value in finite_rows:
                comparison = compare_metric(
                    metric_name,
                    group_value=value,
                    best_value=best,
                    threshold=threshold,
                )
                comparison["group"] = row["group"]
                comparison["protected_attribute"] = attribute
                comparisons.append(comparison)
                if comparison["flagged"]:
                    flagged.append(comparison)

        summary["by_attribute"][attribute] = {
            "by_group": by_group,
            "comparisons": comparisons,
        }

    metadata: dict[str, Any] = {"report_type": "bias", "task_type": resolved_task}
    if report_name is not None:
        metadata["report_name"] = report_name
    if feature_columns is not None:
        metadata["feature_columns"] = list(feature_columns)
    if selected_categorical_columns:
        metadata["categorical_columns"] = selected_categorical_columns

    result = LumosResult(
        metrics={"bias/flags_count": float(len(flagged))},
        summary=summary,
        flagged=flagged,
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
