from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from lumosai.artifacts import (
    artifact_workspace,
    html_artifact_metadata,
    local_html_artifact_path,
    log_result_with_html_artifact,
    should_keep_html_artifact,
)
from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.mlflow import log_result
from lumosai.model.metrics import TaskType, compare_metric, detect_task_type, get_metrics
from lumosai.model.plots import performance_drift_html
from lumosai.model.scores import (
    ClassificationScores,
    ScoreInput,
    normalize_classification_scores,
    safe_label,
)
from lumosai.results import LumosResult
from lumosai.settings import MetricThreshold, settings

_PSI_EPSILON = 1e-6


def _safe_comparison_name(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "baseline"


def _finite_array(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(pd.Series(values).dropna(), dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain numeric values"
        raise LumosValidationError(msg) from exc
    array = array[np.isfinite(array)]
    if len(array) == 0:
        msg = f"{name} must contain at least one finite value"
        raise LumosValidationError(msg)
    return array


def _psi(expected: Any, actual: Any, *, bins: int = 10) -> float:
    expected_array = _finite_array(expected, name="baseline distribution")
    actual_array = _finite_array(actual, name="current distribution")
    quantiles = np.linspace(0.0, 1.0, min(bins, len(expected_array)) + 1)
    edges = np.unique(np.quantile(expected_array, quantiles))
    if len(edges) < 2:
        lower = float(np.min(expected_array))
        upper = float(np.max(expected_array))
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        edges = np.array([lower, upper])
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts, _ = np.histogram(expected_array, bins=edges)
    actual_counts, _ = np.histogram(actual_array, bins=edges)
    expected_share = np.maximum(expected_counts / len(expected_array), _PSI_EPSILON)
    actual_share = np.maximum(actual_counts / len(actual_array), _PSI_EPSILON)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def _score_columns(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    prediction_score: ScoreInput,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    if isinstance(prediction_score, dict):
        require_columns(baseline, list(prediction_score.values()))
        require_columns(current, list(prediction_score.values()))
        return {
            safe_label(label): (baseline[column], current[column])
            for label, column in prediction_score.items()
        }
    require_columns(baseline, [prediction_score])
    require_columns(current, [prediction_score])
    return {"score": (baseline[prediction_score], current[prediction_score])}


def _score_columns_from_normalized(
    baseline_scores: ClassificationScores,
    current_scores: ClassificationScores,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if baseline_scores.labels != current_scores.labels:
        msg = "baseline and current score labels must match"
        raise LumosValidationError(msg)
    if baseline_scores.positive_label is not None:
        index = baseline_scores.label_index(baseline_scores.positive_label)
        return {"score": (baseline_scores.values[:, index], current_scores.values[:, index])}
    return {
        safe_label(label): (
            baseline_scores.values[:, baseline_scores.label_index(label)],
            current_scores.values[:, current_scores.label_index(label)],
        )
        for label in baseline_scores.labels
    }


def _classification_residuals(
    frame: pd.DataFrame,
    *,
    target: str,
    scores: ClassificationScores,
) -> np.ndarray:
    if scores.positive_label is not None:
        index = scores.label_index(scores.positive_label)
        actual = (frame[target] == scores.positive_label).to_numpy(dtype=float)
        return actual - scores.values[:, index]

    residuals: list[float] = []
    for row_index, actual_label in enumerate(frame[target]):
        class_index = scores.label_index(actual_label)
        residuals.append(1.0 - float(scores.values[row_index, class_index]))
    return np.asarray(residuals, dtype=float)


def _regression_residuals(frame: pd.DataFrame, *, target: str, prediction: str) -> np.ndarray:
    return frame[target].to_numpy(dtype=float) - frame[prediction].to_numpy(dtype=float)


def _validate_signals(
    *,
    target: str | None,
    prediction: str | None,
    prediction_score: ScoreInput | None,
    score_labels: list[Any] | None,
) -> None:
    if (target is None) != (prediction is None):
        msg = "target and prediction must be provided together"
        raise LumosValidationError(msg)
    if prediction_score is None and (target is None or prediction is None):
        msg = "performance_drift_report requires prediction_score or target and prediction"
        raise LumosValidationError(msg)
    if score_labels is not None and prediction_score is None:
        msg = "score_labels require prediction_score"
        raise LumosValidationError(msg)


def performance_drift_report(
    baseline: Any,
    current: Any,
    *,
    target: str | None = None,
    prediction: str | None = None,
    prediction_score: ScoreInput | None = None,
    score_labels: list[Any] | None = None,
    task_type: TaskType | None = None,
    comparison: str = "baseline",
    metric_thresholds: dict[str, MetricThreshold] | None = None,
    psi_threshold: float | None = None,
    report_name: str | None = None,
    include_plots: bool = True,
    experiment_name: str | None = None,
) -> LumosResult:
    """Compare baseline and current prediction or performance distributions."""

    _validate_signals(
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
        score_labels=score_labels,
    )
    baseline_pd = to_pandas(baseline)
    current_pd = to_pandas(current)
    safe_comparison = _safe_comparison_name(comparison)
    resolved_psi_threshold = (
        settings.model.performance_drift_psi_threshold
        if psi_threshold is None
        else psi_threshold
    )
    if resolved_psi_threshold < 0:
        msg = "psi_threshold must be non-negative"
        raise LumosValidationError(msg)

    mode = "labeled" if target is not None and prediction is not None else "prediction_only"
    metrics: dict[str, float] = {}
    artifacts: dict[str, Any] = {}
    flagged: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "baseline_rows": len(baseline_pd),
        "current_rows": len(current_pd),
    }
    metadata: dict[str, Any] = {
        "report_type": "performance_drift",
        "comparison": safe_comparison,
        "mode": mode,
    }
    if report_name is not None:
        metadata["report_name"] = report_name
    metric_summary: dict[str, Any] | None = None
    score_plot_distributions: dict[str, tuple[Any, Any]] = {}
    residual_plot_distribution: tuple[Any, Any] | None = None
    residual_scatter: tuple[Any, Any, str] | None = None

    baseline_scores: ClassificationScores | None = None
    current_scores: ClassificationScores | None = None
    resolved_task: TaskType | None = task_type
    if target is not None and prediction is not None:
        require_columns(baseline_pd, [target, prediction])
        require_columns(current_pd, [target, prediction])
        resolved_task = task_type or detect_task_type(baseline_pd[target], baseline_pd[prediction])
        if prediction_score is not None and resolved_task == "classification":
            baseline_scores = normalize_classification_scores(
                baseline_pd,
                target=target,
                prediction=prediction,
                prediction_score=prediction_score,
                score_labels=score_labels,
            )
            current_scores = normalize_classification_scores(
                current_pd,
                target=target,
                prediction=prediction,
                prediction_score=prediction_score,
                score_labels=score_labels,
            )
            metadata.update(current_scores.metadata())
        baseline_raw_metrics = get_metrics(
            baseline_pd[target],
            baseline_pd[prediction],
            y_score=baseline_scores.values if baseline_scores is not None else None,
            score_labels=baseline_scores.labels if baseline_scores is not None else None,
            task_type=resolved_task,
        )
        current_raw_metrics = get_metrics(
            current_pd[target],
            current_pd[prediction],
            y_score=current_scores.values if current_scores is not None else None,
            score_labels=current_scores.labels if current_scores is not None else None,
            task_type=resolved_task,
        )
        metric_summary = {}
        thresholds = {**settings.model.metric_thresholds, **(metric_thresholds or {})}
        for metric_name in sorted(set(baseline_raw_metrics) & set(current_raw_metrics)):
            baseline_value = baseline_raw_metrics[metric_name]
            current_value = current_raw_metrics[metric_name]
            comparison_result = compare_metric(
                metric_name,
                group_value=current_value,
                best_value=baseline_value,
                threshold=thresholds.get(metric_name),
            )
            metrics[f"performance_drift/{safe_comparison}/baseline/{metric_name}"] = (
                baseline_value
            )
            metrics[f"performance_drift/{safe_comparison}/current/{metric_name}"] = (
                current_value
            )
            metrics[f"performance_drift/{safe_comparison}/delta/{metric_name}"] = float(
                comparison_result["diff"]
            )
            metrics[f"performance_drift/{safe_comparison}/ratio/{metric_name}"] = float(
                comparison_result["ratio"]
            )
            metric_summary[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "delta": comparison_result["diff"],
                "ratio": comparison_result["ratio"],
                "comparison_mode": comparison_result["comparison_mode"],
                "threshold": comparison_result["threshold"],
                "flagged": comparison_result["flagged"],
            }
            if comparison_result["flagged"]:
                flagged.append(
                    {
                        "comparison": safe_comparison,
                        "metric": "metric_drift",
                        "performance_metric": metric_name,
                        "baseline": baseline_value,
                        "current": current_value,
                        "delta": comparison_result["diff"],
                        "ratio": comparison_result["ratio"],
                        "threshold": comparison_result["threshold"],
                        "comparison_mode": comparison_result["comparison_mode"],
                    }
                )
        summary["metrics"] = metric_summary

    if prediction_score is not None:
        score_columns = (
            _score_columns_from_normalized(baseline_scores, current_scores)
            if baseline_scores is not None and current_scores is not None
            else _score_columns(baseline_pd, current_pd, prediction_score)
        )
        score_summary: dict[str, Any] = {"columns": list(score_columns)}
        for key, (baseline_score, current_score) in score_columns.items():
            score_plot_distributions[key] = (baseline_score, current_score)
            value = _psi(baseline_score, current_score)
            metric_name = (
                f"performance_drift/{safe_comparison}/score_psi"
                if key == "score"
                else f"performance_drift/{safe_comparison}/score_psi/{key}"
            )
            metrics[metric_name] = value
            score_summary[key] = {"psi": value}
            if value > resolved_psi_threshold:
                flagged.append(
                    {
                        "comparison": safe_comparison,
                        "metric": "score_psi",
                        "value": value,
                        "threshold": resolved_psi_threshold,
                    }
                )
        summary["score"] = score_summary

    if target is not None and prediction is not None and resolved_task is not None:
        if (
            resolved_task == "classification"
            and baseline_scores is not None
            and current_scores is not None
        ):
            baseline_residual = _classification_residuals(
                baseline_pd,
                target=target,
                scores=baseline_scores,
            )
            current_residual = _classification_residuals(
                current_pd,
                target=target,
                scores=current_scores,
            )
            residual_kind = "classification_probability"
        elif resolved_task == "regression":
            baseline_residual = _regression_residuals(
                baseline_pd,
                target=target,
                prediction=prediction,
            )
            current_residual = _regression_residuals(
                current_pd,
                target=target,
                prediction=prediction,
            )
            residual_kind = "regression"
        else:
            baseline_residual = None
            current_residual = None
            residual_kind = None

        if baseline_residual is not None and current_residual is not None:
            residual_psi = _psi(baseline_residual, current_residual)
            metrics[f"performance_drift/{safe_comparison}/residual_psi"] = residual_psi
            summary["residual"] = {"kind": residual_kind, "psi": residual_psi}
            residual_plot_distribution = (baseline_residual, current_residual)
            if residual_kind == "classification_probability" and current_scores is not None:
                if current_scores.positive_label is not None:
                    index = current_scores.label_index(current_scores.positive_label)
                    x_values = current_scores.values[:, index]
                else:
                    x_values = np.asarray(range(len(current_residual)), dtype=float)
                residual_scatter = (x_values, current_residual, "Predicted Probability")
            elif residual_kind == "regression":
                residual_scatter = (
                    current_pd[prediction].to_numpy(dtype=float),
                    current_residual,
                    "Prediction",
                )
            if residual_psi > resolved_psi_threshold:
                flagged.append(
                    {
                        "comparison": safe_comparison,
                        "metric": "residual_psi",
                        "value": residual_psi,
                        "threshold": resolved_psi_threshold,
                    }
                )

    result = LumosResult(
        metrics=metrics,
        summary=summary,
        flagged=flagged,
        artifacts=artifacts,
        metadata=metadata,
    )
    if include_plots:
        title = report_name or "Performance Drift Report"
        keep_local = should_keep_html_artifact(experiment_name=experiment_name)
        with artifact_workspace(keep_local=keep_local) as workspace:
            html_path = local_html_artifact_path(
                workspace,
                "performance_drift_report.html",
                report_name=report_name,
            )
            html_path.write_text(
                performance_drift_html(
                    title=title,
                    metrics=metrics,
                    metric_summary=metric_summary,
                    score_distributions=score_plot_distributions,
                    residual_distribution=residual_plot_distribution,
                    residual_scatter=residual_scatter,
                ),
                encoding="utf-8",
            )
            artifacts, _ = html_artifact_metadata(
                html_path,
                artifact_path="performance_drift",
                experiment_name=experiment_name,
            )
            result = LumosResult(
                metrics=metrics,
                summary=summary,
                flagged=flagged,
                artifacts=artifacts,
                metadata=metadata,
            )
            return log_result_with_html_artifact(
                result,
                html_path=html_path,
                artifact_path="performance_drift",
                experiment_name=experiment_name,
            )
    log_result(result, experiment_name=experiment_name)
    return result
