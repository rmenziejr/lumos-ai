from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

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
from lumosai.model.lift import lift_metrics
from lumosai.model.metrics import (
    MetricPreset,
    PerformanceMetric,
    TaskType,
    detect_task_type,
    get_metrics,
)
from lumosai.model.performance_plots import selective_performance_html
from lumosai.model.scores import ClassificationScores, ScoreInput, normalize_classification_scores
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult
from lumosai.schema import validate_categorical_columns
from lumosai.settings import settings


class PerformancePlot(StrEnum):
    CONFUSION_MATRIX = "confusion_matrix"
    ROC = "roc"
    PRECISION_RECALL = "precision_recall"
    LIFT = "lift"
    CAPTURE = "capture"
    THRESHOLD_PERFORMANCE = "threshold_performance"
    DECISION_CURVE = "decision_curve"
    PREDICTED_VS_ACTUAL = "predicted_vs_actual"
    RESIDUALS_VS_PREDICTION = "residuals_vs_prediction"
    RESIDUAL_DISTRIBUTION = "residual_distribution"
    RESIDUAL_QQ = "residual_qq"


_CLASSIFICATION_PLOTS = {
    PerformancePlot.CONFUSION_MATRIX,
    PerformancePlot.ROC,
    PerformancePlot.PRECISION_RECALL,
    PerformancePlot.LIFT,
    PerformancePlot.CAPTURE,
    PerformancePlot.THRESHOLD_PERFORMANCE,
    PerformancePlot.DECISION_CURVE,
}
_REGRESSION_PLOTS = {
    PerformancePlot.PREDICTED_VS_ACTUAL,
    PerformancePlot.RESIDUALS_VS_PREDICTION,
    PerformancePlot.RESIDUAL_DISTRIBUTION,
    PerformancePlot.RESIDUAL_QQ,
}


def _resolve_plots(
    *,
    plots: list[PerformancePlot] | None,
    include_plots: bool,
    task_type: TaskType,
) -> set[str]:
    applicable = _CLASSIFICATION_PLOTS if task_type == "classification" else _REGRESSION_PLOTS
    if plots is None:
        return {plot.value for plot in applicable} if include_plots else set()
    selected: set[PerformancePlot] = set()
    for plot in plots:
        try:
            selected.add(PerformancePlot(plot))
        except ValueError as exc:
            raise LumosValidationError(f"unknown performance plot: {plot!r}") from exc
    invalid = selected - applicable
    if invalid:
        names = ", ".join(sorted(plot.value for plot in invalid))
        raise LumosValidationError(
            f"performance plots are not applicable to {task_type}: {names}"
        )
    return {plot.value for plot in selected}


def performance_report(
    current: Any,
    target: str,
    prediction: str,
    prediction_score: ScoreInput | None = None,
    score_labels: list[Any] | None = None,
    train: Any | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    include_lift: bool | None = None,
    report_name: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    include_plots: bool | None = None,
    include_train_plots: bool = False,
    experiment_name: str | None = None,
    positive_label: Any = 1,
    plots: list[PerformancePlot] | None = None,
    metrics: MetricPreset | list[PerformanceMetric] = "default",
    profile: Literal["standard", "metrics_only"] = "standard",
    mlflow_step: int | None = None,
    log_dict: bool | None = None,
) -> LumosResult:
    """Evaluate model predictions and return namespaced performance metrics.

    ``plots`` selects individual diagnostics with typed ``PerformancePlot`` values.
    ``metrics`` selects built-in metric families. ``profile="metrics_only"``
    defaults to scalar metrics without plot or JSON artifacts for repeated
    fold/tuning evaluation. Explicit plot and logging arguments override profile
    defaults.
    """
    if profile not in {"standard", "metrics_only"}:
        raise LumosValidationError("profile must be 'standard' or 'metrics_only'")

    requested_metrics_argument = metrics if isinstance(metrics, str) else list(metrics)
    if include_plots is None:
        resolved_include_plots = profile != "metrics_only"
    else:
        resolved_include_plots = include_plots
    resolved_log_dict = False if profile == "metrics_only" and log_dict is None else log_dict

    current_pd = to_pandas(current)
    validate_prediction_frame(
        current_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score if isinstance(prediction_score, str) else None,
    )
    if feature_columns is not None:
        require_columns(current_pd, feature_columns)
    selected_categorical_columns = validate_categorical_columns(
        current_pd,
        categorical_columns=categorical_columns,
        analysis_columns=feature_columns,
    )
    resolved_task = task_type or detect_task_type(current_pd[target], current_pd[prediction])
    selected_plots = _resolve_plots(
        plots=plots,
        include_plots=resolved_include_plots,
        task_type=resolved_task,
    )
    scores = (
        normalize_classification_scores(
            current_pd,
            target=target,
            prediction=prediction,
            prediction_score=prediction_score,
            score_labels=score_labels,
        )
        if resolved_task == "classification" and prediction_score is not None
        else None
    )
    _set_binary_positive_label(scores, positive_label)
    raw_metrics = get_metrics(
        current_pd[target],
        current_pd[prediction],
        y_score=cast(Sequence[Any], scores.values) if scores is not None else None,
        score_labels=scores.labels if scores is not None else None,
        task_type=resolved_task,
        custom_metrics=custom_metrics,
        positive_label=positive_label,
        metrics=metrics,
    )
    summary: dict[str, Any] = {"rows": len(current_pd), "metrics": raw_metrics}
    lift_summary: dict[str, Any] | None = None
    if include_lift:
        if resolved_task != "classification" or scores is None:
            raise LumosValidationError("include_lift=True requires classification prediction_score")
        lift_raw_metrics, lift_summary = lift_metrics(current_pd[target], scores)
        raw_metrics.update(lift_raw_metrics)
        summary["lift"] = lift_summary
    elif resolved_task == "classification" and scores is not None and {
        "lift",
        "capture",
    } & selected_plots:
        _, lift_summary = lift_metrics(current_pd[target], scores)

    train_raw_metrics: dict[str, float] | None = None
    if train is not None:
        train_pd = to_pandas(train)
        validate_prediction_frame(
            train_pd,
            target=target,
            prediction=prediction,
            prediction_score=prediction_score if isinstance(prediction_score, str) else None,
        )
        if feature_columns is not None:
            require_columns(train_pd, feature_columns)
        validate_categorical_columns(
            train_pd,
            categorical_columns=categorical_columns,
            analysis_columns=feature_columns,
        )
        train_scores = (
            normalize_classification_scores(
                train_pd,
                target=target,
                prediction=prediction,
                prediction_score=prediction_score,
                score_labels=score_labels,
            )
            if resolved_task == "classification" and prediction_score is not None
            else None
        )
        _set_binary_positive_label(train_scores, positive_label)
        train_raw_metrics = get_metrics(
            train_pd[target],
            train_pd[prediction],
            y_score=cast(Sequence[Any], train_scores.values) if train_scores is not None else None,
            score_labels=train_scores.labels if train_scores is not None else None,
            task_type=resolved_task,
            custom_metrics=custom_metrics,
            positive_label=positive_label,
            metrics=metrics,
        )

    namespaced_metrics = (
        _comparative_performance_metrics(
            train_metrics=train_raw_metrics,
            holdout_metrics=raw_metrics,
        )
        if train_raw_metrics is not None
        else {f"performance/{name}": value for name, value in raw_metrics.items()}
    )
    metadata: dict[str, Any] = {
        "report_type": "performance",
        "task_type": resolved_task,
        "plots": sorted(selected_plots),
        "profile": profile,
        "metrics_argument": requested_metrics_argument,
    }
    if mlflow_step is not None:
        metadata["mlflow_step"] = mlflow_step
    if scores is not None:
        metadata.update(scores.metadata())
    elif resolved_task == "classification" and _is_binary(
        current_pd[target], current_pd[prediction]
    ):
        metadata["positive_label"] = positive_label
    if train_raw_metrics is not None:
        summary["train_metrics"] = train_raw_metrics
        summary["holdout_metrics"] = raw_metrics
        summary["comparison"] = _comparative_performance_summary(
            train_metrics=train_raw_metrics,
            holdout_metrics=raw_metrics,
        )
        summary["splits"] = ["train", "holdout"]
        metadata["train_metrics_included"] = True
        metadata["include_train_plots"] = include_train_plots
    else:
        metadata["train_metrics_included"] = False
    if report_name is not None:
        metadata["report_name"] = report_name
    if feature_columns is not None:
        metadata["feature_columns"] = list(feature_columns)
    if selected_categorical_columns:
        metadata["categorical_columns"] = selected_categorical_columns

    artifacts: dict[str, Any] = {}
    if selected_plots:
        title = report_name or "Model Performance Report"
        keep_local = should_keep_html_artifact(experiment_name=experiment_name)
        with artifact_workspace(keep_local=keep_local) as workspace:
            html_path: Path = local_html_artifact_path(
                workspace,
                "performance_report.html",
                report_name=report_name,
            )
            html_path.write_text(
                selective_performance_html(
                    title=title,
                    frame=current_pd,
                    target=target,
                    prediction=prediction,
                    task_type=resolved_task,
                    metrics=namespaced_metrics,
                    scores=scores,
                    lift_summary=lift_summary,
                    plots=selected_plots,
                ),
                encoding="utf-8",
            )
            artifacts, _ = html_artifact_metadata(
                html_path,
                artifact_path="performance",
                experiment_name=experiment_name,
            )
            result = LumosResult(
                metrics=namespaced_metrics,
                summary=summary,
                artifacts=artifacts,
                metadata=metadata,
            )
            return log_result_with_html_artifact(
                result,
                html_path=html_path,
                artifact_path="performance",
                experiment_name=experiment_name,
                log_dict=resolved_log_dict,
                mlflow_step=mlflow_step,
            )
    result = LumosResult(
        metrics=namespaced_metrics,
        summary=summary,
        artifacts=artifacts,
        metadata=metadata,
    )
    log_result(
        result,
        experiment_name=experiment_name,
        log_dict=resolved_log_dict,
        mlflow_step=mlflow_step,
    )
    return result


def _set_binary_positive_label(scores: ClassificationScores | None, positive_label: Any) -> None:
    if scores is None or len(scores.labels) != 2:
        return
    if not any(label == positive_label for label in scores.labels):
        raise LumosValidationError(
            f"positive_label={positive_label!r} is not present in binary score_labels "
            f"{scores.labels!r}; pass positive_label explicitly"
        )
    scores.positive_label = positive_label


def _is_binary(y_true: Any, y_pred: Any) -> bool:
    return (
        pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)
        .dropna()
        .nunique()
        == 2
    )


def _metric_greater_is_better(metric: str) -> bool:
    threshold = settings.model.metric_thresholds.get(metric)
    return (
        threshold.greater_is_better
        if threshold is not None
        else metric not in {"log_loss", "mae", "rmse", "mean_absolute_error", "mse"}
    )


def _metric_gap(*, metric: str, train_value: float, holdout_value: float) -> float:
    return (
        train_value - holdout_value
        if _metric_greater_is_better(metric)
        else holdout_value - train_value
    )


def _metric_ratio(*, train_value: float, holdout_value: float) -> float:
    return float("nan") if train_value == 0 else holdout_value / train_value


def _comparative_performance_metrics(
    *,
    train_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
) -> dict[str, float]:
    metrics = {f"performance/holdout/{name}": value for name, value in holdout_metrics.items()}
    metrics.update({f"performance/train/{name}": value for name, value in train_metrics.items()})
    for name in sorted(set(train_metrics).intersection(holdout_metrics)):
        metrics[f"performance/gap/{name}"] = _metric_gap(
            metric=name,
            train_value=train_metrics[name],
            holdout_value=holdout_metrics[name],
        )
        metrics[f"performance/ratio/{name}"] = _metric_ratio(
            train_value=train_metrics[name],
            holdout_value=holdout_metrics[name],
        )
    return metrics


def _comparative_performance_summary(
    *,
    train_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "train": train_metrics[name],
            "holdout": holdout_metrics[name],
            "gap": _metric_gap(
                metric=name,
                train_value=train_metrics[name],
                holdout_value=holdout_metrics[name],
            ),
            "ratio": _metric_ratio(
                train_value=train_metrics[name],
                holdout_value=holdout_metrics[name],
            ),
        }
        for name in sorted(set(train_metrics).intersection(holdout_metrics))
    }
