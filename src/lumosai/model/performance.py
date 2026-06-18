from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

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
from lumosai.model.metrics import TaskType, detect_task_type, get_metrics
from lumosai.model.plots import performance_html
from lumosai.model.scores import ScoreInput, normalize_classification_scores
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult
from lumosai.schema import validate_categorical_columns
from lumosai.settings import settings


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
    include_plots: bool = True,
    include_train_plots: bool = False,
    experiment_name: str | None = None,
) -> LumosResult:
    """Evaluate model predictions and return namespaced performance metrics.

    MLflow logging is enabled when `experiment_name` is provided or
    `settings.mlflow.default_experiment_name` is set.
    """

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
    raw_metrics = get_metrics(
        current_pd[target],
        current_pd[prediction],
        y_score=cast(Sequence[Any], scores.values) if scores is not None else None,
        score_labels=scores.labels if scores is not None else None,
        task_type=resolved_task,
        custom_metrics=custom_metrics,
    )
    summary: dict[str, Any] = {"rows": len(current_pd), "metrics": raw_metrics}
    lift_summary: dict[str, Any] | None = None
    if include_lift:
        if resolved_task != "classification" or scores is None:
            msg = "include_lift=True requires classification prediction_score"
            raise LumosValidationError(msg)
        lift_raw_metrics, lift_summary = lift_metrics(current_pd[target], scores)
        raw_metrics.update(lift_raw_metrics)
        summary["lift"] = lift_summary
    elif resolved_task == "classification" and scores is not None and include_plots:
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
        train_raw_metrics = get_metrics(
            train_pd[target],
            train_pd[prediction],
            y_score=cast(Sequence[Any], train_scores.values) if train_scores is not None else None,
            score_labels=train_scores.labels if train_scores is not None else None,
            task_type=resolved_task,
            custom_metrics=custom_metrics,
        )

    metrics = (
        _comparative_performance_metrics(
            train_metrics=train_raw_metrics,
            holdout_metrics=raw_metrics,
        )
        if train_raw_metrics is not None
        else {f"performance/{name}": value for name, value in raw_metrics.items()}
    )
    metadata: dict[str, Any] = {"report_type": "performance", "task_type": resolved_task}
    if scores is not None:
        metadata.update(scores.metadata())
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
    html_path: Path | None = None
    if include_plots:
        title = report_name or "Model Performance Report"
        keep_local = should_keep_html_artifact(experiment_name=experiment_name)
        with artifact_workspace(keep_local=keep_local) as workspace:
            html_path = local_html_artifact_path(
                workspace,
                "performance_report.html",
                report_name=report_name,
            )
            html_path.write_text(
                performance_html(
                    title=title,
                    frame=current_pd,
                    target=target,
                    prediction=prediction,
                    task_type=resolved_task,
                    metrics=metrics,
                    scores=scores,
                    lift_summary=lift_summary,
                ),
                encoding="utf-8",
            )
            artifacts, _ = html_artifact_metadata(
                html_path,
                artifact_path="performance",
                experiment_name=experiment_name,
            )
            result = LumosResult(
                metrics=metrics,
                summary=summary,
                artifacts=artifacts,
                metadata=metadata,
            )
            return log_result_with_html_artifact(
                result,
                html_path=html_path,
                artifact_path="performance",
                experiment_name=experiment_name,
            )
    result = LumosResult(
        metrics=metrics,
        summary=summary,
        artifacts=artifacts,
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result


def _metric_greater_is_better(metric: str) -> bool:
    threshold = settings.model.metric_thresholds.get(metric)
    if threshold is not None:
        return threshold.greater_is_better
    return metric not in {"log_loss", "mae", "rmse", "mean_absolute_error", "mse"}


def _metric_gap(*, metric: str, train_value: float, holdout_value: float) -> float:
    if _metric_greater_is_better(metric):
        return train_value - holdout_value
    return holdout_value - train_value


def _metric_ratio(*, train_value: float, holdout_value: float) -> float:
    if train_value == 0:
        return float("nan")
    return holdout_value / train_value


def _comparative_performance_metrics(
    *,
    train_metrics: dict[str, float],
    holdout_metrics: dict[str, float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in holdout_metrics.items():
        metrics[f"performance/holdout/{name}"] = value
    for name, value in train_metrics.items():
        metrics[f"performance/train/{name}"] = value
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
    comparison: dict[str, dict[str, float]] = {}
    for name in sorted(set(train_metrics).intersection(holdout_metrics)):
        comparison[name] = {
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
    return comparison
