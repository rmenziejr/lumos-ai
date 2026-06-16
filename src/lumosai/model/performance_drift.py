from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.mlflow import log_result
from lumosai.model.metrics import MetricThreshold, TaskType
from lumosai.model.scores import ScoreInput, safe_label
from lumosai.results import LumosResult
from lumosai.settings import settings

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

    if prediction_score is not None:
        score_columns = _score_columns(baseline_pd, current_pd, prediction_score)
        score_summary: dict[str, Any] = {"columns": list(score_columns)}
        for key, (baseline_score, current_score) in score_columns.items():
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

    result = LumosResult(
        metrics=metrics,
        summary=summary,
        flagged=flagged,
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
