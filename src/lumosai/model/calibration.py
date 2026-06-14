from __future__ import annotations

from typing import Any, Literal

import numpy as np

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.mlflow import log_result
from lumosai.model.scores import (
    ClassificationScores,
    ScoreInput,
    normalize_classification_scores,
    safe_label,
)
from lumosai.results import LumosResult


def _class_key(label: Any, scores: ClassificationScores) -> str:
    return "positive" if scores.positive_label is not None else safe_label(label)


def _validate_class_keys(labels: list[Any], scores: ClassificationScores) -> None:
    seen: dict[str, Any] = {}
    for label in labels:
        class_key = _class_key(label, scores)
        if class_key in seen:
            msg = (
                "safe_label metric key collision for class labels "
                f"{seen[class_key]!r} and {label!r}: both resolve to {class_key!r}"
            )
            raise LumosValidationError(msg)
        seen[class_key] = label


def _calibration_bins(
    events: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    brier = float(np.mean((probabilities - events) ** 2))
    rows: list[dict[str, Any]] = []
    total = len(events)
    ece = 0.0

    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(events[mask].mean())
        else:
            mean_probability = 0.0
            observed_rate = 0.0
        absolute_error = abs(mean_probability - observed_rate)
        ece += (count / total) * absolute_error if total else 0.0
        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "rows": count,
                "mean_predicted_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": float(absolute_error),
            }
        )

    return brier, float(ece), rows


def calibration_report(
    current: Any,
    target: str,
    prediction_score: ScoreInput,
    *,
    score_labels: list[Any] | None = None,
    n_bins: int = 10,
    strategy: Literal["uniform"] = "uniform",
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    if n_bins < 2:
        msg = "n_bins must be at least 2"
        raise LumosValidationError(msg)
    if strategy != "uniform":
        msg = "strategy must be 'uniform'"
        raise LumosValidationError(msg)

    frame = to_pandas(current)
    require_columns(frame, [target])
    proxy_prediction = "__lumosai_calibration_prediction__"
    working = frame.copy()
    working[proxy_prediction] = working[target]
    scores = normalize_classification_scores(
        working,
        target=target,
        prediction=proxy_prediction,
        prediction_score=prediction_score,
        score_labels=score_labels,
    )

    labels_to_score = (
        [scores.positive_label] if scores.positive_label is not None else scores.labels
    )
    _validate_class_keys(labels_to_score, scores)

    metrics: dict[str, float] = {}
    class_summaries: dict[str, list[dict[str, Any]]] = {}
    brier_values: list[float] = []
    ece_values: list[float] = []

    for label in labels_to_score:
        class_index = scores.label_index(label)
        class_key = _class_key(label, scores)
        events = (working[target] == label).to_numpy(dtype=float)
        probabilities = scores.values[:, class_index]
        brier, ece, rows = _calibration_bins(events, probabilities, n_bins=n_bins)
        metrics[f"calibration/{class_key}/brier"] = brier
        metrics[f"calibration/{class_key}/ece"] = ece
        class_summaries[class_key] = rows
        brier_values.append(brier)
        ece_values.append(ece)

    metrics["calibration/macro_brier"] = float(np.mean(brier_values))
    metrics["calibration/macro_ece"] = float(np.mean(ece_values))
    metadata: dict[str, Any] = {
        "report_type": "calibration",
        "strategy": strategy,
        "n_bins": n_bins,
        "score_source": scores.source,
        **scores.metadata(),
    }
    if report_name is not None:
        metadata["report_name"] = report_name

    result = LumosResult(
        metrics=metrics,
        summary={
            "calibration": {
                "strategy": strategy,
                "n_bins": n_bins,
                "classes": class_summaries,
            }
        },
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
