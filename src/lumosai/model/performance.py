from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.mlflow import log_result
from lumosai.model.metrics import TaskType, detect_task_type, get_metrics
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult
from lumosai.schema import validate_categorical_columns


def _score_values(current_pd: pd.DataFrame, prediction_score: str | None) -> Any:
    if prediction_score is None:
        return None
    scores = current_pd[prediction_score]
    if scores.map(lambda value: isinstance(value, list | tuple | np.ndarray)).all():
        return scores.tolist()
    return scores


def performance_report(
    current: Any,
    target: str,
    prediction: str,
    prediction_score: str | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    report_name: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
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
    resolved_task = task_type or detect_task_type(current_pd[target], current_pd[prediction])
    raw_metrics = get_metrics(
        current_pd[target],
        current_pd[prediction],
        y_score=_score_values(current_pd, prediction_score),
        task_type=resolved_task,
        custom_metrics=custom_metrics,
    )
    metrics = {f"performance/{name}": value for name, value in raw_metrics.items()}
    metadata: dict[str, Any] = {"report_type": "performance", "task_type": resolved_task}
    if report_name is not None:
        metadata["report_name"] = report_name
    if feature_columns is not None:
        metadata["feature_columns"] = list(feature_columns)
    if selected_categorical_columns:
        metadata["categorical_columns"] = selected_categorical_columns
    result = LumosResult(
        metrics=metrics,
        summary={"rows": len(current_pd), "metrics": raw_metrics},
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
