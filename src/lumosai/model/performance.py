from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.mlflow import log_result
from lumosai.model.metrics import TaskType, detect_task_type, get_metrics
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult


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
    experiment_name: str | None = None,
) -> LumosResult:
    current_pd = to_pandas(current)
    validate_prediction_frame(
        current_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
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
    result = LumosResult(
        metrics=metrics,
        summary={"rows": len(current_pd), "metrics": raw_metrics},
        metadata={"report_type": "performance", "task_type": resolved_task},
    )
    log_result(result, experiment_name=experiment_name)
    return result
