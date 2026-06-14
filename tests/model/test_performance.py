from __future__ import annotations

import pandas as pd

from lumosai.model.performance import performance_report
from lumosai.results import LumosResult


def test_performance_report_returns_namespaced_metrics() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "prediction_score": [0.1, 0.9, 0.8, 0.2],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
    )

    assert isinstance(result, LumosResult)
    assert result.metrics["performance/accuracy"] == 1.0
    assert result.metrics["performance/roc_auc"] == 1.0
    assert result.metadata["report_type"] == "performance"
    assert result.metadata["task_type"] == "classification"


def test_performance_report_to_dict_is_json_safe() -> None:
    frame = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.5, 2.5]})

    result = performance_report(
        frame, target="actual", prediction="prediction", task_type="regression"
    )

    assert result.to_dict()["metrics"]["performance/mae"] > 0
