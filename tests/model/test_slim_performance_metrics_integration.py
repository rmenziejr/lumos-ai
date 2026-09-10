from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model import PerformancePlot
from lumosai.model.metrics import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_PROBABILITY_METRICS,
    PERFORMANCE_METRICS,
    REGRESSION_METRICS,
    get_metrics,
)
from lumosai.model.performance import performance_report
from lumosai.results import LumosResult
from lumosai.settings import settings


def test_metric_constants_list_supported_metric_families() -> None:
    assert CLASSIFICATION_METRICS == ("accuracy", "precision", "recall", "f1")
    assert CLASSIFICATION_PROBABILITY_METRICS == ("roc_auc", "pr_auc", "log_loss")
    assert REGRESSION_METRICS == ("mae", "rmse", "r2")
    assert "f1" in PERFORMANCE_METRICS
    assert "rmse" in PERFORMANCE_METRICS


def test_get_metrics_filters_binary_metric_families() -> None:
    result = get_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        y_score=[0.1, 0.9, 0.4, 0.2],
        task_type="classification",
        metrics=["f1", "roc_auc"],
        positive_label=1,
    )

    assert set(result) == {"f1", "roc_auc"}


def test_get_metrics_preserves_macro_and_weighted_outputs_for_selected_multiclass_family() -> None:
    result = get_metrics(
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 2, 1],
        task_type="classification",
        metrics=["precision"],
    )

    assert set(result) == {"macro_precision", "weighted_precision"}


def test_get_metrics_default_skips_probability_metrics_without_scores() -> None:
    result = get_metrics([0, 1], [0, 1], task_type="classification")
    assert set(result) == {"accuracy", "precision", "recall", "f1"}


def test_get_metrics_explicit_probability_metric_requires_scores() -> None:
    with pytest.raises(LumosValidationError, match="require prediction scores"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["roc_auc"])


def test_get_metrics_rejects_task_mismatch_and_unknown_metric() -> None:
    with pytest.raises(LumosValidationError, match="not valid for classification"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["rmse"])
    with pytest.raises(LumosValidationError, match="Unsupported metrics: banana"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["banana"])


def test_performance_report_metrics_only_defaults_to_no_plots_but_explicit_plots_win(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "score": [0.1, 0.9, 0.4, 0.2],
        }
    )

    slim = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        task_type="classification",
        metrics=["f1"],
        profile="metrics_only",
    )
    assert slim.artifacts == {}
    assert slim.metadata["profile"] == "metrics_only"

    plotted = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        task_type="classification",
        metrics=["f1"],
        profile="metrics_only",
        plots=[PerformancePlot.CONFUSION_MATRIX],
    )
    assert "html" in plotted.artifacts


def test_performance_report_metrics_only_passes_step_and_suppresses_dict_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"actual": [0, 1], "prediction": [0, 1]})
    captured: dict[str, object] = {}

    def fake_log_result(
        result: LumosResult,
        *,
        experiment_name: str | None = None,
        log_dict: bool | None = None,
        mlflow_step: int | None = None,
    ) -> LumosResult:
        captured["log_dict"] = log_dict
        captured["mlflow_step"] = mlflow_step
        return result

    monkeypatch.setattr("lumosai.model.performance.log_result", fake_log_result)

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        task_type="classification",
        metrics=["f1"],
        profile="metrics_only",
        mlflow_step=3,
        experiment_name="folds",
    )

    assert result.metadata["mlflow_step"] == 3
    assert captured == {"log_dict": False, "mlflow_step": 3}


def test_train_holdout_comparison_respects_selected_metrics() -> None:
    train = pd.DataFrame({"actual": [0, 1, 1, 0], "prediction": [0, 1, 1, 0]})
    holdout = pd.DataFrame({"actual": [0, 1, 1, 0], "prediction": [0, 1, 0, 0]})

    result = performance_report(
        holdout,
        target="actual",
        prediction="prediction",
        train=train,
        task_type="classification",
        metrics=["f1"],
        include_plots=False,
    )

    assert set(result.metrics) == {
        "performance/train/f1",
        "performance/holdout/f1",
        "performance/gap/f1",
        "performance/ratio/f1",
    }
