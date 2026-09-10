from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.performance import PerformancePlot, performance_report
from lumosai.settings import settings


def test_performance_plot_enum_has_stable_values() -> None:
    assert PerformancePlot.ROC == "roc"
    assert PerformancePlot.RESIDUAL_QQ == "residual_qq"


def test_performance_report_renders_only_selected_binary_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0, 1, 0],
            "prediction": [0, 1, 1, 0, 0, 0],
            "score": [0.1, 0.9, 0.8, 0.2, 0.45, 0.3],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        task_type="classification",
        plots=[PerformancePlot.CAPTURE, PerformancePlot.DECISION_CURVE],
    )

    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "Observed Event Rate and Cumulative Capture" in html
    assert "Decision Curve Analysis" in html
    assert "ROC Curve" not in html
    assert "Precision-Recall Curve" not in html
    assert "Threshold Performance" not in html
    assert "Confusion Matrix" not in html


def test_performance_report_rejects_plot_for_wrong_task() -> None:
    frame = pd.DataFrame({"actual": [1.0, 2.0], "prediction": [1.1, 1.9]})

    with pytest.raises(LumosValidationError, match="not applicable"):
        performance_report(
            frame,
            target="actual",
            prediction="prediction",
            task_type="regression",
            plots=[PerformancePlot.ROC],
        )


def test_regression_residual_qq_can_be_selected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [1.0, 2.0, 3.0, 4.0, 5.0],
            "prediction": [1.1, 1.8, 3.2, 3.9, 5.3],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        task_type="regression",
        plots=[PerformancePlot.RESIDUAL_QQ],
    )

    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "Residual Q-Q Plot" in html
    assert "Predicted vs Actual" not in html
    assert "Residuals vs Prediction" not in html
