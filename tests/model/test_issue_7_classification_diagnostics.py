from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lumosai.model.lift import lift_metrics
from lumosai.model.performance import performance_report
from lumosai.model.scores import ClassificationScores


def _binary_scores(probabilities: list[float]) -> ClassificationScores:
    positive = np.asarray(probabilities, dtype=float)
    return ClassificationScores(
        values=np.column_stack([1.0 - positive, positive]),
        labels=[0, 1],
        labels_inferred=True,
        positive_label=1,
        source="column",
    )


def test_lift_deciles_include_cumulative_capture_fields() -> None:
    y_true = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = _binary_scores([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    _, summary = lift_metrics(y_true, scores)

    rows = summary["classes"]["positive"]
    first = rows[0]
    assert first["population_fraction"] == pytest.approx(0.1)
    assert first["cumulative_rows"] == 1
    assert first["cumulative_event_count"] == 1
    assert first["cumulative_capture_rate"] == pytest.approx(0.25)
    assert rows[-1]["cumulative_capture_rate"] == pytest.approx(1.0)


def test_binary_performance_html_includes_new_classification_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lumosai.settings import settings

    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
            "prediction": [0, 1, 1, 0, 1, 0, 0, 0, 1, 1],
            "prediction_score": [0.05, 0.95, 0.85, 0.1, 0.75, 0.2, 0.45, 0.3, 0.6, 0.8],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
        report_name="Classification Diagnostics",
    )

    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "Observed Event Rate and Cumulative Capture" in html
    assert "Threshold Performance" in html
    assert "Decision Curve Analysis" in html
    assert "Treat All" in html
    assert "Treat None" in html


def test_multiclass_performance_html_skips_binary_only_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lumosai.settings import settings

    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 2, 0, 1, 2],
            "prediction": [0, 1, 2, 0, 1, 2],
            "prediction_score": [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        score_labels=[0, 1, 2],
        task_type="classification",
    )

    html = Path(result.artifacts["html"]).read_text(encoding="utf-8")
    assert "Threshold Performance" not in html
    assert "Decision Curve Analysis" not in html
