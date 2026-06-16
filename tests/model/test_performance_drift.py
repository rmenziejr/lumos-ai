from __future__ import annotations

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.performance_drift import performance_drift_report
from lumosai.settings import Settings, settings


def test_performance_drift_settings_defaults_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = Settings()
    assert loaded.model.performance_drift_psi_threshold == 0.2

    monkeypatch.setenv("LUMOSAI_MODEL__PERFORMANCE_DRIFT_PSI_THRESHOLD", "0.35")
    loaded = Settings()
    assert loaded.model.performance_drift_psi_threshold == 0.35


def test_prediction_only_score_psi_flags_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.model, "performance_drift_psi_threshold", 0.01)
    baseline = pd.DataFrame({"score": [0.01, 0.02, 0.03, 0.04, 0.05]})
    current = pd.DataFrame({"score": [0.95, 0.96, 0.97, 0.98, 0.99]})

    result = performance_drift_report(
        baseline,
        current,
        prediction_score="score",
        include_plots=False,
    )

    assert result.metadata["mode"] == "prediction_only"
    assert result.metrics["performance_drift/baseline/score_psi"] > 0.01
    assert result.summary["score"]["columns"] == ["score"]
    assert result.flagged == [
        {
            "comparison": "baseline",
            "metric": "score_psi",
            "value": result.metrics["performance_drift/baseline/score_psi"],
            "threshold": 0.01,
        }
    ]


def test_performance_drift_requires_signal() -> None:
    frame = pd.DataFrame({"score": [0.1, 0.2]})

    with pytest.raises(LumosValidationError, match="requires prediction_score or target"):
        performance_drift_report(frame, frame, include_plots=False)
