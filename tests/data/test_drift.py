from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from lumosai.data.drift import drift_report, safe_comparison_name


def test_safe_comparison_name_normalizes_metric_path_component() -> None:
    assert safe_comparison_name("Previous Window!") == "previous_window"


def test_drift_report_returns_namespaced_metrics_without_evidently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [10.0]})

    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            self.reference_data = reference_data
            self.current_data = current_data

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": True,
                            "number_of_drifted_columns": 1,
                            "share_of_drifted_columns": 1.0,
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        comparison="Previous Window!",
    )

    assert result.metrics["drift/previous_window/n_drifted_columns"] == 1.0
    assert result.metrics["drift/previous_window/share_drifted_columns"] == 1.0
    assert result.metadata["comparison"] == "previous_window"
    assert result.flagged == [
        {
            "comparison": "previous_window",
            "metric": "share_drifted_columns",
            "value": 1.0,
            "threshold": 0.1,
        }
    ]


def test_drift_report_excludes_temporal_features(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [10.0]})
    captured: dict[str, pd.DataFrame] = {}

    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            captured["reference"] = reference_data
            captured["current"] = current_data

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "number_of_drifted_columns": 0,
                            "share_of_drifted_columns": 0.0,
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())

    drift_report(reference, current, temporal_features=["event_date"])

    assert list(captured["reference"].columns) == ["x"]
    assert list(captured["current"].columns) == ["x"]
