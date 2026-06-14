from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from lumosai.data.drift import drift_report, safe_comparison_name
from lumosai.exceptions import LumosValidationError
from lumosai.settings import settings


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


def test_drift_report_rejects_all_temporal_columns() -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"]})
    current = pd.DataFrame({"event_date": ["2026-01-02"]})

    with pytest.raises(LumosValidationError, match="non-temporal"):
        drift_report(reference, current, temporal_features=["event_date"])


def test_drift_report_rejects_temporal_columns_in_mapping() -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [2.0]})

    with pytest.raises(LumosValidationError, match="column_mapping"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            column_mapping={"datetime": "event_date"},
        )


def test_drift_report_threshold_boundary_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [2.0]})

    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            return None

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "number_of_drifted_columns": 1,
                            "share_of_drifted_columns": 0.1,
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 0.1)

    result = drift_report(reference, current, temporal_features=["event_date"])

    assert result.flagged == []


@pytest.mark.integration
def test_drift_report_uses_installed_evidently_api() -> None:
    reference = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=20),
            "x": list(range(20)),
        }
    )
    current = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-02-01", periods=20),
            "x": list(range(100, 120)),
        }
    )

    result = drift_report(reference, current, temporal_features=["event_date"])

    assert "drift/benchmark/n_drifted_columns" in result.metrics
    assert "drift/benchmark/share_drifted_columns" in result.metrics
