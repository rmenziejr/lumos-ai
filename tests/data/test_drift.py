from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lumosai.data.drift import drift_report, safe_comparison_name
from lumosai.exceptions import LumosValidationError
from lumosai.results import LumosResult
from lumosai.settings import settings


def test_safe_comparison_name_normalizes_metric_path_component() -> None:
    assert safe_comparison_name("Previous Window!") == "previous_window"


def test_drift_report_returns_namespaced_metrics_without_evidently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
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
    html_path = Path(result.artifacts["html"])
    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert "Data Drift Report" in html
    assert "previous_window" in html


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


def test_drift_report_filters_feature_columns_and_passes_evidently_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame(
        {
            "event_date": ["2026-01-01"],
            "x": [1.0],
            "day_of_week": [1],
            "ignored": [100],
        }
    )
    current = pd.DataFrame(
        {
            "event_date": ["2026-01-02"],
            "x": [10.0],
            "day_of_week": [2],
            "ignored": [200],
        }
    )
    captured: dict[str, Any] = {}

    class FakeReport:
        def __init__(self, metrics: list[Any], tags: list[str] | None = None) -> None:
            captured["metrics"] = metrics
            captured["tags"] = tags

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            captured["reference_columns"] = list(reference_data.columns)
            captured["current_columns"] = list(current_data.columns)

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

    class FakePreset:
        def __init__(self, drift_share: float, columns: list[str]) -> None:
            captured["drift_share"] = drift_share
            captured["preset_columns"] = columns

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", FakePreset)

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        feature_columns=["x", "day_of_week"],
        categorical_columns=["day_of_week"],
        report_name="Benchmark Drift",
        evidently_kwargs={
            "preset": {"drift_share": 0.4},
            "report": {"tags": ["production"]},
        },
    )

    assert captured["reference_columns"] == ["x", "day_of_week"]
    assert captured["current_columns"] == ["x", "day_of_week"]
    assert captured["drift_share"] == 0.4
    assert captured["preset_columns"] == ["x", "day_of_week"]
    assert captured["tags"] == ["production"]
    assert result.metadata["report_name"] == "Benchmark Drift"
    assert result.metadata["feature_columns"] == ["x", "day_of_week"]
    assert result.metadata["categorical_columns"] == ["day_of_week"]


def test_drift_report_rejects_unsupported_evidently_kwargs() -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [2.0]})

    with pytest.raises(LumosValidationError, match="unsupported"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            evidently_kwargs={"preset": {"not_supported": True}},
        )


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


def test_drift_report_rejects_categorical_outside_features() -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0], "day": [1]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [2.0], "day": [2]})

    with pytest.raises(LumosValidationError, match="categorical_columns"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            feature_columns=["x"],
            categorical_columns=["day"],
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


def _aggregate_only_report(monkeypatch: pytest.MonkeyPatch) -> None:
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
                            "number_of_drifted_columns": 0,
                            "share_of_drifted_columns": 0.0,
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())


def test_drift_report_rejects_important_features_outside_analysis_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5]})

    with pytest.raises(LumosValidationError, match="important_features"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            feature_columns=["x"],
            important_features=["y"],
        )


def test_drift_report_rejects_non_string_important_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5]})

    with pytest.raises(LumosValidationError, match="string feature names"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            important_features=[1],  # type: ignore[list-item]
        )


def test_drift_report_rejects_string_important_features_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5]})

    with pytest.raises(LumosValidationError, match="list of string feature names"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            important_features="x",  # type: ignore[arg-type]
        )


def test_drift_report_rejects_integer_important_features_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5]})

    with pytest.raises(LumosValidationError, match="list of string feature names"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            important_features=1,  # type: ignore[arg-type]
        )


def test_drift_report_records_explicit_important_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame(
        {"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0]}
    )
    current = pd.DataFrame(
        {"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5]}
    )

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["y", "x"],
    )

    assert result.metadata["important_features"] == ["y", "x"]
    assert result.metadata["important_feature_source"] == "explicit"


def test_drift_report_rejects_importance_result_without_permutation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5]})
    importance = LumosResult(metrics={}, summary={"methods": {"shap": {"features": []}}})

    with pytest.raises(LumosValidationError, match="permutation"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            importance_result=importance,
        )


def test_drift_report_extracts_top_n_permutation_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    monkeypatch.setattr(settings.data, "important_drift_top_n", 2)
    reference = pd.DataFrame(
        {"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0], "z": [3.0]}
    )
    current = pd.DataFrame(
        {"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5], "z": [3.5]}
    )
    importance = LumosResult(
        metrics={},
        summary={
            "methods": {
                "permutation": {
                    "features": [
                        {"feature": "y", "importance_mean": 0.9},
                        {"feature": "x", "importance_mean": 0.7},
                        {"feature": "z", "importance_mean": 0.2},
                    ]
                }
            }
        },
    )

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        importance_result=importance,
    )

    assert result.metadata["important_features"] == ["y", "x"]
    assert result.metadata["important_feature_source"] == "importance_result"


def test_drift_report_flags_drifted_important_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame(
        {"event_date": ["2026-01-01"], "glucose": [100.0], "age": [40.0]}
    )
    current = pd.DataFrame(
        {"event_date": ["2026-01-02"], "glucose": [180.0], "age": [41.0]}
    )

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
                            "share_of_drifted_columns": 0.5,
                            "drift_by_columns": {
                                "glucose": {"drift_detected": True},
                                "age": {"drift_detected": False},
                            },
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["glucose", "age"],
    )

    assert result.metrics["drift/benchmark/important_n_drifted_columns"] == 1.0
    assert result.metrics["drift/benchmark/important_share_drifted_columns"] == 0.5
    assert result.metrics["drift/benchmark/important_feature/glucose/drifted"] == 1.0
    assert result.metrics["drift/benchmark/important_feature/age/drifted"] == 0.0
    assert result.flagged == [
        {
            "comparison": "benchmark",
            "metric": "important_feature_drift",
            "feature": "glucose",
            "importance_rank": 1,
        }
    ]
    assert result.summary["important_features"] == {
        "features": ["glucose", "age"],
        "drifted_features": ["glucose"],
        "n_drifted_columns": 1,
        "share_drifted_columns": 0.5,
    }


def test_drift_report_important_feature_flags_include_importance_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "glucose": [100.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "glucose": [180.0]})

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
                            "share_of_drifted_columns": 1.0,
                            "drift_by_columns": {"glucose": {"drift_detected": True}},
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)
    importance = LumosResult(
        metrics={},
        summary={
            "methods": {
                "permutation": {
                    "features": [{"feature": "glucose", "importance_mean": 0.9}]
                }
            }
        },
    )

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        importance_result=importance,
    )

    assert result.flagged == [
        {
            "comparison": "benchmark",
            "metric": "important_feature_drift",
            "feature": "glucose",
            "importance_method": "permutation",
            "importance_rank": 1,
        }
    ]


def test_drift_report_can_disable_important_feature_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "glucose": [100.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "glucose": [180.0]})

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
                            "share_of_drifted_columns": 1.0,
                            "drift_by_columns": {"glucose": {"drift_detected": True}},
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)
    monkeypatch.setattr(settings.data, "alert_on_important_feature_drift", False)

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["glucose"],
    )

    assert result.metrics["drift/benchmark/important_feature/glucose/drifted"] == 1.0
    assert result.flagged == []


def test_drift_report_extracts_evidently_value_drift_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame(
        {"event_date": ["2026-01-01"], "x": [1.0], "y": [1.0]}
    )
    current = pd.DataFrame(
        {"event_date": ["2026-01-02"], "x": [10.0], "y": [1.0]}
    )

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
                        "metric_name": "DriftedColumnsCount(columns=x,y,drift_share=0.5)",
                        "config": {
                            "type": "evidently:metric_v2:DriftedColumnsCount",
                            "columns": ["x", "y"],
                            "drift_share": 0.5,
                        },
                        "value": {"count": 1.0, "share": 0.5},
                    },
                    {
                        "metric_name": "ValueDrift(column=x,method=K-S p_value,threshold=0.05)",
                        "config": {
                            "type": "evidently:metric_v2:ValueDrift",
                            "column": "x",
                            "method": "K-S p_value",
                            "threshold": 0.05,
                        },
                        "value": 0.01,
                    },
                    {
                        "metric_name": "ValueDrift(column=y,method=Z-test p_value,threshold=0.05)",
                        "config": {
                            "type": "evidently:metric_v2:ValueDrift",
                            "column": "y",
                            "method": "Z-test p_value",
                            "threshold": 0.05,
                        },
                        "value": 1.0,
                    },
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["x", "y"],
    )

    assert result.metrics["drift/benchmark/important_feature/x/drifted"] == 1.0
    assert result.metrics["drift/benchmark/important_feature/y/drifted"] == 0.0
    assert result.metrics["drift/benchmark/important_n_drifted_columns"] == 1.0
    assert result.flagged == [
        {
            "comparison": "benchmark",
            "metric": "important_feature_drift",
            "feature": "x",
            "importance_rank": 1,
        }
    ]


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

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        feature_columns=["x"],
        important_features=["x"],
        report_name="Installed Evidently Drift",
        evidently_kwargs={"report": {"tags": ["integration"]}},
    )

    assert "drift/benchmark/n_drifted_columns" in result.metrics
    assert "drift/benchmark/share_drifted_columns" in result.metrics
    assert result.metrics["drift/benchmark/important_feature/x/drifted"] == 1.0
    assert result.metrics["drift/benchmark/important_n_drifted_columns"] == 1.0
    assert any(
        flag.get("metric") == "important_feature_drift" and flag.get("feature") == "x"
        for flag in result.flagged
    )
    assert result.metadata["report_name"] == "Installed Evidently Drift"
