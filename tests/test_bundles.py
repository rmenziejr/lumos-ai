from __future__ import annotations

import pandas as pd
import pytest

from lumosai.bundles import monitoring_report
from lumosai.exceptions import LumosValidationError


def make_monitoring_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "amount": [10, 12, 14, 16, 18, 20],
            "age": [30, 31, 32, 33, 34, 35],
            "target": [0, 1, 0, 1, 0, 1],
            "prediction": [0, 1, 0, 0, 0, 1],
            "region": ["a", "a", "b", "b", "a", "b"],
        }
    )


def test_monitoring_report_requires_temporal_features_for_drift() -> None:
    with pytest.raises(LumosValidationError, match="temporal_features"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            feature_columns=["amount", "age"],
        )


def test_monitoring_report_requires_prediction_when_performance_enabled() -> None:
    with pytest.raises(LumosValidationError, match="prediction"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            target="target",
            include_performance=True,
            feature_columns=["amount", "age"],
        )


def test_monitoring_report_requires_protected_attribute_when_bias_enabled() -> None:
    with pytest.raises(LumosValidationError, match="protected_attribute"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            target="target",
            prediction="prediction",
            include_bias=True,
            feature_columns=["amount", "age"],
        )
