from __future__ import annotations

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.schema import (
    filter_supported_kwargs,
    select_analysis_frame,
    validate_categorical_columns,
)


def test_select_analysis_frame_orders_target_first() -> None:
    frame = pd.DataFrame(
        {
            "feature": [1, 2],
            "target": [0, 1],
            "other": [3, 4],
        }
    )

    result = select_analysis_frame(
        frame,
        target="target",
        feature_columns=["feature"],
    )

    assert list(result.columns) == ["target", "feature"]


def test_select_analysis_frame_keeps_time_column_for_sampling() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=2),
            "feature": [1, 2],
            "target": [0, 1],
        }
    )

    result = select_analysis_frame(
        frame,
        target="target",
        feature_columns=["feature"],
        required_columns=["event_date"],
    )

    assert list(result.columns) == ["target", "feature", "event_date"]


def test_select_analysis_frame_rejects_target_in_features() -> None:
    frame = pd.DataFrame({"target": [0, 1], "feature": [1, 2]})

    with pytest.raises(LumosValidationError, match="target"):
        select_analysis_frame(frame, target="target", feature_columns=["target", "feature"])


def test_validate_categorical_columns_requires_analysis_column() -> None:
    frame = pd.DataFrame({"feature": [1, 2], "other": [3, 4]})

    with pytest.raises(LumosValidationError, match="categorical_columns"):
        validate_categorical_columns(
            frame,
            categorical_columns=["other"],
            analysis_columns=["feature"],
        )


def test_filter_supported_kwargs_rejects_unknown_keys() -> None:
    with pytest.raises(LumosValidationError, match="unsupported"):
        filter_supported_kwargs(
            {"known": True, "unknown": False},
            allowed={"known"},
            parameter_name="example_kwargs",
        )
