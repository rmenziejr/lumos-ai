from __future__ import annotations

import pandas as pd
import pytest

from lumosai.data.sampling import build_sample
from lumosai.exceptions import LumosValidationError


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "amount": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            "age": [31, 33, 39, 41, 44, 45, 47, 50, 52, 55],
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "prediction": [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        }
    )


def test_train_benchmark_excludes_temporal_columns_and_prefers_feature_columns() -> None:
    result = build_sample(
        make_frame(),
        role="train_benchmark",
        sample_size=6,
        target="target",
        feature_columns=["amount", "age", "day_of_week"],
        categorical_columns=["day_of_week"],
        time_column="event_date",
    )
    sample = result.artifacts["sample"]
    assert list(sample.columns) == ["target", "amount", "age", "day_of_week"]
    assert len(sample) == 6
    assert result.summary["role"] == "train_benchmark"
    assert result.summary["strategy"] == "stratified"
    assert result.summary["categorical_columns"] == ["day_of_week"]
    assert result.metadata["report_type"] == "sample"


def test_train_benchmark_excludes_temporal_columns_even_when_feature_columns_include_them() -> None:
    result = build_sample(
        make_frame(),
        role="train_benchmark",
        sample_size=4,
        target="target",
        feature_columns=["event_date", "amount", "day_of_week"],
        temporal_columns=["day_of_week"],
        time_column="event_date",
    )

    sample = result.artifacts["sample"]

    assert list(sample.columns) == ["target", "amount"]


def test_train_benchmark_rejects_only_temporal_feature_columns() -> None:
    with pytest.raises(LumosValidationError, match="non-temporal column"):
        build_sample(
            make_frame(),
            role="train_benchmark",
            feature_columns=["event_date", "day_of_week"],
            temporal_columns=["day_of_week"],
            time_column="event_date",
        )


def test_holdout_auto_uses_most_recent_rows_when_time_column_is_available() -> None:
    result = build_sample(
        make_frame(),
        role="holdout",
        sample_size=3,
        target="target",
        prediction="prediction",
        feature_columns=["amount"],
        time_column="event_date",
    )
    sample = result.artifacts["sample"]
    assert sample["event_date"].tolist() == list(pd.date_range("2026-01-08", periods=3, freq="D"))
    assert list(sample.columns) == ["event_date", "target", "prediction", "amount"]
    assert result.summary["strategy"] == "temporal_recent"


def test_monitoring_window_returns_selected_frame_when_sample_size_is_none() -> None:
    result = build_sample(
        make_frame(),
        role="monitoring_window",
        sample_size=None,
        feature_columns=["amount"],
        time_column="event_date",
    )
    sample = result.artifacts["sample"]
    assert len(sample) == 10
    assert list(sample.columns) == ["event_date", "amount"]


def test_build_sample_rejects_categorical_column_outside_selected_columns() -> None:
    with pytest.raises(LumosValidationError, match="categorical_columns"):
        build_sample(
            make_frame(),
            role="train_benchmark",
            feature_columns=["amount"],
            categorical_columns=["day_of_week"],
        )


def test_build_sample_rejects_prediction_without_target_for_holdout() -> None:
    with pytest.raises(LumosValidationError, match="target"):
        build_sample(make_frame(), role="holdout", prediction="prediction")


def test_build_sample_writes_local_artifact(tmp_path) -> None:
    path = tmp_path / "sample.csv"

    result = build_sample(
        make_frame(),
        role="monitoring_window",
        sample_size=2,
        feature_columns=["amount"],
        time_column="event_date",
        artifact_path=path,
    )

    assert path.exists()
    assert result.metadata["sample_artifact_path"] == str(path)
