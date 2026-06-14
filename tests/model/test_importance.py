from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from lumosai.exceptions import LumosValidationError
from lumosai.model.importance import feature_importance


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [0, 0, 1, 1, 0, 1, 0, 1],
            "noise": [4, 3, 4, 3, 4, 3, 4, 3],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_feature_importance_returns_sorted_permutation_metrics() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    result = feature_importance(
        model,
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        n_repeats=3,
        random_state=42,
        report_name="importance-baseline",
    )

    features = [row["feature"] for row in result.summary["features"]]

    assert features[0] == "signal"
    assert "importance/signal" in result.metrics
    assert result.metadata["report_type"] == "feature_importance"
    assert result.metadata["method"] == "permutation"
    assert result.metadata["report_name"] == "importance-baseline"


def test_feature_importance_samples_rows_before_permutation() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    result = feature_importance(
        model,
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        sample_size=4,
        random_state=42,
    )

    assert result.summary["rows"] == 4


def test_feature_importance_validates_columns() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosValidationError, match="missing required columns"):
        feature_importance(
            model,
            frame,
            target="missing",
            feature_columns=["signal", "absent"],
        )


def test_feature_importance_rejects_empty_feature_columns() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosValidationError, match="feature_columns"):
        feature_importance(
            model,
            frame,
            target="target",
            feature_columns=[],
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_repeats": 0}, "n_repeats"),
        ({"sample_size": 0}, "sample_size"),
    ],
)
def test_feature_importance_validates_positive_counts(kwargs, match) -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosValidationError, match=match):
        feature_importance(
            model,
            frame,
            target="target",
            feature_columns=["signal", "noise"],
            **kwargs,
        )
