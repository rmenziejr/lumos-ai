from __future__ import annotations

import pandas as pd

from lumosai.model.bias import bias_report


def test_bias_report_flags_group_disparity_for_classification() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 1, 1, 1, 1, 1, 1],
            "prediction": [1, 1, 1, 1, 1, 1, 0, 0],
            "segment": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="classification",
    )

    assert result.metadata["report_type"] == "bias"
    assert result.metrics["bias/flags_count"] >= 1
    assert any(flag["protected_attribute"] == "segment" for flag in result.flagged)


def test_bias_report_bins_continuous_protected_attribute() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 0, 0],
            "prediction": [1, 0, 0, 0],
            "age": [22, 35, 62, 70],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute={"age": [0, 40, 120]},
        task_type="classification",
    )

    assert "age" in result.summary["by_attribute"]
    assert len(result.summary["by_attribute"]["age"]["by_group"]) == 2
