from __future__ import annotations

import pandas as pd

from lumosai.model import plots
from lumosai.model.bias import bias_report


def test_bias_metric_gap_plot_uses_readable_horizontal_layout(monkeypatch) -> None:
    rows = [
        {
            "group": group,
            "count": 10,
            "accuracy": 0.95 - index * 0.02,
            "precision": 0.94 - index * 0.02,
            "recall": 0.93 - index * 0.02,
            "f1": 0.92 - index * 0.02,
            "roc_auc": 0.91 - index * 0.02,
            "pr_auc": 0.90 - index * 0.02,
        }
        for index, group in enumerate(["a", "b", "c", "d"])
    ]
    captured = {}

    def capture_figure(fig, alt: str) -> str:
        captured["fig"] = fig
        captured["alt"] = alt
        return "<img>"

    monkeypatch.setattr(plots, "_figure_html", capture_figure)

    html = plots._bias_metric_gap_plot(rows, attribute="segment")

    assert html == "<img>"
    fig = captured["fig"]
    ax = fig.axes[0]
    y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert "accuracy / a" in y_labels
    assert ax.get_xlabel() == "Gap From Best Group"
    assert fig.get_figheight() > 6.0


def test_bias_report_limits_numeric_protected_attribute_to_max_bins() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1] * 10,
            "prediction": [0, 1] * 10,
            "age": list(range(20, 40)),
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["age"],
        task_type="classification",
        max_bins=3,
        include_plots=False,
    )

    groups = result.summary["by_attribute"]["age"]["by_group"]
    assert len(groups) <= 3
    assert sum(group["count"] for group in groups) == len(frame)
