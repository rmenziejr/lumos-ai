from __future__ import annotations

import base64
import html
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from lumosai.model.scores import ClassificationScores, safe_label


def _figure_html(fig: Any, alt: str) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=144)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def _html_document(title: str, sections: list[tuple[str, str]]) -> str:
    safe_title = html.escape(title)
    body = "\n".join(
        f'<section class="report-section"><h2>{html.escape(heading)}</h2>{content}</section>'
        for heading, content in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f7f9fc;
      --surface: #ffffff;
      --surface-soft: #f8fafc;
      --text: #17202a;
      --muted: #5f6b7a;
      --border: #d8dee8;
      --accent: #246bfe;
      --shadow: 0 10px 28px rgba(19, 36, 64, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      margin: 0;
      padding: 32px;
    }}
    .report-shell {{
      margin: 0 auto;
      max-width: 1180px;
    }}
    .report-header {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
      padding: 24px 28px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      margin: 0 0 8px;
      text-transform: uppercase;
    }}
    h1 {{
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: 0;
      line-height: 1.2;
      margin: 0;
    }}
    h2 {{
      border-bottom: 1px solid var(--border);
      font-size: 1.05rem;
      font-weight: 650;
      letter-spacing: 0;
      margin: 0 0 14px;
      padding-bottom: 10px;
    }}
    .report-section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
      overflow: hidden;
      padding: 20px;
    }}
    img {{
      display: block;
      height: auto;
      margin-top: 8px;
      max-width: 980px;
      width: 100%;
    }}
    table {{
      border-collapse: collapse;
      font-size: 0.92rem;
      margin-top: 0.75rem;
      min-width: 420px;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 0.55rem 0.7rem;
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{
      color: var(--text);
      font-weight: 600;
      text-align: left;
    }}
    th {{
      background: var(--surface-soft);
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 720px) {{
      body {{ padding: 16px; }}
      .report-header, .report-section {{
        border-radius: 8px;
        padding: 16px;
      }}
      h1 {{ font-size: 1.45rem; }}
      table {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <main class="report-shell">
    <header class="report-header">
      <p class="eyebrow">Lumos AI Report</p>
      <h1>{safe_title}</h1>
    </header>
    {body}
  </main>
</body>
</html>
"""


def _metric_table(metrics: dict[str, float]) -> str:
    rows = []
    for name, value in sorted(metrics.items()):
        metric_name = name.split("/", 1)[-1]
        rows.append(
            "<tr>"
            f"<td>{html.escape(metric_name)}</td>"
            f"<td>{html.escape(f'{value:.6g}')}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _confusion_matrix_plot(y_true: pd.Series, y_pred: pd.Series) -> str:
    labels = sorted(pd.unique(pd.concat([y_true, y_pred], ignore_index=True)).tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[str(label) for label in labels],
        yticklabels=[str(label) for label in labels],
        ylabel="Actual",
        xlabel="Prediction",
        title="Confusion Matrix",
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="#17202a")
    fig.tight_layout()
    return _figure_html(fig, "Confusion Matrix")


def _class_labels_to_plot(scores: ClassificationScores) -> list[Any]:
    return [scores.positive_label] if scores.positive_label is not None else scores.labels


def _classification_curve_plots(
    y_true: pd.Series,
    scores: ClassificationScores,
) -> list[tuple[str, str]]:
    roc_fig, roc_ax = plt.subplots(figsize=(6, 4))
    pr_fig, pr_ax = plt.subplots(figsize=(6, 4))
    plotted_roc = False
    plotted_pr = False
    for label in _class_labels_to_plot(scores):
        class_index = scores.label_index(label)
        events = (y_true == label).to_numpy(dtype=int)
        if len(np.unique(events)) < 2:
            continue
        probabilities = scores.values[:, class_index]
        fpr, tpr, _ = roc_curve(events, probabilities)
        precision, recall, _ = precision_recall_curve(events, probabilities)
        label_text = "positive" if scores.positive_label is not None else str(label)
        roc_ax.plot(fpr, tpr, label=label_text)
        pr_ax.plot(recall, precision, label=label_text)
        plotted_roc = True
        plotted_pr = True

    sections: list[tuple[str, str]] = []
    if plotted_roc:
        roc_ax.plot([0, 1], [0, 1], color="#6a737d", linestyle="--", linewidth=1)
        roc_ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        roc_ax.legend(loc="lower right")
        roc_fig.tight_layout()
        sections.append(("ROC Curve", _figure_html(roc_fig, "ROC Curve")))
    else:
        plt.close(roc_fig)
    if plotted_pr:
        pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        pr_ax.legend(loc="lower left")
        pr_fig.tight_layout()
        sections.append(
            ("Precision-Recall Curve", _figure_html(pr_fig, "Precision-Recall Curve"))
        )
    else:
        plt.close(pr_fig)
    return sections


def _lift_plot(lift_summary: dict[str, Any] | None) -> str | None:
    if not lift_summary:
        return None
    classes = lift_summary.get("classes", {})
    if not classes:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for class_key, rows in classes.items():
        deciles = [row["decile"] for row in rows if row.get("lift") is not None]
        lifts = [row["lift"] for row in rows if row.get("lift") is not None]
        if deciles:
            ax.plot(deciles, lifts, marker="o", label=str(class_key))
    ax.axhline(1.0, color="#6a737d", linestyle="--", linewidth=1)
    ax.set(xlabel="Decile", ylabel="Lift", title="Lift by Decile")
    ax.invert_xaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    return _figure_html(fig, "Lift by Decile")


def performance_html(
    *,
    title: str,
    frame: pd.DataFrame,
    target: str,
    prediction: str,
    task_type: str,
    metrics: dict[str, float],
    scores: ClassificationScores | None = None,
    lift_summary: dict[str, Any] | None = None,
) -> str:
    sections: list[tuple[str, str]] = [("Metrics", _metric_table(metrics))]
    if task_type == "classification":
        sections.append(
            ("Confusion Matrix", _confusion_matrix_plot(frame[target], frame[prediction]))
        )
        if scores is not None:
            sections.extend(_classification_curve_plots(frame[target], scores))
            lift_content = _lift_plot(lift_summary)
            if lift_content is not None:
                sections.append(("Lift by Decile", lift_content))
    else:
        sections.extend(_regression_plots(frame[target], frame[prediction]))
    return _html_document(title, sections)


def _distribution_plot(
    baseline: Any,
    current: Any,
    *,
    title: str,
    xlabel: str,
) -> str:
    baseline_array = np.asarray(baseline, dtype=float)
    current_array = np.asarray(current, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = min(20, max(3, int(np.sqrt(max(len(baseline_array), len(current_array))))))
    ax.hist(baseline_array, bins=bins, alpha=0.55, label="Baseline")
    ax.hist(current_array, bins=bins, alpha=0.55, label="Current")
    ax.set(xlabel=xlabel, ylabel="Rows", title=title)
    ax.legend(loc="best")
    fig.tight_layout()
    return _figure_html(fig, title)


def _metric_delta_table(metric_summary: dict[str, Any]) -> str:
    rows = []
    for name, values in sorted(metric_summary.items()):
        baseline = f"{float(values['baseline']):.6g}"
        current = f"{float(values['current']):.6g}"
        delta = f"{float(values['delta']):.6g}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(baseline)}</td>"
            f"<td>{html.escape(current)}</td>"
            f"<td>{html.escape(delta)}</td>"
            f"<td>{html.escape(str(values['flagged']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th><th>Flagged</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def performance_drift_html(
    *,
    title: str,
    metrics: dict[str, float],
    metric_summary: dict[str, Any] | None = None,
    score_distributions: dict[str, tuple[Any, Any]] | None = None,
    residual_distribution: tuple[Any, Any] | None = None,
    residual_scatter: tuple[Any, Any, str] | None = None,
) -> str:
    sections: list[tuple[str, str]] = [("Metrics", _metric_table(metrics))]
    if metric_summary:
        sections.append(("Metric Drift", _metric_delta_table(metric_summary)))
    for name, (baseline, current) in (score_distributions or {}).items():
        heading = "Score Distribution" if name == "score" else f"Score Distribution: {name}"
        sections.append(
            (
                heading,
                _distribution_plot(
                    baseline,
                    current,
                    title=heading,
                    xlabel="Prediction Score",
                ),
            )
        )
    if residual_distribution is not None:
        sections.append(
            (
                "Residual Distribution",
                _distribution_plot(
                    residual_distribution[0],
                    residual_distribution[1],
                    title="Residual Distribution",
                    xlabel="Residual",
                ),
            )
        )
    if residual_scatter is not None:
        x_values, residuals, xlabel = residual_scatter
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(np.asarray(x_values, dtype=float), np.asarray(residuals, dtype=float))
        ax.axhline(0.0, color="#6a737d", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="Residual", title="Current Residuals")
        fig.tight_layout()
        sections.append(("Current Residuals", _figure_html(fig, "Current Residuals")))
    return _html_document(title, sections)


def _regression_plots(y_true: pd.Series, y_pred: pd.Series) -> list[tuple[str, str]]:
    residuals = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    predicted = y_pred.to_numpy(dtype=float)
    actual = y_true.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(actual, predicted)
    lower = float(min(np.min(actual), np.min(predicted)))
    upper = float(max(np.max(actual), np.max(predicted)))
    ax.plot([lower, upper], [lower, upper], color="#6a737d", linestyle="--", linewidth=1)
    ax.set(xlabel="Actual", ylabel="Predicted", title="Predicted vs Actual")
    fig.tight_layout()
    predicted_html = _figure_html(fig, "Predicted vs Actual")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(predicted, residuals)
    ax.axhline(0.0, color="#6a737d", linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted", ylabel="Residual", title="Residuals vs Prediction")
    fig.tight_layout()
    residual_html = _figure_html(fig, "Residuals vs Prediction")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=min(20, max(3, len(residuals))))
    ax.set(xlabel="Residual", ylabel="Rows", title="Residual Distribution")
    fig.tight_layout()
    distribution_html = _figure_html(fig, "Residual Distribution")

    return [
        ("Predicted vs Actual", predicted_html),
        ("Residuals vs Prediction", residual_html),
        ("Residual Distribution", distribution_html),
    ]


def calibration_html(*, title: str, calibration_summary: dict[str, Any]) -> str:
    sections: list[tuple[str, str]] = []
    for class_key, rows in calibration_summary["classes"].items():
        fig, ax = plt.subplots(figsize=(5, 4))
        predicted = [row["mean_predicted_probability"] for row in rows if row["rows"] > 0]
        observed = [row["observed_rate"] for row in rows if row["rows"] > 0]
        ax.plot([0, 1], [0, 1], color="#6a737d", linestyle="--", linewidth=1)
        if predicted:
            ax.plot(predicted, observed, marker="o")
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean Predicted Probability",
            ylabel="Observed Rate",
            title=f"Calibration Curve: {class_key}",
        )
        fig.tight_layout()
        sections.append(
            (
                f"Calibration Curve: {safe_label(class_key)}",
                "<p>Observed Rate by Mean Predicted Probability.</p>"
                + _figure_html(fig, f"Calibration Curve {class_key}"),
            )
        )
    return _html_document(title, sections)


def _importance_plot(rows: list[dict[str, Any]], *, title: str, include_error: bool) -> str:
    ordered = list(reversed(rows[:20]))
    features = [str(row["feature"]) for row in ordered]
    means = [float(row["importance_mean"]) for row in ordered]
    errors = [float(row.get("importance_std", 0.0)) for row in ordered] if include_error else None

    fig_height = max(3.0, 0.35 * len(features) + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(features, means, xerr=errors)
    ax.set(xlabel="Mean Importance", title=title)
    fig.tight_layout()
    return _figure_html(fig, title)


def importance_html(*, title: str, methods: dict[str, dict[str, Any]]) -> str:
    sections: list[tuple[str, str]] = []
    if "permutation" in methods:
        sections.append(
            (
                "Permutation Importance",
                _importance_plot(
                    methods["permutation"]["features"],
                    title="Permutation Importance",
                    include_error=True,
                ),
            )
        )
    if "shap" in methods:
        sections.append(
            (
                "SHAP Importance",
                _importance_plot(
                    methods["shap"]["features"],
                    title="SHAP Importance",
                    include_error=False,
                ),
            )
        )
    return _html_document(title, sections)


def _bias_group_size_plot(rows: list[dict[str, Any]], *, attribute: str) -> str:
    groups = [str(row["group"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    fig_height = max(3.0, 0.35 * len(groups) + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(groups, counts)
    ax.set(xlabel="Rows", title=f"Group Size: {attribute}")
    fig.tight_layout()
    return _figure_html(fig, f"Group Size {attribute}")


def _bias_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "positive_prediction_rate",
        "roc_auc",
        "pr_auc",
        "mae",
        "rmse",
        "r2",
        "mean_absolute_residual",
        "abs_mean_residual",
    ]
    available = {key for row in rows for key in row if key not in {"group", "count"}}
    ordered = [metric for metric in preferred if metric in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered[:6]


def _bias_metric_gap_plot(rows: list[dict[str, Any]], *, attribute: str) -> str:
    metric_names = _bias_metric_names(rows)
    gap_rows: list[tuple[str, str, float]] = []
    for metric_name in metric_names:
        finite_rows: list[tuple[str, float]] = []
        for row in rows:
            value = row.get(metric_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                finite_rows.append((str(row["group"]), numeric_value))
        if len(finite_rows) < 2:
            continue
        lower_is_better = metric_name in {
            "mae",
            "rmse",
            "log_loss",
            "mean_absolute_residual",
            "abs_mean_residual",
        }
        best = min(value for _, value in finite_rows) if lower_is_better else max(
            value for _, value in finite_rows
        )
        for group, value in finite_rows:
            gap = value - best if lower_is_better else best - value
            gap_rows.append((metric_name, group, max(0.0, gap)))

    if not gap_rows:
        return "<p>No comparable finite metric gaps.</p>"

    labels = [f"{metric} / {group}" for metric, group, _ in gap_rows]
    gaps = [gap for _, _, gap in gap_rows]
    x = np.arange(len(labels))
    fig_width = max(8.0, 0.45 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    colors = ["#d73a49" if gap > 0 else "#2ea44f" for gap in gaps]
    ax.bar(x, gaps, color=colors)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set(ylabel="Gap From Best Group", title=f"Metric Gap From Best Group: {attribute}")
    fig.tight_layout()
    return _figure_html(fig, f"Metric Gap From Best Group {attribute}")


def _bias_flag_table(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "<p>No flagged comparisons.</p>"
    rows = []
    for flag in flags:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(flag.get('protected_attribute', '')))}</td>"
            f"<td>{html.escape(str(flag.get('group', '')))}</td>"
            f"<td>{html.escape(str(flag.get('metric', '')))}</td>"
            f"<td>{html.escape(str(flag.get('group_value', '')))}</td>"
            f"<td>{html.escape(str(flag.get('threshold', '')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Attribute</th><th>Group</th><th>Metric</th><th>Group Value</th><th>Threshold</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def bias_html(*, title: str, summary: dict[str, Any], flagged: list[dict[str, Any]]) -> str:
    sections: list[tuple[str, str]] = []
    by_attribute = summary.get("by_attribute", {})
    if isinstance(by_attribute, dict):
        for attribute, attribute_summary in by_attribute.items():
            if not isinstance(attribute_summary, dict):
                continue
            rows = attribute_summary.get("by_group", [])
            if not isinstance(rows, list) or not rows:
                continue
            sections.append(
                (
                    f"Group Size: {attribute}",
                    _bias_group_size_plot(rows, attribute=str(attribute)),
                )
            )
            sections.append(
                (
                    f"Metric Gap From Best Group: {attribute}",
                    _bias_metric_gap_plot(rows, attribute=str(attribute)),
                )
            )
    sections.append(("Flagged Comparisons", _bias_flag_table(flagged)))
    return _html_document(title, sections)
