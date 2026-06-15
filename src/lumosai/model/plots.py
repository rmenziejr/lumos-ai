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
        f"<section><h2>{html.escape(heading)}</h2>{content}</section>"
        for heading, content in sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    body {{
      color: #17202a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 2rem;
    }}
    h1 {{ font-size: 1.6rem; margin-bottom: 1.5rem; }}
    h2 {{
      border-bottom: 1px solid #d8dee4;
      font-size: 1.1rem;
      margin-top: 2rem;
      padding-bottom: 0.35rem;
    }}
    img {{ display: block; max-width: 960px; width: 100%; }}
    section {{ margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; margin-top: 0.75rem; }}
    th, td {{ border: 1px solid #d8dee4; padding: 0.35rem 0.55rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  {body}
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


def drift_fallback_html(*, title: str, summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(key))}</td>"
        f"<td>{html.escape(str(value))}</td>"
        "</tr>"
        for key, value in {**metadata, **summary}.items()
    )
    table = (
        "<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )
    return _html_document(title, [("Data Drift Report", table)])
