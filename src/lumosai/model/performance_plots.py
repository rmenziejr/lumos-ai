from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lumosai.model.plots import (
    _capture_plot,
    _classification_curve_plots,
    _confusion_matrix_plot,
    _decision_curve_plot,
    _figure_html,
    _html_document,
    _lift_plot,
    _metric_table,
    _threshold_performance_plot,
)
from lumosai.model.scores import ClassificationScores


def _regression_section(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[str, str]:
    actual = y_true.to_numpy(dtype=float)
    predicted = y_pred.to_numpy(dtype=float)
    residuals = actual - predicted

    if name == "predicted_vs_actual":
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(actual, predicted)
        lower = float(min(np.min(actual), np.min(predicted)))
        upper = float(max(np.max(actual), np.max(predicted)))
        ax.plot([lower, upper], [lower, upper], color="#6a737d", linestyle="--", linewidth=1)
        ax.set(xlabel="Actual", ylabel="Predicted", title="Predicted vs Actual")
        heading = "Predicted vs Actual"
    elif name == "residuals_vs_prediction":
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(predicted, residuals)
        ax.axhline(0.0, color="#6a737d", linestyle="--", linewidth=1)
        ax.set(xlabel="Predicted", ylabel="Residual", title="Residuals vs Prediction")
        heading = "Residuals vs Prediction"
    elif name == "residual_distribution":
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(residuals, bins=min(20, max(3, len(residuals))))
        ax.set(xlabel="Residual", ylabel="Rows", title="Residual Distribution")
        heading = "Residual Distribution"
    elif name == "residual_qq":
        ordered = np.sort(residuals)
        n = len(ordered)
        probabilities = (np.arange(1, n + 1) - 0.5) / n
        normal = NormalDist()
        theoretical = np.asarray([normal.inv_cdf(float(p)) for p in probabilities])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(theoretical, ordered)
        if n > 1 and np.std(theoretical) > 0:
            slope, intercept = np.polyfit(theoretical, ordered, 1)
            ax.plot(theoretical, slope * theoretical + intercept, color="#6a737d", linestyle="--", linewidth=1)
        ax.set(xlabel="Theoretical Normal Quantile", ylabel="Residual Quantile", title="Residual Q-Q Plot")
        heading = "Residual Q-Q Plot"
    else:
        raise ValueError(f"unknown regression plot: {name}")
    fig.tight_layout()
    return heading, _figure_html(fig, heading)


def selective_performance_html(
    *,
    title: str,
    frame: pd.DataFrame,
    target: str,
    prediction: str,
    task_type: str,
    metrics: dict[str, float],
    scores: ClassificationScores | None,
    lift_summary: dict[str, Any] | None,
    plots: set[str],
) -> str:
    sections: list[tuple[str, str]] = [("Metrics", _metric_table(metrics))]
    if task_type == "classification":
        if "confusion_matrix" in plots:
            sections.append(("Confusion Matrix", _confusion_matrix_plot(frame[target], frame[prediction])))
        if scores is not None:
            curves = dict(_classification_curve_plots(frame[target], scores))
            if "roc" in plots and "ROC Curve" in curves:
                sections.append(("ROC Curve", curves["ROC Curve"]))
            if "precision_recall" in plots and "Precision-Recall Curve" in curves:
                sections.append(("Precision-Recall Curve", curves["Precision-Recall Curve"]))
            if "capture" in plots:
                content = _capture_plot(lift_summary)
                if content is not None:
                    sections.append(("Observed Event Rate and Cumulative Capture", content))
            if "lift" in plots:
                content = _lift_plot(lift_summary)
                if content is not None:
                    sections.append(("Lift by Decile", content))
            if "threshold_performance" in plots:
                content = _threshold_performance_plot(frame[target], scores)
                if content is not None:
                    sections.append(("Threshold Performance", content))
            if "decision_curve" in plots:
                content = _decision_curve_plot(frame[target], scores)
                if content is not None:
                    sections.append(("Decision Curve Analysis", content))
    else:
        order = [
            "predicted_vs_actual",
            "residuals_vs_prediction",
            "residual_distribution",
            "residual_qq",
        ]
        for name in order:
            if name in plots:
                sections.append(_regression_section(name, frame[target], frame[prediction]))
    return _html_document(title, sections)
