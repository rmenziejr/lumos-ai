"""Model monitoring helpers."""

from typing import Any

__all__ = [
    "CLASSIFICATION_METRICS",
    "CLASSIFICATION_PROBABILITY_METRICS",
    "ClassificationMetric",
    "MetricPreset",
    "PERFORMANCE_METRICS",
    "PerformanceMetric",
    "REGRESSION_METRICS",
    "RegressionMetric",
    "bias_report",
    "calibration_report",
    "feature_importance",
    "get_metrics",
    "performance_drift_report",
    "performance_report",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CLASSIFICATION_METRICS",
        "CLASSIFICATION_PROBABILITY_METRICS",
        "ClassificationMetric",
        "MetricPreset",
        "PERFORMANCE_METRICS",
        "PerformanceMetric",
        "REGRESSION_METRICS",
        "RegressionMetric",
    }:
        from lumosai.model import metrics

        return getattr(metrics, name)
    if name == "bias_report":
        from lumosai.model.bias import bias_report

        return bias_report
    if name == "calibration_report":
        from lumosai.model.calibration import calibration_report

        return calibration_report
    if name == "feature_importance":
        from lumosai.model.importance import feature_importance

        return feature_importance
    if name == "get_metrics":
        from lumosai.model.metrics import get_metrics

        return get_metrics
    if name == "performance_report":
        from lumosai.model.performance import performance_report

        return performance_report
    if name == "performance_drift_report":
        from lumosai.model.performance_drift import performance_drift_report

        return performance_drift_report
    raise AttributeError(f"module 'lumosai.model' has no attribute {name!r}")
