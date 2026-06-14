"""Model monitoring helpers."""

from typing import Any

__all__ = ["bias_report", "feature_importance", "get_metrics", "performance_report"]


def __getattr__(name: str) -> Any:
    if name == "bias_report":
        from lumosai.model.bias import bias_report

        return bias_report
    if name == "feature_importance":
        from lumosai.model.importance import feature_importance

        return feature_importance
    if name == "get_metrics":
        from lumosai.model.metrics import get_metrics

        return get_metrics
    if name == "performance_report":
        from lumosai.model.performance import performance_report

        return performance_report
    raise AttributeError(f"module 'lumosai.model' has no attribute {name!r}")
