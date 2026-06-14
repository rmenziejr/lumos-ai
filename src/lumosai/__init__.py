"""Public API for lumosai."""

from typing import Any

from lumosai.results import LumosResult, LumosRun
from lumosai.settings import settings

__all__ = [
    "LumosResult",
    "LumosRun",
    "bias_report",
    "build_sample",
    "drift_report",
    "feature_importance",
    "get_metrics",
    "monitoring_report",
    "performance_report",
    "profile",
    "settings",
]


def __getattr__(name: str) -> Any:
    if name in {"build_sample", "drift_report", "profile"}:
        from lumosai import data

        return getattr(data, name)
    if name in {"bias_report", "feature_importance", "get_metrics", "performance_report"}:
        from lumosai import model

        return getattr(model, name)
    if name == "monitoring_report":
        from lumosai.bundles import monitoring_report

        return monitoring_report
    raise AttributeError(f"module 'lumosai' has no attribute {name!r}")
