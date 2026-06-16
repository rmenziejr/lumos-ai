"""Public API for lumosai."""

from typing import Any

from lumosai.results import LumosResult, LumosRun
from lumosai.settings import settings

__all__ = [
    "LumosResult",
    "LumosRun",
    "bias_report",
    "build_sample",
    "calibration_report",
    "drift_report",
    "feature_importance",
    "get_metrics",
    "monitoring_report",
    "performance_drift_report",
    "performance_report",
    "profile",
    "settings",
    "training_report",
]


def __getattr__(name: str) -> Any:
    if name in {"build_sample", "drift_report", "profile"}:
        from lumosai import data

        return getattr(data, name)
    if name in {
        "bias_report",
        "calibration_report",
        "feature_importance",
        "get_metrics",
        "performance_drift_report",
        "performance_report",
    }:
        from lumosai import model

        return getattr(model, name)
    if name == "monitoring_report":
        from lumosai.bundles import monitoring_report

        return monitoring_report
    if name == "training_report":
        from lumosai.bundles import training_report

        return training_report
    raise AttributeError(f"module 'lumosai' has no attribute {name!r}")
