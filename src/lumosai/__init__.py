"""Public API for lumosai."""

from typing import Any

from lumosai.results import LumosResult
from lumosai.settings import settings

__all__ = [
    "LumosResult",
    "bias_report",
    "drift_report",
    "get_metrics",
    "performance_report",
    "profile",
    "settings",
]


def __getattr__(name: str) -> Any:
    if name in {"drift_report", "profile"}:
        from lumosai import data

        return getattr(data, name)
    if name in {"bias_report", "get_metrics", "performance_report"}:
        from lumosai import model

        return getattr(model, name)
    raise AttributeError(f"module 'lumosai' has no attribute {name!r}")
