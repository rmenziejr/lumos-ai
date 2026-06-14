"""Data monitoring helpers."""

from typing import Any

__all__ = ["drift_report", "profile", "temporal_sample"]


def __getattr__(name: str) -> Any:
    if name == "drift_report":
        from lumosai.data.drift import drift_report

        return drift_report
    if name in {"profile", "temporal_sample"}:
        from lumosai.data.profiling import profile, temporal_sample

        return {"profile": profile, "temporal_sample": temporal_sample}[name]
    raise AttributeError(f"module 'lumosai.data' has no attribute {name!r}")
