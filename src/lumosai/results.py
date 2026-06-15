from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def json_safe_artifacts(value: Any) -> Any:
    """Convert artifact metadata into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return f"<DataFrame shape={value.shape}>"
    if isinstance(value, dict):
        return {str(key): json_safe_artifacts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_artifacts(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_artifacts(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return f"<{type(value).__name__}>"


@dataclass(slots=True)
class LumosResult:
    """Structured result returned by lumosai report functions."""

    metrics: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    report: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe fields for logging, APIs, or persisted summaries."""
        return {
            "metrics": self.metrics,
            "summary": self.summary,
            "flagged": self.flagged,
            "artifacts": json_safe_artifacts(self.artifacts),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class LumosRun:
    """Grouped result returned by lumosai bundle functions."""

    run_type: str
    results: dict[str, LumosResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, float]:
        """Return child result metrics merged into one dictionary."""

        merged: dict[str, float] = {}
        for result in self.results.values():
            merged.update(result.metrics)
        return merged

    @property
    def flagged(self) -> list[dict[str, Any]]:
        """Return flagged child findings annotated with their result key."""

        findings: list[dict[str, Any]] = []
        for key, result in self.results.items():
            for item in result.flagged:
                findings.append({**item, "result_key": key})
        return findings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the grouped run."""

        return {
            "run_type": self.run_type,
            "metrics": self.metrics,
            "flagged": self.flagged,
            "metadata": self.metadata,
            "results": {key: result.to_dict() for key, result in self.results.items()},
        }
