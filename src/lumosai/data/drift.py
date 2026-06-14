"""Data drift reporting."""

from __future__ import annotations

import re
from typing import Any

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_no_duplicate_columns, validate_temporal_features
from lumosai.exceptions import LumosOptionalDependencyError
from lumosai.mlflow import log_result
from lumosai.results import LumosResult
from lumosai.settings import settings

Report: Any | None = None
DataDriftPreset: Any | None = None


def _evidently_classes() -> tuple[Any, Any]:
    global DataDriftPreset, Report
    if Report is not None and DataDriftPreset is not None:
        return Report, DataDriftPreset
    try:
        from evidently.metric_preset import (  # type: ignore[import-untyped]
            DataDriftPreset as EvidentlyDataDriftPreset,
        )
        from evidently.report import Report as EvidentlyReport  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "drift reporting requires evidently and its runtime dependencies"
        raise LumosOptionalDependencyError(msg) from exc
    DataDriftPreset = EvidentlyDataDriftPreset
    Report = EvidentlyReport
    return Report, DataDriftPreset


def safe_comparison_name(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "benchmark"


def _extract_drift_summary(report_payload: dict[str, Any]) -> dict[str, Any]:
    for metric in report_payload.get("metrics", []):
        result = metric.get("result", {})
        if "number_of_drifted_columns" in result or "share_of_drifted_columns" in result:
            return {
                "dataset_drift": bool(result.get("dataset_drift", False)),
                "n_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
                "share_drifted_columns": float(result.get("share_of_drifted_columns", 0.0)),
            }
    return {"dataset_drift": False, "n_drifted_columns": 0, "share_drifted_columns": 0.0}


def drift_report(
    reference: Any,
    current: Any,
    temporal_features: list[str],
    column_mapping: Any = None,
    comparison: str = "benchmark",
    experiment_name: str | None = None,
) -> LumosResult:
    reference_pd = to_pandas(reference)
    current_pd = to_pandas(current)
    require_no_duplicate_columns(reference_pd)
    require_no_duplicate_columns(current_pd)
    validate_temporal_features(reference_pd, temporal_features)
    validate_temporal_features(current_pd, temporal_features)

    excluded = set(temporal_features)
    reference_for_drift = reference_pd.drop(columns=list(excluded), errors="ignore")
    current_for_drift = current_pd.drop(columns=list(excluded), errors="ignore")

    report_cls, preset_cls = _evidently_classes()
    report = report_cls(metrics=[preset_cls()])
    report.run(
        reference_data=reference_for_drift,
        current_data=current_for_drift,
        column_mapping=column_mapping,
    )
    summary = _extract_drift_summary(report.as_dict())
    safe_comparison = safe_comparison_name(comparison)
    metrics = {
        f"drift/{safe_comparison}/n_drifted_columns": float(summary["n_drifted_columns"]),
        f"drift/{safe_comparison}/share_drifted_columns": float(
            summary["share_drifted_columns"]
        ),
    }
    flagged: list[dict[str, Any]] = []
    if summary["share_drifted_columns"] > settings.data.drift_share_threshold:
        flagged.append(
            {
                "comparison": safe_comparison,
                "metric": "share_drifted_columns",
                "value": summary["share_drifted_columns"],
                "threshold": settings.data.drift_share_threshold,
            }
        )

    result = LumosResult(
        metrics=metrics,
        summary=summary,
        flagged=flagged,
        report=report,
        metadata={"report_type": "drift", "comparison": safe_comparison},
    )
    log_result(result, experiment_name=experiment_name)
    return result
