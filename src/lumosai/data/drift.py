"""Data drift reporting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from inspect import signature
from typing import Any

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_no_duplicate_columns, validate_temporal_features
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
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
        from evidently import Report as EvidentlyReport  # type: ignore[import-untyped]
        from evidently.presets import (  # type: ignore[import-untyped]
            DataDriftPreset as EvidentlyDataDriftPreset,
        )
    except ImportError:
        try:
            from evidently.metric_preset import (  # type: ignore[import-untyped]
                DataDriftPreset as EvidentlyDataDriftPreset,
            )
            from evidently.report import Report as EvidentlyReport  # type: ignore[import-untyped]
        except ImportError as legacy_exc:
            msg = "drift reporting requires evidently and its runtime dependencies"
            raise LumosOptionalDependencyError(msg) from legacy_exc
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
        value = metric.get("value", {})
        if isinstance(value, dict) and {"count", "share"}.intersection(value):
            share = float(value.get("share", 0.0))
            return {
                "dataset_drift": share > settings.data.drift_share_threshold,
                "n_drifted_columns": int(value.get("count", 0)),
                "share_drifted_columns": share,
            }
    return {"dataset_drift": False, "n_drifted_columns": 0, "share_drifted_columns": 0.0}


def _report_payload(report: Any, run_result: Any) -> dict[str, Any]:
    payload_source = run_result if run_result is not None else report
    for method_name in ("as_dict", "dict", "dump_dict"):
        method = getattr(payload_source, method_name, None)
        if method is not None:
            payload = method()
            if isinstance(payload, dict):
                return payload
    msg = "evidently report did not expose a supported dictionary payload"
    raise LumosOptionalDependencyError(msg)


def _contains_temporal_column(value: Any, temporal_features: set[str]) -> bool:
    if isinstance(value, str):
        return value in temporal_features
    if isinstance(value, Mapping):
        return any(
            _contains_temporal_column(key, temporal_features)
            or _contains_temporal_column(item, temporal_features)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(_contains_temporal_column(item, temporal_features) for item in value)
    return False


def _validate_column_mapping(column_mapping: Any, temporal_features: set[str]) -> None:
    if column_mapping is None:
        return
    if _contains_temporal_column(column_mapping, temporal_features):
        msg = "column_mapping must not reference temporal features excluded from drift"
        raise LumosValidationError(msg)


def _make_report(report_cls: Any, preset_cls: Any) -> Any:
    try:
        return report_cls(metrics=[preset_cls()])
    except TypeError:
        return report_cls([preset_cls()])


def _run_report(
    report: Any,
    *,
    reference_data: Any,
    current_data: Any,
    column_mapping: Any,
) -> Any:
    parameters = signature(report.run).parameters
    if "column_mapping" in parameters:
        return report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )
    if column_mapping is not None:
        msg = "column_mapping is not supported by the installed Evidently API"
        raise LumosValidationError(msg)
    return report.run(reference_data=reference_data, current_data=current_data)


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
    _validate_column_mapping(column_mapping, excluded)
    reference_for_drift = reference_pd.drop(columns=list(excluded), errors="ignore")
    current_for_drift = current_pd.drop(columns=list(excluded), errors="ignore")
    if reference_for_drift.shape[1] == 0 or current_for_drift.shape[1] == 0:
        msg = "drift_report requires at least one non-temporal column"
        raise LumosValidationError(msg)

    report_cls, preset_cls = _evidently_classes()
    report = _make_report(report_cls, preset_cls)
    run_result = _run_report(
        report,
        reference_data=reference_for_drift,
        current_data=current_for_drift,
        column_mapping=column_mapping,
    )
    summary = _extract_drift_summary(_report_payload(report, run_result))
    safe_comparison = safe_comparison_name(comparison)
    metrics = {
        f"drift/{safe_comparison}/n_drifted_columns": float(summary["n_drifted_columns"]),
        f"drift/{safe_comparison}/share_drifted_columns": float(summary["share_drifted_columns"]),
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
