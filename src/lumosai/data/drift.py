"""Data drift reporting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from inspect import signature
from pathlib import Path
from typing import Any

import pandas as pd

from lumosai.artifacts import (
    artifact_workspace,
    html_artifact_metadata,
    log_result_with_html_artifact,
    should_keep_html_artifact,
)
from lumosai.data.ingest import to_pandas
from lumosai.data.validation import (
    require_columns,
    require_no_duplicate_columns,
    validate_temporal_features,
)
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.mlflow import log_result
from lumosai.model.plots import drift_fallback_html
from lumosai.results import LumosResult
from lumosai.schema import filter_supported_kwargs, validate_categorical_columns
from lumosai.settings import settings

Report: Any | None = None
DataDriftPreset: Any | None = None
EVIDENTLY_PRESET_KWARGS = {
    "columns",
    "drift_share",
    "method",
    "cat_method",
    "num_method",
    "text_method",
    "threshold",
    "cat_threshold",
    "num_threshold",
    "text_threshold",
    "per_column_threshold",
}
EVIDENTLY_REPORT_KWARGS = {
    "metadata",
    "tags",
    "include_tests",
    "model_id",
    "reference_id",
    "batch_size",
    "dataset_id",
}


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


def _importance_feature_rows(importance_result: LumosResult) -> list[dict[str, Any]]:
    methods = importance_result.summary.get("methods")
    if not isinstance(methods, dict):
        msg = "importance_result must include permutation rows in summary['methods']"
        raise LumosValidationError(msg)
    permutation = methods.get("permutation")
    if not isinstance(permutation, dict):
        msg = "importance_result must include permutation rows in summary['methods']['permutation']"
        raise LumosValidationError(msg)
    rows = permutation.get("features")
    if not isinstance(rows, list):
        msg = "importance_result must include permutation rows in summary['methods']['permutation']['features']"
        raise LumosValidationError(msg)
    return rows


def _important_features_from_result(importance_result: LumosResult) -> list[str]:
    features: list[str] = []
    for row in _importance_feature_rows(importance_result):
        if not isinstance(row, dict) or not isinstance(row.get("feature"), str):
            msg = "importance_result permutation rows must include string feature names"
            raise LumosValidationError(msg)
        features.append(row["feature"])
    return features[: settings.data.important_drift_top_n]


def _resolve_important_features(
    *,
    important_features: list[str] | None,
    importance_result: LumosResult | None,
    analysis_columns: pd.Index,
) -> tuple[list[str], str | None]:
    if important_features is not None:
        resolved = list(important_features)
        if any(not isinstance(feature, str) for feature in resolved):
            msg = "important_features must contain string feature names"
            raise LumosValidationError(msg)
        source = "explicit"
    elif importance_result is not None:
        resolved = _important_features_from_result(importance_result)
        source = "importance_result"
    else:
        return [], None

    missing = [feature for feature in resolved if feature not in analysis_columns]
    if missing:
        msg = "important_features must be included in analyzed drift columns: "
        msg += ", ".join(missing)
        raise LumosValidationError(msg)
    return resolved, source


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


def _split_evidently_kwargs(
    evidently_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    top_level = filter_supported_kwargs(
        evidently_kwargs,
        allowed={"preset", "report"},
        parameter_name="evidently_kwargs",
    )
    preset_kwargs = filter_supported_kwargs(
        top_level.get("preset"),
        allowed=EVIDENTLY_PRESET_KWARGS,
        parameter_name="evidently_kwargs['preset']",
    )
    report_kwargs = filter_supported_kwargs(
        top_level.get("report"),
        allowed=EVIDENTLY_REPORT_KWARGS,
        parameter_name="evidently_kwargs['report']",
    )
    return preset_kwargs, report_kwargs


def _supported_constructor_kwargs(
    callable_obj: Any,
    kwargs: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    parameters = signature(callable_obj).parameters
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    unsupported = sorted(set(kwargs) - set(parameters))
    if unsupported:
        msg = f"{name} does not support keys for installed Evidently API: "
        msg += ", ".join(unsupported)
        raise LumosValidationError(msg)
    return kwargs


def _make_report(
    report_cls: Any,
    preset_cls: Any,
    *,
    preset_kwargs: dict[str, Any],
    report_kwargs: dict[str, Any],
) -> Any:
    preset_kwargs = _supported_constructor_kwargs(
        preset_cls,
        preset_kwargs,
        "evidently_kwargs['preset']",
    )
    report_kwargs = _supported_constructor_kwargs(
        report_cls,
        report_kwargs,
        "evidently_kwargs['report']",
    )
    preset = preset_cls(**preset_kwargs)
    try:
        return report_cls(metrics=[preset], **report_kwargs)
    except TypeError:
        return report_cls([preset], **report_kwargs)


def _current_evidently_dataset(
    data: pd.DataFrame,
    *,
    categorical_columns: list[str],
) -> Any:
    if not categorical_columns:
        return data
    try:
        from evidently import DataDefinition, Dataset
    except ImportError:
        return data
    numerical_columns = [column for column in data.columns if column not in categorical_columns]
    definition = DataDefinition(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    return Dataset.from_pandas(data, data_definition=definition)


def _run_report(
    report: Any,
    *,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: Any,
    categorical_columns: list[str],
    report_name: str | None,
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
    kwargs: dict[str, Any] = {
        "reference_data": _current_evidently_dataset(
            reference_data,
            categorical_columns=categorical_columns,
        ),
        "current_data": _current_evidently_dataset(
            current_data,
            categorical_columns=categorical_columns,
        ),
    }
    if "name" in parameters and report_name is not None:
        kwargs["name"] = report_name
    return report.run(**kwargs)


def _write_drift_html(
    *,
    report: Any,
    run_result: Any,
    html_path: Path,
    title: str,
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    export_errors: list[str] = []
    for source in (run_result, report):
        if source is None:
            continue
        for method_name in ("save_html", "save"):
            method = getattr(source, method_name, None)
            if method is None:
                continue
            try:
                method(html_path)
                if html_path.exists():
                    return
            except TypeError as exc:
                export_errors.append(str(exc))
                try:
                    method(str(html_path))
                    if html_path.exists():
                        return
                except Exception as str_exc:
                    export_errors.append(str(str_exc))
            except Exception as exc:
                export_errors.append(str(exc))
        for method_name in ("as_html", "to_html", "html"):
            method = getattr(source, method_name, None)
            if method is None:
                continue
            try:
                rendered = method()
            except Exception as exc:
                export_errors.append(str(exc))
                rendered = None
            if isinstance(rendered, str):
                html_path.write_text(rendered, encoding="utf-8")
                return

    fallback_metadata = dict(metadata)
    if export_errors:
        fallback_metadata["native_html_export_errors"] = len(export_errors)
    html_path.write_text(
        drift_fallback_html(title=title, summary=summary, metadata=fallback_metadata),
        encoding="utf-8",
    )


def drift_report(
    reference: Any,
    current: Any,
    temporal_features: list[str],
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    column_mapping: Any = None,
    comparison: str = "benchmark",
    report_name: str | None = None,
    evidently_kwargs: dict[str, Any] | None = None,
    important_features: list[str] | None = None,
    importance_result: LumosResult | None = None,
    include_html: bool = True,
    experiment_name: str | None = None,
) -> LumosResult:
    """Compare reference and current frames for feature drift.

    Temporal features are validated and excluded from drift calculations.
    MLflow logging is enabled when `experiment_name` is provided or
    `settings.mlflow.default_experiment_name` is set.
    """

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
    if feature_columns is not None:
        temporal_feature_columns = [column for column in feature_columns if column in excluded]
        if temporal_feature_columns:
            msg = "feature_columns must not include temporal features: "
            msg += ", ".join(temporal_feature_columns)
            raise LumosValidationError(msg)
        if categorical_columns is not None:
            categorical_outside_features = [
                column for column in categorical_columns if column not in feature_columns
            ]
            if categorical_outside_features:
                msg = "categorical_columns must be included in analyzed columns: "
                msg += ", ".join(categorical_outside_features)
                raise LumosValidationError(msg)
        require_no_duplicate_columns(reference_for_drift)
        require_no_duplicate_columns(current_for_drift)

        require_columns(reference_for_drift, feature_columns)
        require_columns(current_for_drift, feature_columns)
        reference_for_drift = reference_for_drift[feature_columns].copy()
        current_for_drift = current_for_drift[feature_columns].copy()
    if reference_for_drift.shape[1] == 0 or current_for_drift.shape[1] == 0:
        msg = "drift_report requires at least one non-temporal column"
        raise LumosValidationError(msg)
    selected_categorical_columns = validate_categorical_columns(
        reference_for_drift,
        categorical_columns=categorical_columns,
        analysis_columns=reference_for_drift.columns,
    )
    resolved_important_features, important_feature_source = _resolve_important_features(
        important_features=important_features,
        importance_result=importance_result,
        analysis_columns=reference_for_drift.columns,
    )

    report_cls, preset_cls = _evidently_classes()
    preset_kwargs, report_kwargs = _split_evidently_kwargs(evidently_kwargs)
    if feature_columns is not None:
        if "columns" in preset_kwargs and list(preset_kwargs["columns"]) != feature_columns:
            msg = "evidently_kwargs['preset']['columns'] must match feature_columns"
            raise LumosValidationError(msg)
        preset_kwargs["columns"] = list(feature_columns)
    report = _make_report(
        report_cls,
        preset_cls,
        preset_kwargs=preset_kwargs,
        report_kwargs=report_kwargs,
    )
    run_result = _run_report(
        report,
        reference_data=reference_for_drift,
        current_data=current_for_drift,
        column_mapping=column_mapping,
        categorical_columns=selected_categorical_columns,
        report_name=report_name,
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

    metadata = {
        "report_type": "drift",
        "comparison": safe_comparison,
        **({"report_name": report_name} if report_name is not None else {}),
        **({"feature_columns": list(feature_columns)} if feature_columns is not None else {}),
        **(
            {"categorical_columns": selected_categorical_columns}
            if selected_categorical_columns
            else {}
        ),
        **(
            {"important_features": resolved_important_features}
            if resolved_important_features
            else {}
        ),
        **(
            {"important_feature_source": important_feature_source}
            if important_feature_source is not None
            else {}
        ),
    }
    artifacts: dict[str, Any] = {}
    html_path: Path | None = None
    if include_html:
        title = report_name or "Data Drift Report"
        keep_local = should_keep_html_artifact(experiment_name=experiment_name)
        with artifact_workspace(keep_local=keep_local) as workspace:
            html_path = workspace / "drift_report.html"
            _write_drift_html(
                report=report,
                run_result=run_result,
                html_path=html_path,
                title=title,
                summary=summary,
                metadata=metadata,
            )
            artifacts, _ = html_artifact_metadata(
                html_path,
                artifact_path="drift",
                experiment_name=experiment_name,
            )
            result = LumosResult(
                metrics=metrics,
                summary=summary,
                flagged=flagged,
                artifacts=artifacts,
                report=report,
                metadata=metadata,
            )
            return log_result_with_html_artifact(
                result,
                html_path=html_path,
                artifact_path="drift",
                experiment_name=experiment_name,
            )

    result = LumosResult(
        metrics=metrics,
        summary=summary,
        flagged=flagged,
        artifacts=artifacts,
        report=report,
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
