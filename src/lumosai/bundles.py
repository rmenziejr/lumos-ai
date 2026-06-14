from __future__ import annotations

from typing import Any

import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.results import LumosRun


def _resolve_temporal_features(
    temporal_features: list[str] | None,
    time_column: str | None,
) -> list[str]:
    if temporal_features is not None:
        return list(temporal_features)
    if time_column is not None:
        return [time_column]
    msg = "monitoring_report requires temporal_features or time_column for drift"
    raise LumosValidationError(msg)


def _performance_expected(
    *,
    target: str | None,
    prediction: str | None,
    include_performance: bool | None,
) -> bool:
    return include_performance is True or (target is not None and prediction is not None)


def _bias_expected(
    *,
    protected_attribute: str | list[str] | dict[str, list[float]] | None,
    include_bias: bool | None,
) -> bool:
    return include_bias is True or protected_attribute is not None


def _protected_columns(
    protected_attribute: str | list[str] | dict[str, list[float]],
) -> list[str]:
    if isinstance(protected_attribute, str):
        return [protected_attribute]
    if isinstance(protected_attribute, dict):
        return list(protected_attribute)
    return list(protected_attribute)


def _preflight_monitoring_report(
    *,
    current: pd.DataFrame,
    benchmark: pd.DataFrame,
    previous_window: pd.DataFrame | None,
    target: str | None,
    prediction: str | None,
    prediction_score: str | None,
    feature_columns: list[str] | None,
    categorical_columns: list[str] | None,
    protected_attribute: str | list[str] | dict[str, list[float]] | None,
    temporal_features: list[str],
    include_performance: bool | None,
    include_bias: bool | None,
) -> None:
    drift_columns = list(feature_columns or current.columns)
    require_columns(current, drift_columns)
    require_columns(benchmark, drift_columns)
    require_columns(current, temporal_features)
    require_columns(benchmark, temporal_features)
    if previous_window is not None:
        require_columns(previous_window, drift_columns)
        require_columns(previous_window, temporal_features)
    if categorical_columns is not None:
        missing_categorical = [
            column for column in categorical_columns if column not in drift_columns
        ]
        if missing_categorical:
            msg = "categorical_columns must be included in feature_columns: "
            msg += ", ".join(missing_categorical)
            raise LumosValidationError(msg)
    if _performance_expected(
        target=target,
        prediction=prediction,
        include_performance=include_performance,
    ):
        if target is None:
            msg = "monitoring_report expected performance but target is missing"
            raise LumosValidationError(msg)
        if prediction is None:
            msg = "monitoring_report expected performance but prediction is missing"
            raise LumosValidationError(msg)
        required = [target, prediction]
        if prediction_score is not None:
            required.append(prediction_score)
        require_columns(current, required)
    if _bias_expected(protected_attribute=protected_attribute, include_bias=include_bias):
        if protected_attribute is None:
            msg = "monitoring_report expected bias but protected_attribute is missing"
            raise LumosValidationError(msg)
        if target is None or prediction is None:
            msg = "monitoring_report expected bias but target and prediction are required"
            raise LumosValidationError(msg)
        require_columns(current, [target, prediction, *_protected_columns(protected_attribute)])


def monitoring_report(
    current: Any,
    *,
    benchmark: Any,
    previous_window: Any | None = None,
    target: str | None = None,
    prediction: str | None = None,
    prediction_score: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    protected_attribute: str | list[str] | dict[str, list[float]] | None = None,
    temporal_features: list[str] | None = None,
    time_column: str | None = None,
    sample_size: int | None = None,
    include_performance: bool | None = None,
    include_bias: bool | None = None,
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosRun:
    current_pd = to_pandas(current)
    benchmark_pd = to_pandas(benchmark)
    previous_pd = to_pandas(previous_window) if previous_window is not None else None
    resolved_temporal_features = _resolve_temporal_features(temporal_features, time_column)
    _preflight_monitoring_report(
        current=current_pd,
        benchmark=benchmark_pd,
        previous_window=previous_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        protected_attribute=protected_attribute,
        temporal_features=resolved_temporal_features,
        include_performance=include_performance,
        include_bias=include_bias,
    )
    return LumosRun(
        run_type="monitoring",
        results={},
        metadata={
            "report_name": report_name,
            "skipped_reports": {},
            "sample_size": sample_size,
            "experiment_name": experiment_name,
        },
    )
