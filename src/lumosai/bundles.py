from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import pandas as pd

from lumosai.data.drift import drift_report
from lumosai.data.ingest import to_pandas
from lumosai.data.profiling import profile
from lumosai.data.sampling import build_sample
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.mlflow import log_run, mlflow_run, resolve_experiment_name
from lumosai.model.bias import ProtectedAttribute, bias_report
from lumosai.model.importance import feature_importance
from lumosai.model.performance import performance_report
from lumosai.results import LumosResult, LumosRun
from lumosai.settings import Settings, settings


@contextmanager
def _suppress_child_result_dict_logging() -> Iterator[None]:
    original = settings.mlflow.log_dicts
    settings.mlflow.log_dicts = False
    try:
        yield
    finally:
        settings.mlflow.log_dicts = original


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
    protected_attribute: str | ProtectedAttribute | None,
    include_bias: bool | None,
) -> bool:
    return include_bias is True or protected_attribute is not None


def _profile_expected(*, include_profile: bool | None) -> bool:
    return include_profile is True


def _feature_importance_expected(
    *,
    model: Any | None,
    include_feature_importance: bool | None,
    loaded_settings: Settings,
) -> bool:
    if include_feature_importance is not None:
        return include_feature_importance
    return model is not None and loaded_settings.bundles.include_feature_importance_in_training


def _protected_columns(
    protected_attribute: str | ProtectedAttribute,
) -> list[str]:
    if isinstance(protected_attribute, str):
        return [protected_attribute]
    if isinstance(protected_attribute, dict):
        return list(protected_attribute)
    return list(protected_attribute)


def _bias_protected_attribute(
    protected_attribute: str | ProtectedAttribute,
) -> ProtectedAttribute:
    if isinstance(protected_attribute, str):
        return [protected_attribute]
    return protected_attribute


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
    protected_attribute: str | ProtectedAttribute | None,
    temporal_features: list[str],
    include_performance: bool | None,
    include_bias: bool | None,
) -> None:
    drift_columns = list(feature_columns or current.columns)
    overlap = sorted(set(temporal_features).intersection(feature_columns or []))
    if overlap:
        msg = "temporal_features must not be included in feature_columns: "
        msg += ", ".join(overlap)
        raise LumosValidationError(msg)
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


def _preflight_training_report(
    *,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    prediction: str | None,
    model: Any | None,
    feature_columns: list[str] | None,
    protected_attribute: str | ProtectedAttribute | None,
    include_performance: bool | None,
    include_bias: bool | None,
    include_feature_importance: bool | None,
    loaded_settings: Settings,
) -> None:
    require_columns(train, [target])
    require_columns(holdout, [target])
    if feature_columns is not None:
        require_columns(train, feature_columns)
        require_columns(holdout, feature_columns)

    if _performance_expected(
        target=target,
        prediction=prediction,
        include_performance=include_performance,
    ):
        if prediction is None:
            msg = "training_report expected performance but prediction is missing"
            raise LumosValidationError(msg)
        require_columns(holdout, [target, prediction])

    if _bias_expected(protected_attribute=protected_attribute, include_bias=include_bias):
        if protected_attribute is None:
            msg = "training_report expected bias but protected_attribute is missing"
            raise LumosValidationError(msg)
        if prediction is None:
            msg = "training_report expected bias but prediction is missing"
            raise LumosValidationError(msg)
        require_columns(holdout, [target, prediction, *_protected_columns(protected_attribute)])

    if _feature_importance_expected(
        model=model,
        include_feature_importance=include_feature_importance,
        loaded_settings=loaded_settings,
    ):
        if model is None:
            msg = "training_report expected feature importance but model is missing"
            raise LumosValidationError(msg)
        if not feature_columns:
            msg = "training_report expected feature importance but feature_columns is missing"
            raise LumosValidationError(msg)


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
    protected_attribute: str | ProtectedAttribute | None = None,
    temporal_features: list[str] | None = None,
    time_column: str | None = None,
    sample_size: int | None = None,
    include_performance: bool | None = None,
    include_bias: bool | None = None,
    report_name: str | None = None,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> LumosRun:
    if not loaded_settings.bundles.fail_fast:
        msg = "monitoring_report currently requires fail_fast=True"
        raise LumosValidationError(msg)

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
    resolved_experiment = resolve_experiment_name(experiment_name, loaded_settings)
    run_context = (
        mlflow_run(resolved_experiment, loaded_settings)
        if resolved_experiment
        else nullcontext((None, None))
    )
    results: dict[str, LumosResult] = {}
    skipped_reports: dict[str, str] = {}

    with run_context:
        with _suppress_child_result_dict_logging():
            results["monitoring_window"] = build_sample(
                current_pd,
                role="monitoring_window",
                sample_size=sample_size,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                time_column=time_column,
                experiment_name=resolved_experiment,
            )
            results["drift_benchmark"] = drift_report(
                benchmark_pd,
                current_pd,
                temporal_features=resolved_temporal_features,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                comparison="benchmark",
                report_name=f"{report_name} Benchmark Drift" if report_name else None,
                experiment_name=resolved_experiment,
            )
            if previous_pd is not None and loaded_settings.bundles.include_previous_window_drift:
                results["drift_previous_window"] = drift_report(
                    previous_pd,
                    current_pd,
                    temporal_features=resolved_temporal_features,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    comparison="previous_window",
                    report_name=f"{report_name} Previous Window Drift" if report_name else None,
                    experiment_name=resolved_experiment,
                )
            elif previous_pd is not None:
                skipped_reports["drift_previous_window"] = (
                    "previous_window drift disabled by settings"
                )
            else:
                skipped_reports["drift_previous_window"] = "previous_window not provided"

            if _performance_expected(
                target=target,
                prediction=prediction,
                include_performance=include_performance,
            ):
                assert target is not None
                assert prediction is not None
                results["performance"] = performance_report(
                    current_pd,
                    target=target,
                    prediction=prediction,
                    prediction_score=prediction_score,
                    report_name=f"{report_name} Performance" if report_name else None,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    experiment_name=resolved_experiment,
                )
            else:
                skipped_reports["performance"] = "target and prediction not provided"

            if _bias_expected(
                protected_attribute=protected_attribute,
                include_bias=include_bias,
            ):
                assert target is not None
                assert prediction is not None
                assert protected_attribute is not None
                results["bias"] = bias_report(
                    current_pd,
                    target=target,
                    prediction=prediction,
                    protected_attribute=_bias_protected_attribute(protected_attribute),
                    prediction_score=prediction_score,
                    report_name=f"{report_name} Bias" if report_name else None,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    experiment_name=resolved_experiment,
                )
            else:
                skipped_reports["bias"] = "protected_attribute not provided"

        run = LumosRun(
            run_type="monitoring",
            results=results,
            metadata={
                "report_name": report_name,
                "skipped_reports": skipped_reports,
            },
        )
        log_run(
            run,
            experiment_name=resolved_experiment,
            loaded_settings=loaded_settings,
        )
        return run


def training_report(
    train: Any,
    holdout: Any,
    *,
    target: str,
    prediction: str | None = None,
    prediction_score: str | None = None,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    protected_attribute: str | ProtectedAttribute | None = None,
    time_column: str | None = None,
    sample_size: int | None = None,
    include_profile: bool | None = None,
    include_performance: bool | None = None,
    include_bias: bool | None = None,
    include_feature_importance: bool | None = None,
    report_name: str | None = None,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> LumosRun:
    if not loaded_settings.bundles.fail_fast:
        msg = "training_report currently requires fail_fast=True"
        raise LumosValidationError(msg)

    train_pd = to_pandas(train)
    holdout_pd = to_pandas(holdout)
    _preflight_training_report(
        train=train_pd,
        holdout=holdout_pd,
        target=target,
        prediction=prediction,
        model=model,
        feature_columns=feature_columns,
        protected_attribute=protected_attribute,
        include_performance=include_performance,
        include_bias=include_bias,
        include_feature_importance=include_feature_importance,
        loaded_settings=loaded_settings,
    )
    resolved_experiment = resolve_experiment_name(experiment_name, loaded_settings)
    run_context = (
        mlflow_run(resolved_experiment, loaded_settings)
        if resolved_experiment
        else nullcontext((None, None))
    )
    results: dict[str, LumosResult] = {}
    skipped_reports: dict[str, str] = {}

    with run_context:
        with _suppress_child_result_dict_logging():
            results["train_sample"] = build_sample(
                train_pd,
                role="train_benchmark",
                sample_size=sample_size,
                target=target,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                time_column=time_column,
                experiment_name=resolved_experiment,
            )
            results["holdout_sample"] = build_sample(
                holdout_pd,
                role="holdout",
                sample_size=sample_size,
                target=target,
                prediction=prediction,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                time_column=time_column,
                experiment_name=resolved_experiment,
            )

            if _profile_expected(include_profile=include_profile):
                results["profile"] = profile(
                    train_pd,
                    target=target,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    time_column=time_column,
                    sample_size=sample_size,
                    report_name=f"{report_name} Profile" if report_name else None,
                    experiment_name=resolved_experiment,
                )
            else:
                skipped_reports["profile"] = "include_profile not enabled"

            if _performance_expected(
                target=target,
                prediction=prediction,
                include_performance=include_performance,
            ):
                assert prediction is not None
                results["performance"] = performance_report(
                    holdout_pd,
                    target=target,
                    prediction=prediction,
                    prediction_score=prediction_score,
                    report_name=f"{report_name} Performance" if report_name else None,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    experiment_name=resolved_experiment,
                )
            else:
                skipped_reports["performance"] = "prediction not provided"

            if _bias_expected(
                protected_attribute=protected_attribute,
                include_bias=include_bias,
            ):
                assert prediction is not None
                assert protected_attribute is not None
                results["bias"] = bias_report(
                    holdout_pd,
                    target=target,
                    prediction=prediction,
                    protected_attribute=_bias_protected_attribute(protected_attribute),
                    prediction_score=prediction_score,
                    report_name=f"{report_name} Bias" if report_name else None,
                    feature_columns=feature_columns,
                    categorical_columns=categorical_columns,
                    experiment_name=resolved_experiment,
                )
            else:
                skipped_reports["bias"] = "protected_attribute not provided"

            if _feature_importance_expected(
                model=model,
                include_feature_importance=include_feature_importance,
                loaded_settings=loaded_settings,
            ):
                assert model is not None
                assert feature_columns is not None
                results["feature_importance"] = feature_importance(
                    model,
                    holdout_pd,
                    target=target,
                    feature_columns=feature_columns,
                    report_name=f"{report_name} Feature Importance" if report_name else None,
                    experiment_name=resolved_experiment,
                )
            elif include_feature_importance is False:
                skipped_reports["feature_importance"] = "include_feature_importance disabled"
            elif not loaded_settings.bundles.include_feature_importance_in_training:
                skipped_reports["feature_importance"] = "feature importance disabled by settings"
            else:
                skipped_reports["feature_importance"] = "model not provided"

        run = LumosRun(
            run_type="training",
            results=results,
            metadata={
                "report_name": report_name,
                "skipped_reports": skipped_reports,
            },
        )
        log_run(
            run,
            experiment_name=resolved_experiment,
            loaded_settings=loaded_settings,
        )
        return run
