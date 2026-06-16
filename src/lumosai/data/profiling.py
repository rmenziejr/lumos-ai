"""Dataset profiling helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lumosai.artifacts import artifact_workspace, local_html_artifact_path
from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.mlflow import mlflow_run, resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.schema import (
    filter_supported_kwargs,
    select_analysis_frame,
    validate_categorical_columns,
)
from lumosai.settings import settings

ProfileReport: Any | None = None
YDATA_KWARGS = {"explorative", "dark_mode", "config_file", "vars", "sort", "sensitive"}
YDATA_REJECTED_KWARGS = {"title", "minimal"}


def _profile_report_class() -> Any:
    global ProfileReport
    if ProfileReport is not None:
        return ProfileReport
    try:
        from ydata_profiling import (  # type: ignore[import-untyped]
            ProfileReport as YDataProfileReport,
        )
    except ImportError as exc:
        msg = "profiling requires ydata-profiling and its runtime dependencies"
        raise LumosOptionalDependencyError(msg) from exc
    ProfileReport = YDataProfileReport
    return ProfileReport


def temporal_sample(
    df: pd.DataFrame,
    time_column: str,
    freq: str = "M",
    sample_size: int = 1000,
    min_per_period: int = 1,
) -> pd.DataFrame:
    """Sample rows from each time bucket while preserving temporal coverage."""

    require_columns(df, [time_column])
    if sample_size < 1:
        msg = "sample_size must be greater than zero"
        raise LumosValidationError(msg)
    if min_per_period < 1:
        msg = "min_per_period must be greater than zero"
        raise LumosValidationError(msg)

    working = df.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    if working[time_column].isna().any():
        msg = f"time_column contains null or invalid timestamps: {time_column}"
        raise LumosValidationError(msg)
    periods = working[time_column].dt.to_period(freq)

    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        n = min(len(group), max(sample_size, min_per_period))
        return group.sample(n=n, random_state=42)

    return working.groupby(periods, group_keys=False).apply(sample_group).reset_index(drop=True)


def _log_profile_result(
    result: LumosResult,
    *,
    html_path: str | None,
    experiment_name: str | None,
) -> LumosResult:
    resolved = resolve_experiment_name(experiment_name)
    if resolved is None:
        result.metadata["logged_to_mlflow"] = False
        return result

    with mlflow_run(resolved) as (mlflow, run_id):
        if mlflow is None:
            result.metadata["logged_to_mlflow"] = False
            return result
        result.metadata["logged_to_mlflow"] = True
        result.metadata["mlflow_run_id"] = run_id
        if html_path is not None and settings.mlflow.log_artifacts:
            mlflow.log_artifact(html_path, artifact_path="profile")
        if settings.mlflow.log_dicts:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
    return result


def profile(
    df: Any,
    target: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    time_column: str | None = None,
    freq: str = "M",
    sample_size: int | None = None,
    min_per_period: int = 1,
    minimal: bool | None = None,
    log_analysis: bool | None = None,
    report_name: str | None = None,
    ydata_kwargs: dict[str, Any] | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    """Create a ydata-profiling report and return a structured result.

    MLflow logging is enabled when `experiment_name` is provided or
    `settings.mlflow.default_experiment_name` is set. Passing
    `log_analysis=False` disables profile artifact generation and MLflow logging
    for this call. With no experiment configured, generated artifacts are
    retained locally according to artifact settings.
    """

    frame = to_pandas(df)
    analysis_frame = select_analysis_frame(
        frame,
        target=target,
        feature_columns=feature_columns,
        required_columns=[time_column] if time_column is not None else None,
    )
    selected_categorical_columns = validate_categorical_columns(
        analysis_frame,
        categorical_columns=categorical_columns,
        analysis_columns=analysis_frame.columns,
    )
    profile_kwargs = filter_supported_kwargs(
        ydata_kwargs,
        allowed=YDATA_KWARGS,
        rejected=YDATA_REJECTED_KWARGS,
        parameter_name="ydata_kwargs",
    )
    if report_name is not None:
        profile_kwargs["title"] = report_name

    if time_column is None:
        profiled = analysis_frame
        resolved_minimal = settings.data.profile_minimal_default if minimal is None else minimal
        sampling_summary: dict[str, Any] = {"mode": "full"}
    else:
        rows_per_period = 1000 if sample_size is None else sample_size
        profiled = temporal_sample(
            analysis_frame,
            time_column,
            freq,
            rows_per_period,
            min_per_period,
        )
        resolved_minimal = False if minimal is None else minimal
        sampling_summary = {
            "mode": "temporal",
            "time_column": time_column,
            "freq": freq,
            "sample_size_per_period": rows_per_period,
            "sampled_rows": len(profiled),
        }

    report = _profile_report_class()(profiled, minimal=resolved_minimal, **profile_kwargs)
    should_log_analysis = settings.data.log_analysis if log_analysis is None else log_analysis
    artifacts: dict[str, Any] = {}
    metadata: dict[str, Any] = {"report_type": "profile", "minimal": resolved_minimal}
    if report_name is not None:
        metadata["report_name"] = report_name
    if target is not None:
        metadata["target"] = target
    if feature_columns is not None:
        metadata["feature_columns"] = list(feature_columns)
    if selected_categorical_columns:
        metadata["categorical_columns"] = selected_categorical_columns
    if should_log_analysis:
        logging_requested = resolve_experiment_name(experiment_name) is not None
        keep_local = not logging_requested or not settings.mlflow.log_artifacts
        with artifact_workspace(keep_local=keep_local) as workspace:
            html_path = local_html_artifact_path(
                workspace,
                "profile.html",
                report_name=report_name,
            )
            report.to_file(html_path)
            if keep_local:
                artifacts["html"] = str(html_path)
            else:
                artifacts["html"] = {
                    "mlflow_artifact_path": f"profile/{html_path.name}"
                }
            result = LumosResult(
                summary={
                    "rows": len(profiled),
                    "columns": list(profiled.columns),
                    "sampling": sampling_summary,
                },
                artifacts=artifacts,
                report=report,
                metadata=metadata,
            )
            return _log_profile_result(
                result,
                html_path=str(html_path),
                experiment_name=experiment_name,
            )

    result = LumosResult(
        summary={
            "rows": len(profiled),
            "columns": list(profiled.columns),
            "sampling": sampling_summary,
        },
        artifacts=artifacts,
        report=report,
        metadata=metadata,
    )
    result.metadata["logged_to_mlflow"] = False
    return result
