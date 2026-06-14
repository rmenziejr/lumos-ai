"""Dataset profiling helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lumosai.artifacts import artifact_workspace
from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.mlflow import mlflow_run, resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.settings import settings

ProfileReport: Any | None = None


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
    time_column: str | None = None,
    freq: str = "M",
    sample_size: int | None = None,
    min_per_period: int = 1,
    minimal: bool | None = None,
    log_analysis: bool | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    frame = to_pandas(df)
    if time_column is None:
        profiled = frame
        resolved_minimal = settings.data.profile_minimal_default if minimal is None else minimal
        sampling_summary: dict[str, Any] = {"mode": "full"}
    else:
        rows_per_period = 1000 if sample_size is None else sample_size
        profiled = temporal_sample(frame, time_column, freq, rows_per_period, min_per_period)
        resolved_minimal = False if minimal is None else minimal
        sampling_summary = {
            "mode": "temporal",
            "time_column": time_column,
            "freq": freq,
            "sample_size_per_period": rows_per_period,
            "sampled_rows": len(profiled),
        }

    report = _profile_report_class()(profiled, minimal=resolved_minimal)
    should_log_analysis = settings.data.log_analysis if log_analysis is None else log_analysis
    artifacts: dict[str, Any] = {}
    if should_log_analysis:
        logging_requested = resolve_experiment_name(experiment_name) is not None
        keep_local = not logging_requested or not settings.mlflow.log_artifacts
        with artifact_workspace(keep_local=keep_local) as workspace:
            html_path = workspace / "profile.html"
            report.to_file(html_path)
            if keep_local:
                artifacts["html"] = str(html_path)
            else:
                artifacts["html"] = {"mlflow_artifact_path": "profile/profile.html"}
            result = LumosResult(
                summary={
                    "rows": len(profiled),
                    "columns": list(profiled.columns),
                    "sampling": sampling_summary,
                },
                artifacts=artifacts,
                report=report,
                metadata={"report_type": "profile", "minimal": resolved_minimal},
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
        metadata={"report_type": "profile", "minimal": resolved_minimal},
    )
    return _log_profile_result(result, html_path=None, experiment_name=experiment_name)
