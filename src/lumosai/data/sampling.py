"""Representative sample builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.profiling import temporal_sample
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.mlflow import log_sample
from lumosai.results import LumosResult
from lumosai.schema import select_analysis_frame, validate_categorical_columns
from lumosai.settings import settings

SampleRole = Literal["train_benchmark", "holdout", "monitoring_window"]
SampleStrategy = Literal["auto", "random", "stratified", "temporal_recent", "temporal_bucket"]
ResolvedStrategy = Literal["random", "stratified", "temporal_recent", "temporal_bucket"]


def _dedupe_preserve_order(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    columns: list[str] = []
    for value in values:
        if value is not None and value not in seen:
            columns.append(value)
            seen.add(value)
    return columns


def _required_columns_for_role(
    *,
    role: SampleRole,
    target: str | None,
    prediction: str | None,
    time_column: str | None,
) -> list[str]:
    if role == "train_benchmark":
        return [target] if target is not None else []
    if role == "holdout":
        return _dedupe_preserve_order([time_column, target, prediction])
    return [time_column] if time_column is not None else []


def _exclude_train_benchmark_temporal_columns(
    columns: list[str],
    *,
    time_column: str | None,
    temporal_columns: list[str] | None,
) -> list[str]:
    excluded = {column for column in [time_column, *(temporal_columns or [])] if column}
    selected = [column for column in columns if column not in excluded]
    if not selected:
        msg = "train_benchmark sample must include at least one non-temporal column"
        raise LumosValidationError(msg)
    return selected


def _select_sample_columns(
    frame: pd.DataFrame,
    *,
    role: SampleRole,
    target: str | None,
    prediction: str | None,
    feature_columns: list[str] | None,
    time_column: str | None,
    temporal_columns: list[str] | None,
) -> pd.DataFrame:
    required_columns = _required_columns_for_role(
        role=role,
        target=target,
        prediction=prediction,
        time_column=time_column,
    )
    if prediction is not None and role != "train_benchmark":
        require_columns(frame, [prediction])
    if role == "train_benchmark" and time_column is not None:
        require_columns(frame, [time_column])
    if temporal_columns is not None:
        require_columns(frame, temporal_columns)

    if feature_columns is not None:
        if role == "train_benchmark":
            columns = _dedupe_preserve_order([target, *feature_columns])
            require_columns(frame, columns)
            columns = _exclude_train_benchmark_temporal_columns(
                columns,
                time_column=time_column,
                temporal_columns=temporal_columns,
            )
        elif role == "holdout":
            columns = _dedupe_preserve_order([time_column, target, prediction, *feature_columns])
            require_columns(frame, columns)
        else:
            columns = _dedupe_preserve_order([time_column, *feature_columns])
            require_columns(frame, columns)
        return frame[columns].copy()

    analysis_frame = select_analysis_frame(
        frame,
        target=target if role != "monitoring_window" else None,
        feature_columns=None,
        required_columns=required_columns,
    )
    if role == "train_benchmark":
        columns = _exclude_train_benchmark_temporal_columns(
            list(analysis_frame.columns),
            time_column=time_column,
            temporal_columns=temporal_columns,
        )
        return analysis_frame[columns].copy()
    if role == "holdout":
        ordered = _dedupe_preserve_order([time_column, target, prediction])
        remaining = [column for column in analysis_frame.columns if column not in ordered]
        return analysis_frame[[*ordered, *remaining]].copy()
    if time_column is not None:
        ordered = [time_column]
        remaining = [column for column in analysis_frame.columns if column != time_column]
        return analysis_frame[[*ordered, *remaining]].copy()
    return analysis_frame.copy()


def _resolve_strategy(
    strategy: SampleStrategy,
    *,
    role: SampleRole,
    target: str | None,
    stratify_by: str | list[str] | None,
    time_column: str | None,
) -> ResolvedStrategy:
    if strategy != "auto":
        return strategy
    if role in {"holdout", "monitoring_window"} and time_column is not None:
        return "temporal_recent"
    if role == "train_benchmark" and (target is not None or stratify_by is not None):
        return "stratified"
    return "random"


def _random_sample(frame: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _temporal_recent_sample(
    frame: pd.DataFrame, sample_size: int | None, time_column: str
) -> pd.DataFrame:
    require_columns(frame, [time_column])
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    if working[time_column].isna().any():
        msg = f"time_column contains null or invalid timestamps: {time_column}"
        raise LumosValidationError(msg)
    sorted_frame = working.sort_values(time_column, ascending=True)
    if sample_size is None or sample_size >= len(sorted_frame):
        return sorted_frame.reset_index(drop=True)
    return sorted_frame.tail(sample_size).reset_index(drop=True)


def _stratified_sample(
    frame: pd.DataFrame,
    *,
    sample_size: int | None,
    stratify_by: str | list[str] | None,
    target: str | None,
    random_state: int,
) -> pd.DataFrame:
    strata_columns = [stratify_by] if isinstance(stratify_by, str) else stratify_by
    if strata_columns is None:
        strata_columns = [target] if target is not None else None
    if not strata_columns:
        msg = "stratified sampling requires target or stratify_by"
        raise LumosValidationError(msg)
    require_columns(frame, strata_columns)
    if sample_size is None or sample_size >= len(frame):
        return frame.copy().reset_index(drop=True)

    grouped = frame.groupby(strata_columns, dropna=False, sort=False)
    sampled_indices: list[Any] = []
    for _, group in grouped:
        proportion = len(group) / len(frame)
        n = min(len(group), max(1, round(sample_size * proportion)))
        sampled = group.sample(n=n, random_state=random_state)
        sampled_indices.extend(sampled.index.tolist())

    if len(sampled_indices) > sample_size:
        sampled_indices = (
            frame.loc[sampled_indices]
            .sample(n=sample_size, random_state=random_state)
            .index.tolist()
        )
    elif len(sampled_indices) < sample_size:
        remaining = frame.drop(index=sampled_indices)
        if not remaining.empty:
            top_up_n = min(sample_size - len(sampled_indices), len(remaining))
            top_up = remaining.sample(n=top_up_n, random_state=random_state)
            sampled_indices.extend(top_up.index.tolist())

    return (
        frame.loc[sampled_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
    )


def _schema_summary(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _digest_frame(frame: pd.DataFrame) -> str:
    stable_frame = frame.apply(
        lambda column: column.map(lambda value: json.dumps(value, sort_keys=True, default=str))
    )
    hashed = pd.util.hash_pandas_object(stable_frame, index=False).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _time_range(frame: pd.DataFrame, time_column: str) -> dict[str, str | None]:
    values = pd.to_datetime(frame[time_column], errors="coerce")
    if values.empty or values.isna().all():
        return {"min": None, "max": None}
    return {
        "min": values.min().isoformat(),
        "max": values.max().isoformat(),
    }


def build_sample(
    data: Any,
    *,
    role: Literal["train_benchmark", "holdout", "monitoring_window"],
    sample_size: int | None = None,
    strategy: Literal[
        "auto", "random", "stratified", "temporal_recent", "temporal_bucket"
    ] = "auto",
    target: str | None = None,
    prediction: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    time_column: str | None = None,
    temporal_columns: list[str] | None = None,
    stratify_by: str | list[str] | None = None,
    random_state: int = 42,
    artifact_path: str | Path | None = None,
    experiment_name: str | None = None,
    log_metadata: bool | None = None,
    log_artifact: bool | None = None,
) -> LumosResult:
    frame = to_pandas(data)
    if role not in {"train_benchmark", "holdout", "monitoring_window"}:
        msg = "role must be one of: train_benchmark, holdout, monitoring_window"
        raise LumosValidationError(msg)
    if strategy not in {"auto", "random", "stratified", "temporal_recent", "temporal_bucket"}:
        msg = "strategy must be one of: auto, random, stratified, temporal_recent, temporal_bucket"
        raise LumosValidationError(msg)
    if sample_size is not None and sample_size < 1:
        msg = "sample_size must be greater than zero"
        raise LumosValidationError(msg)
    if role == "holdout" and prediction is not None and target is None:
        msg = "holdout samples require target when prediction is provided"
        raise LumosValidationError(msg)

    selected = _select_sample_columns(
        frame,
        role=role,
        target=target,
        prediction=prediction,
        feature_columns=feature_columns,
        time_column=time_column,
        temporal_columns=temporal_columns,
    ).reset_index(drop=True)
    selected_categorical_columns = validate_categorical_columns(
        frame,
        categorical_columns=categorical_columns,
        analysis_columns=selected.columns,
    )
    resolved_strategy = _resolve_strategy(
        strategy,
        role=role,
        target=target,
        stratify_by=stratify_by,
        time_column=time_column,
    )

    if resolved_strategy == "random":
        sampled = _random_sample(selected, sample_size, random_state)
    elif resolved_strategy == "temporal_recent":
        if time_column is None:
            msg = "temporal_recent sampling requires time_column"
            raise LumosValidationError(msg)
        sampled = _temporal_recent_sample(selected, sample_size, time_column)
    elif resolved_strategy == "temporal_bucket":
        if time_column is None:
            msg = "temporal_bucket sampling requires time_column"
            raise LumosValidationError(msg)
        rows_per_period = settings.data.default_sample_size if sample_size is None else sample_size
        sampled = temporal_sample(selected, time_column, sample_size=rows_per_period)
    else:
        sampled = _stratified_sample(
            selected,
            sample_size=sample_size,
            stratify_by=stratify_by,
            target=target,
            random_state=random_state,
        )

    summary: dict[str, Any] = {
        "role": role,
        "strategy": resolved_strategy,
        "source_rows": len(frame),
        "sample_rows": len(sampled),
        "columns": list(sampled.columns),
        "categorical_columns": selected_categorical_columns,
        "schema": _schema_summary(sampled),
        "digest": _digest_frame(sampled),
    }
    if time_column is not None and time_column in sampled.columns:
        summary["time_range"] = _time_range(sampled, time_column)

    result = LumosResult(
        summary=summary,
        artifacts={"sample": sampled},
        metadata={"report_type": "sample", "sample_role": role},
    )
    return log_sample(
        result,
        artifact_path=artifact_path,
        experiment_name=experiment_name,
        log_metadata=log_metadata,
        log_artifact=log_artifact,
    )
