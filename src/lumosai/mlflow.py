from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

from lumosai.exceptions import LumosConfigurationError, LumosOptionalDependencyError
from lumosai.results import LumosResult
from lumosai.settings import Settings, settings


def resolve_experiment_name(
    experiment_name: str | None,
    loaded_settings: Settings = settings,
) -> str | None:
    return experiment_name or loaded_settings.mlflow.default_experiment_name


def require_mlflow() -> Any:
    try:
        mlflow = import_module("mlflow")
    except ImportError as exc:
        msg = "mlflow logging requires the optional dependency: pip install lumosai[mlflow]"
        raise LumosOptionalDependencyError(msg) from exc
    return mlflow


def configure_mlflow(mlflow: Any, loaded_settings: Settings = settings) -> None:
    if loaded_settings.mlflow.username is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = loaded_settings.mlflow.username
    if loaded_settings.mlflow.password is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = loaded_settings.mlflow.password
    if loaded_settings.mlflow.tracking_uri is not None:
        mlflow.set_tracking_uri(loaded_settings.mlflow.tracking_uri)


@contextmanager
def mlflow_run(
    experiment_name: str | None,
    loaded_settings: Settings = settings,
) -> Iterator[tuple[Any, str | None]]:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        yield None, None
        return

    mlflow = require_mlflow()
    configure_mlflow(mlflow, loaded_settings)
    mlflow.set_experiment(resolved)

    active_run = mlflow.active_run()
    if active_run is not None:
        yield mlflow, active_run.info.run_id
        return

    if loaded_settings.mlflow.run_mode == "require_active":
        msg = "MLflow logging requested but no active run exists and run_mode='require_active'"
        raise LumosConfigurationError(msg)

    with mlflow.start_run() as run:
        yield mlflow, run.info.run_id


def log_result(
    result: LumosResult,
    *,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> LumosResult:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        result.metadata["logged_to_mlflow"] = False
        return result

    context = mlflow_run(resolved, loaded_settings) if resolved else nullcontext((None, None))
    with context as (mlflow, run_id):
        if mlflow is None:
            result.metadata["logged_to_mlflow"] = False
            return result
        result.metadata["logged_to_mlflow"] = True
        result.metadata["mlflow_run_id"] = run_id
        if result.metrics:
            mlflow.log_metrics(result.metrics)
        if loaded_settings.mlflow.log_dicts:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
    return result


def log_sample(
    result: LumosResult,
    *,
    artifact_path: str | Path | None = None,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
    log_metadata: bool | None = None,
    log_artifact: bool | None = None,
) -> LumosResult:
    sample = result.artifacts.get("sample")
    local_path = Path(artifact_path) if artifact_path is not None else None
    written_path: Path | None = None
    if isinstance(sample, pd.DataFrame) and local_path is not None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.suffix == ".csv":
            sample.to_csv(local_path, index=False)
        else:
            sample.to_parquet(local_path, index=False)
        written_path = local_path
        result.metadata["sample_artifact_path"] = str(local_path)

    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        result.metadata["logged_to_mlflow"] = False
        return result

    should_log_metadata = (
        loaded_settings.data.log_sample_metadata if log_metadata is None else log_metadata
    )
    should_log_artifact = (
        loaded_settings.data.log_sample_artifacts if log_artifact is None else log_artifact
    )

    with mlflow_run(resolved, loaded_settings) as (mlflow, run_id):
        if mlflow is None:
            result.metadata["logged_to_mlflow"] = False
            return result
        result.metadata["logged_to_mlflow"] = True
        result.metadata["mlflow_run_id"] = run_id
        if should_log_metadata:
            mlflow.log_dict(result.summary, "lumosai_sample_metadata.json")
        if should_log_artifact and written_path is not None:
            mlflow.log_artifact(str(written_path), artifact_path="samples")
    return result


def log_artifact_paths(
    paths: dict[str, Path],
    *,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> dict[str, str]:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None or not loaded_settings.mlflow.log_artifacts:
        return {name: str(path) for name, path in paths.items()}

    with mlflow_run(resolved, loaded_settings) as (mlflow, _run_id):
        if mlflow is None:
            return {name: str(path) for name, path in paths.items()}
        for name, path in paths.items():
            mlflow.log_artifact(str(path), artifact_path=name)
    return {name: str(path) for name, path in paths.items()}
