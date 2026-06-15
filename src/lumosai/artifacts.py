from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lumosai.settings import Settings, settings


@contextmanager
def artifact_workspace(
    loaded_settings: Settings = settings,
    *,
    keep_local: bool | None = None,
) -> Iterator[Path]:
    """Yield a directory for artifacts and clean it up unless local retention is enabled."""
    resolved_keep_local = loaded_settings.artifacts.keep_local if keep_local is None else keep_local
    if resolved_keep_local:
        directory = loaded_settings.artifacts.local_dir or Path("lumosai-artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        yield directory
        return

    with TemporaryDirectory(prefix="lumosai-") as temp_dir:
        yield Path(temp_dir)


def should_keep_html_artifact(
    *,
    experiment_name: str | None,
    loaded_settings: Settings = settings,
) -> bool:
    """Return whether report HTML should remain available as a local file."""
    from lumosai.mlflow import resolve_experiment_name

    logging_requested = resolve_experiment_name(experiment_name, loaded_settings) is not None
    return not logging_requested or not loaded_settings.mlflow.log_artifacts


def html_artifact_metadata(
    html_path: Path,
    *,
    artifact_path: str,
    experiment_name: str | None,
    loaded_settings: Settings = settings,
) -> tuple[dict[str, Any], bool]:
    """Return result artifact metadata and whether the local file should be retained."""
    keep_local = should_keep_html_artifact(
        experiment_name=experiment_name,
        loaded_settings=loaded_settings,
    )
    if keep_local:
        return {"html": str(html_path)}, keep_local
    return {"html": {"mlflow_artifact_path": f"{artifact_path}/{html_path.name}"}}, keep_local


def log_result_with_html_artifact(
    result: Any,
    *,
    html_path: Path | None,
    artifact_path: str,
    experiment_name: str | None,
    loaded_settings: Settings = settings,
) -> Any:
    """Log a Lumos result and optional HTML artifact in one MLflow run."""
    from lumosai.mlflow import mlflow_run, resolve_experiment_name

    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        result.metadata["logged_to_mlflow"] = False
        return result

    with mlflow_run(resolved, loaded_settings) as (mlflow, run_id):
        if mlflow is None:
            result.metadata["logged_to_mlflow"] = False
            return result
        result.metadata["logged_to_mlflow"] = True
        result.metadata["mlflow_run_id"] = run_id
        if result.metrics:
            mlflow.log_metrics(result.metrics)
        if html_path is not None and loaded_settings.mlflow.log_artifacts:
            mlflow.log_artifact(str(html_path), artifact_path=artifact_path)
        if loaded_settings.mlflow.log_dicts:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
    return result
