from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lumosai.settings import Settings, settings


def require_mlflow() -> Any:
    from lumosai.mlflow import require_mlflow as _require_mlflow

    return _require_mlflow()


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


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return stem or "report"


def local_html_artifact_path(
    workspace: Path,
    default_filename: str,
    *,
    report_name: str | None = None,
) -> Path:
    """Return a local HTML path that will not overwrite an existing report."""
    default_path = Path(default_filename)
    suffix = default_path.suffix or ".html"
    default_stem = default_path.stem or "report"
    stem = _safe_artifact_stem(report_name) if report_name else default_stem
    candidate = workspace / f"{stem}{suffix}"
    index = 2
    while candidate.exists():
        candidate = workspace / f"{stem}-{index}{suffix}"
        index += 1
    return candidate


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
    artifact = {"mlflow_artifact_path": f"{artifact_path}/{html_path.name}"}
    if loaded_settings.artifacts.cache_mlflow_html:
        cache_dir = loaded_settings.artifacts.display_cache_dir / artifact_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / html_path.name
        if html_path.resolve() != cached_path.resolve():
            shutil.copyfile(html_path, cached_path)
        artifact["local_path"] = str(cached_path)
    return {"html": artifact}, keep_local


def log_result_with_html_artifact(
    result: Any,
    *,
    html_path: Path | None,
    artifact_path: str,
    experiment_name: str | None,
    loaded_settings: Settings = settings,
    log_dict: bool | None = None,
    mlflow_step: int | None = None,
) -> Any:
    """Log a Lumos result and optional HTML artifact in one MLflow run."""
    from lumosai.exceptions import LumosConfigurationError
    from lumosai.mlflow import configure_mlflow, resolve_experiment_name

    if mlflow_step is not None:
        result.metadata["mlflow_step"] = mlflow_step

    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        result.metadata["logged_to_mlflow"] = False
        return result

    should_log_dict = loaded_settings.mlflow.log_dicts if log_dict is None else log_dict
    mlflow = require_mlflow()
    configure_mlflow(mlflow, loaded_settings)
    mlflow.set_experiment(resolved)

    active_run = mlflow.active_run()
    context = nullcontext(active_run) if active_run is not None else None
    if context is None:
        if loaded_settings.mlflow.run_mode == "require_active":
            msg = "MLflow logging requested but no active run exists and run_mode='require_active'"
            raise LumosConfigurationError(msg)
        context = mlflow.start_run()

    with context as run:
        result.metadata["logged_to_mlflow"] = True
        result.metadata["mlflow_run_id"] = run.info.run_id
        if result.metrics:
            mlflow.log_metrics(result.metrics, step=mlflow_step)
        if html_path is not None and loaded_settings.mlflow.log_artifacts:
            mlflow.log_artifact(str(html_path), artifact_path=artifact_path)
        if should_log_dict:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
    return result
