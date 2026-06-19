from __future__ import annotations

from pathlib import Path

from lumosai.artifacts import artifact_workspace, html_artifact_metadata
from lumosai.settings import Settings


def test_artifact_workspace_uses_temp_dir_when_not_keeping_local() -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = False

    with artifact_workspace(loaded_settings=loaded) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")
        assert path.exists()

    assert not path.exists()


def test_artifact_workspace_keeps_configured_local_dir(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = True
    loaded.artifacts.local_dir = tmp_path

    with artifact_workspace(loaded_settings=loaded) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")

    assert path.exists()


def test_artifact_workspace_can_force_local_retention(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = False
    loaded.artifacts.local_dir = tmp_path

    with artifact_workspace(loaded_settings=loaded, keep_local=True) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")

    assert path.exists()


def test_html_artifact_metadata_caches_mlflow_html_for_display(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.mlflow.default_experiment_name = "experiment"
    loaded.mlflow.log_artifacts = True
    loaded.artifacts.cache_mlflow_html = True
    loaded.artifacts.display_cache_dir = tmp_path / "display-cache"
    html_path = tmp_path / "source.html"
    html_path.write_text("<html>report</html>", encoding="utf-8")

    artifacts, keep_local = html_artifact_metadata(
        html_path,
        artifact_path="performance",
        experiment_name=None,
        loaded_settings=loaded,
    )

    assert keep_local is False
    assert artifacts["html"]["mlflow_artifact_path"] == "performance/source.html"
    cached_path = Path(artifacts["html"]["local_path"])
    assert cached_path.exists()
    assert cached_path.read_text(encoding="utf-8") == "<html>report</html>"
    assert cached_path.parent == loaded.artifacts.display_cache_dir / "performance"
