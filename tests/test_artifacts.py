from __future__ import annotations

from pathlib import Path

from lumosai.artifacts import artifact_workspace
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
