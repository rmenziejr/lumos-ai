from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

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
