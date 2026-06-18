from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lumosai.results import LumosResult


def test_display_report_prefers_native_notebook_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []

    class NativeReport:
        def to_notebook_iframe(self) -> str:
            return "native iframe"

    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(LumosResult(report=NativeReport()))

    assert returned == "native iframe"
    assert displayed == ["native iframe"]


def test_display_report_treats_native_show_as_displayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    show_calls: list[str] = []

    class NativeReport:
        def show(self) -> None:
            show_calls.append("shown")

    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(
        LumosResult(report=NativeReport(), artifacts={"html": {"remote": "report.html"}})
    )

    assert returned is None
    assert show_calls == ["shown"]
    assert displayed == []


def test_display_report_uses_iframe_for_local_html_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body>report</body></html>", encoding="utf-8")

    class FakeIFrame:
        def __init__(self, src: str, width: str | int, height: int) -> None:
            self.src = src
            self.width = width
            self.height = height

    monkeypatch.setattr(notebook, "_display", displayed.append)
    monkeypatch.setattr(
        notebook,
        "_iframe",
        lambda src, *, width, height: FakeIFrame(src=src, width=width, height=height),
    )

    returned = notebook.display_report(
        LumosResult(artifacts={"html": str(html_path)}),
        title="Report",
        height=640,
    )

    assert returned is displayed[-1]
    assert returned.src == str(html_path)
    assert returned.height == 640


def test_display_report_falls_back_to_iframe_when_native_display_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body>report</body></html>", encoding="utf-8")

    class NativeReport:
        def to_notebook_iframe(self) -> object:
            raise ModuleNotFoundError("No module named 'ipywidgets'")

    class FakeIFrame:
        def __init__(self, src: str, width: str | int, height: int) -> None:
            self.src = src
            self.width = width
            self.height = height

    monkeypatch.setattr(notebook, "_display", displayed.append)
    monkeypatch.setattr(
        notebook,
        "_iframe",
        lambda src, *, width, height: FakeIFrame(src=src, width=width, height=height),
    )

    returned = notebook.display_report(
        LumosResult(report=NativeReport(), artifacts={"html": str(html_path)})
    )

    assert returned is displayed[-1]
    assert returned.src == str(html_path)


def test_display_report_displays_artifact_metadata_when_no_local_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    artifact = {"mlflow_artifact_path": "profile/profile.html"}
    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(LumosResult(artifacts={"html": artifact}))

    assert returned == artifact
    assert displayed == [artifact]
