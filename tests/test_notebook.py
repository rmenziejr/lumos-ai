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

    assert returned is None
    assert displayed == ["native iframe"]


def test_display_report_prefers_native_repr_object_over_rendered_html_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []

    class EvidentlySnapshot:
        def _repr_html_(self) -> str:
            return "<html>snapshot iframe</html>"

    snapshot = EvidentlySnapshot()
    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(LumosResult(report=snapshot))

    assert returned is None
    assert displayed == [snapshot]


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


def test_display_report_stops_after_side_effect_notebook_iframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    native_calls: list[str] = []

    class NativeReport:
        def to_notebook_iframe(self) -> None:
            native_calls.append("iframe")

        def to_html(self) -> str:
            native_calls.append("html")
            return "<html>fallback string</html>"

    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(
        LumosResult(report=NativeReport(), artifacts={"html": {"remote": "report.html"}})
    )

    assert returned is None
    assert native_calls == ["iframe"]
    assert displayed == []


def test_display_report_embeds_local_html_artifact_as_srcdoc_iframe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    html_path = tmp_path / "report.html"
    html_path.write_text("<html><body><h1>Report</h1></body></html>", encoding="utf-8")

    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(
        LumosResult(artifacts={"html": str(html_path)}),
        title="Report",
        height=640,
    )

    assert returned is None
    rendered = displayed[-1]
    assert rendered.__class__.__name__ == "HTML"
    assert (
        'srcdoc="&lt;html&gt;&lt;body&gt;&lt;h1&gt;Report&lt;/h1&gt;&lt;/body&gt;&lt;/html&gt;"'
        in rendered.data
    )
    assert 'height="640"' in rendered.data
    assert "src=" not in rendered.data


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

    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(
        LumosResult(report=NativeReport(), artifacts={"html": str(html_path)})
    )

    assert returned is None
    rendered = displayed[-1]
    assert rendered.__class__.__name__ == "HTML"
    assert "srcdoc=" in rendered.data


def test_display_report_displays_artifact_metadata_when_no_local_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumosai import notebook

    displayed: list[Any] = []
    artifact = {"mlflow_artifact_path": "profile/profile.html"}
    monkeypatch.setattr(notebook, "_display", displayed.append)

    returned = notebook.display_report(LumosResult(artifacts={"html": artifact}))

    assert returned is None
    assert displayed == [artifact]
