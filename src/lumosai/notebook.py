"""Notebook display helpers for Lumos reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from lumosai.exceptions import LumosOptionalDependencyError
from lumosai.results import LumosResult

_NATIVE_SIDE_EFFECT_METHODS = (
    "to_notebook_iframe",
    "show",
    "report",
)
_NATIVE_RENDER_METHODS = (
    "_repr_html_",
    "as_html",
    "to_html",
    "html",
)


def _display(value: Any) -> None:
    try:
        from IPython.display import display  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "notebook display requires IPython"
        raise LumosOptionalDependencyError(msg) from exc
    display(value)


def _html(value: str) -> Any:
    try:
        from IPython.display import HTML  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "notebook display requires IPython"
        raise LumosOptionalDependencyError(msg) from exc
    return HTML(value)


def _html_iframe_srcdoc(html_path: Path, *, width: str | int, height: int) -> Any:
    escaped_html = escape(html_path.read_text(encoding="utf-8"), quote=True)
    escaped_width = escape(str(width), quote=True)
    return _html(
        "\n".join(
            [
                "<iframe",
                f'    width="{escaped_width}"',
                f'    height="{height}"',
                f'    srcdoc="{escaped_html}"',
                '    frameborder="0"',
                "    allowfullscreen",
                "></iframe>",
            ]
        )
    )


def _native_display_object(report: Any) -> tuple[bool, Any | None]:
    if callable(getattr(report, "_repr_html_", None)):
        return True, report

    for method_name in _NATIVE_SIDE_EFFECT_METHODS:
        method = getattr(report, method_name, None)
        if method is None or not callable(method):
            continue
        try:
            return True, method()
        except ImportError:
            continue
    for method_name in _NATIVE_RENDER_METHODS:
        method = getattr(report, method_name, None)
        if method is None or not callable(method):
            continue
        try:
            rendered = method()
        except ImportError:
            continue
        if rendered is not None:
            return True, rendered
    return False, None


def _html_artifact_local_path(html_artifact: Any) -> Path | None:
    if isinstance(html_artifact, str):
        path = Path(html_artifact)
        return path if path.exists() else None
    if isinstance(html_artifact, dict):
        local_path = html_artifact.get("local_path")
        if isinstance(local_path, str):
            path = Path(local_path)
            return path if path.exists() else None
    return None


def display_report(
    result: LumosResult,
    title: str | None = None,
    *,
    width: str | int = "100%",
    height: int = 900,
) -> Any:
    """Display a Lumos result in a notebook.

    Native backend report objects are preferred because libraries such as
    ydata-profiling can provide richer notebook display behavior than saved
    HTML embedded directly in an output cell.
    """

    if title is not None:
        _display(title)

    if result.report is not None:
        handled, rendered = _native_display_object(result.report)
        if handled and rendered is not None:
            _display(rendered)
            return None
        if handled:
            return None

    html_artifact = result.artifacts.get("html")
    html_path = _html_artifact_local_path(html_artifact)
    if html_path is not None:
        rendered = _html_iframe_srcdoc(html_path, width=width, height=height)
        _display(rendered)
        return None

    if html_artifact is not None:
        _display(html_artifact)
        return None

    _display(result)
    return None
