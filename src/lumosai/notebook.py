"""Notebook display helpers for Lumos reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lumosai.exceptions import LumosOptionalDependencyError
from lumosai.results import LumosResult

_NATIVE_SIDE_EFFECT_METHODS = (
    "to_notebook_iframe",
    "show",
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


def _iframe(src: str, *, width: str | int, height: int) -> Any:
    try:
        from IPython.display import IFrame  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "notebook display requires IPython"
        raise LumosOptionalDependencyError(msg) from exc
    return IFrame(src=src, width=width, height=height)


def _native_display_object(report: Any) -> tuple[bool, Any | None]:
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


def display_report(
    result: LumosResult,
    title: str | None = None,
    *,
    width: str | int = "100%",
    height: int = 900,
) -> Any:
    """Display a Lumos result in a notebook and return the displayed object.

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
            return rendered
        if handled:
            return rendered

    html_artifact = result.artifacts.get("html")
    if isinstance(html_artifact, str) and Path(html_artifact).exists():
        rendered = _iframe(html_artifact, width=width, height=height)
        _display(rendered)
        return rendered

    if html_artifact is not None:
        _display(html_artifact)
        return html_artifact

    _display(result)
    return result
