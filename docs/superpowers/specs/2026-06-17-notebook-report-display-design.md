# Notebook Report Display Design

Lumos should make notebook report display interactive when the underlying backend supports it. Saved HTML artifacts remain the durable output for MLflow, local sharing, and docs, but notebook users should not have to manually inspect artifact paths or inline full HTML documents.

## Contract

`LumosResult.report` is the native in-memory report object for interactive use. It is intentionally excluded from `LumosResult.to_dict()` and should not be treated as JSON-safe or persistent.

`LumosResult.artifacts["html"]` is the durable saved HTML artifact. It may be a local path or MLflow artifact metadata.

`lumosai.notebook.display_report(result, ...)` is the supported notebook display helper. It prefers native backend display behavior, then falls back to rendering a local HTML artifact in an iframe, then displays artifact metadata.

## Backend Behavior

For ydata-profiling, `profile()` already retains the native `ProfileReport` object. The display helper should call native notebook methods such as `to_notebook_iframe()` when available.

For Evidently drift reports, `drift_report()` should retain the object returned by the Evidently run when that object exists, because newer Evidently APIs may attach display/export behavior to the run result rather than the report definition.

For Lumos custom matplotlib HTML reports, there may be no native report object. Those reports should display via iframe from the saved local HTML artifact.

## Notebook UX

The Pima walkthrough should import and use `display_report()` instead of a local helper that inlines HTML with `IPython.display.HTML(Path(...).read_text())`.

The helper should fail softly outside notebooks with a clear optional dependency error only when IPython is unavailable and native display/fallback display cannot be used.

