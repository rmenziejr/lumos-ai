# Report Artifact Display Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MLflow-logged HTML reports render locally via `display_report()` and improve first-party HTML report presentation.

**Architecture:** Add a shared artifact display-cache path for MLflow HTML uploads, so `result.artifacts["html"]` can include both `local_path` and `mlflow_artifact_path`. Update `display_report()` to render cached local HTML from dict metadata. Add a shared first-party report shell in `lumosai.model.plots` and keep ydata/Evidently native HTML unchanged.

**Tech Stack:** Python, pathlib/shutil, pytest, existing MLflow/artifact helpers, matplotlib-generated first-party HTML.

---

### Task 1: Cache MLflow HTML Artifacts

**Files:**
- Modify: `src/lumosai/settings.py`
- Modify: `src/lumosai/artifacts.py`
- Test: `tests/test_artifacts.py`

- [ ] Add artifact settings `cache_mlflow_html: bool = True` and `display_cache_dir: Path = Path(".lumosai-artifacts/display-cache")`.
- [ ] Add a failing test showing `html_artifact_metadata()` returns both `local_path` and `mlflow_artifact_path` when MLflow logging is requested.
- [ ] Implement cache copying inside `html_artifact_metadata()`.
- [ ] Run `uv run pytest tests/test_artifacts.py -v`.

### Task 2: Use Shared Artifact Logic For Profile

**Files:**
- Modify: `src/lumosai/data/profiling.py`
- Test: `tests/data/test_profiling.py`

- [ ] Add a failing test showing `profile(..., experiment_name=...)` returns `artifacts["html"]["local_path"]`.
- [ ] Replace profile's bespoke HTML artifact metadata with shared `html_artifact_metadata()`.
- [ ] Run `uv run pytest tests/data/test_profiling.py -v`.

### Task 3: Display Cached MLflow HTML

**Files:**
- Modify: `src/lumosai/notebook.py`
- Test: `tests/test_notebook.py`

- [ ] Add a failing test showing `display_report()` embeds `artifacts["html"]["local_path"]` when the artifact is a dict.
- [ ] Implement local path extraction from HTML artifact dicts.
- [ ] Run `uv run pytest tests/test_notebook.py -v`.

### Task 4: Improve First-Party HTML Shell

**Files:**
- Modify: `src/lumosai/model/plots.py`
- Test: existing report HTML tests in `tests/model/`.

- [ ] Refine `_html_document()` CSS/layout for a professional report shell.
- [ ] Keep all existing headings and plot content stable.
- [ ] Run `uv run pytest tests/model/test_performance.py tests/model/test_calibration.py tests/model/test_bias.py tests/model/test_importance.py tests/model/test_performance_drift.py -v`.

### Task 5: Docs And Full Verification

**Files:**
- Modify: `docs/api.md`

- [ ] Document cached local HTML behavior for `display_report()`.
- [ ] Run `uv run ruff check src tests docs`.
- [ ] Run `uv run pytest -v`.
- [ ] Run `uv run mkdocs build --strict`.
