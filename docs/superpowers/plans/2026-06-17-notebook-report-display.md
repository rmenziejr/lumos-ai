# Notebook Report Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a notebook display helper that preserves native interactive report behavior and falls back to iframe-rendered local HTML artifacts.

**Architecture:** Add a small notebook-specific module that depends on IPython only at call time. Keep report generation responsible for retaining native report objects, and keep serialization unchanged.

**Tech Stack:** Python 3.12, IPython display objects, ydata-profiling native display methods, Evidently report/run objects, pytest, nbformat for notebook updates.

---

### Task 1: Native Display Helper

**Files:**
- Create: `src/lumosai/notebook.py`
- Modify: `src/lumosai/__init__.py`
- Test: `tests/test_notebook.py`

- [ ] Write tests for native display priority, iframe fallback, and metadata fallback.
- [ ] Implement `display_report(result, title=None, width="100%", height=900)`.
- [ ] Export `display_report` from the top-level package.
- [ ] Run `uv run pytest tests/test_notebook.py -v`.

### Task 2: Evidently Display Object Retention

**Files:**
- Modify: `src/lumosai/data/drift.py`
- Test: `tests/data/test_drift.py`

- [ ] Add a test proving `drift_report()` stores `run_result` in `LumosResult.report` when Evidently returns one.
- [ ] Change `drift_report()` to set `report=run_result or report`.
- [ ] Run `uv run pytest tests/data/test_drift.py -v`.

### Task 3: Notebook and Docs

**Files:**
- Modify: `examples/notebooks/pima_diabetes_walkthrough.ipynb`
- Modify: `docs/api.md`
- Test: `tests/test_public_api.py`

- [ ] Update the Pima notebook import and helper calls to use `display_report()`.
- [ ] Document `display_report()` in the API docs.
- [ ] Add public API coverage.
- [ ] Run `uv run pytest tests/test_public_api.py tests/test_notebook.py -v`.

### Task 4: Verification

**Files:**
- All changed files

- [ ] Run `uv run ruff check src tests docs`.
- [ ] Run `uv run pytest -v`.
- [ ] Run `uv run mkdocs build --strict`.
- [ ] Commit and push when all checks pass.

