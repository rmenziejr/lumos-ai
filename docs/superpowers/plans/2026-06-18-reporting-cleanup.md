# Reporting Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up report display, importance summaries, drift HTML artifacts, drift metrics, and bias plot artifacts.

**Architecture:** Keep backend-native reports native where they exist: ydata and Evidently should display through their report objects. Lumos custom reports should render saved local HTML in notebooks. Drift remains an Evidently wrapper but exposes Evidently metric values plus Lumos important-feature alert metrics.

**Tech Stack:** Python, pandas, Evidently, ydata-profiling, matplotlib, pytest, uv.

---

### Task 1: Notebook Display Contract

**Files:**
- Modify: `src/lumosai/notebook.py`
- Test: `tests/test_notebook.py`
- Docs: `docs/api.md`

- [ ] Write tests that `display_report()` displays native reports or local HTML but returns `None`.
- [ ] Run `uv run pytest tests/test_notebook.py -v` and verify the updated tests fail.
- [ ] Update `display_report()` so it calls native display/render methods first, falls back to local HTML `srcdoc`, then artifact metadata/result, and always returns `None`.
- [ ] Update API docs to describe side-effect display behavior.
- [ ] Re-run `uv run pytest tests/test_notebook.py -v` and verify it passes.

### Task 2: Importance Summary Shape

**Files:**
- Modify: `src/lumosai/model/importance.py`
- Test: `tests/model/test_importance.py`
- Docs: `docs/api.md`

- [ ] Write tests that `feature_importance().summary` contains `methods` but not top-level `features`.
- [ ] Run `uv run pytest tests/model/test_importance.py -v` and verify the updated tests fail.
- [ ] Remove `summary["features"]` while preserving per-method feature rows and metrics.
- [ ] Re-run `uv run pytest tests/model/test_importance.py -v`.

### Task 3: Native Evidently Drift Artifacts and Metrics

**Files:**
- Modify: `src/lumosai/data/drift.py`
- Test: `tests/data/test_drift.py`
- Docs: `docs/api.md`, `docs/recipes/pima-diabetes.md`

- [ ] Write tests for native-only drift HTML export: when Evidently export works, save that HTML; when it does not, omit the HTML artifact and record a metadata warning.
- [ ] Write tests for Evidently metric extraction using metric names/config/value payloads, including per-column names such as `drift/benchmark/age/ks_p_value`.
- [ ] Run targeted drift tests and verify failures.
- [ ] Replace fallback custom drift HTML with native-only export behavior.
- [ ] Add Evidently metric extraction helpers that normalize metric and column names into stable metric paths.
- [ ] Preserve Lumos aggregate drift metrics and important-feature alert metrics.
- [ ] Re-run `uv run pytest tests/data/test_drift.py -v`.

### Task 4: Bias Plot Artifact

**Files:**
- Modify: `src/lumosai/model/bias.py`
- Modify: `src/lumosai/model/plots.py`
- Test: `tests/model/test_bias.py`
- Docs: `docs/api.md`

- [ ] Write tests that `bias_report(..., include_plots=True)` creates a local HTML artifact with group metrics and flagged comparisons.
- [ ] Run `uv run pytest tests/model/test_bias.py -v` and verify the updated tests fail.
- [ ] Add `bias_html()` to `plots.py` with group-size, metric comparison, and flagged comparison sections.
- [ ] Add `include_plots` to `bias_report()` and save/log the HTML artifact using existing artifact helpers.
- [ ] Re-run `uv run pytest tests/model/test_bias.py -v`.

### Task 5: Notebook and Full Verification

**Files:**
- Modify: `examples/notebooks/pima_diabetes_walkthrough.ipynb`
- Verify: package tests and docs

- [ ] Re-run the Pima notebook so outputs use the updated display behavior and bias artifact.
- [ ] Run `uv run ruff check src tests docs`.
- [ ] Run `uv run pytest -v`.
- [ ] Run `uv run mkdocs build --strict`.
