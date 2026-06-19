# Tabbed Reports And Bias Residuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-party HTML reports easier to navigate with tabs and add a residual-by-subgroup plot to bias reports.

**Architecture:** Update the shared `_html_document()` shell in `src/lumosai/model/plots.py` so each report section renders as a CSS-only tab. Extend `bias_report()` to compute row-level residuals by protected group and pass them into `bias_html()` for a new "Residuals by Group" tab.

**Tech Stack:** Python, pandas, numpy, matplotlib, pytest, existing Lumos HTML artifact helpers.

---

### Task 1: CSS-Only Tabs

**Files:**
- Modify: `src/lumosai/model/plots.py`
- Test: `tests/model/test_performance.py`

- [x] Add a failing assertion to an existing performance HTML test that generated HTML includes tab controls and checked default tab state.
- [x] Update `_html_document()` to wrap sections as radio-button driven tabs.
- [x] Run `uv run pytest tests/model/test_performance.py -v`.

### Task 2: Bias Residuals By Group

**Files:**
- Modify: `src/lumosai/model/bias.py`
- Modify: `src/lumosai/model/plots.py`
- Test: `tests/model/test_bias.py`

- [x] Add a failing bias HTML test that expects "Residuals by Group: segment" and the residual plot alt text.
- [x] Compute residual rows in `bias_report()` for each protected attribute.
- [x] Add `_bias_residual_plot()` and render it in `bias_html()`.
- [x] Run `uv run pytest tests/model/test_bias.py -v`.

### Task 3: Verification And Commit

**Files:**
- Modify: `docs/api.md`

- [x] Document that custom HTML reports use tabs.
- [x] Run `uv run ruff check src tests docs`.
- [x] Run `uv run pytest -v`.
- [x] Run `uv run mkdocs build --strict`.
- [ ] Commit with message `Add tabbed report HTML and bias residual plots`.
