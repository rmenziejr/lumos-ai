# Report Kwargs And MkDocs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add report names, schema columns, controlled third-party kwargs, and local MkDocs docs.

**Architecture:** Add shared schema/kwargs validation helpers, wire them into report wrappers, and keep third-party kwargs allowlisted. MkDocs uses the existing Markdown docs without publishing workflow.

**Tech Stack:** Python, pandas, Pydantic settings, ydata-profiling, Evidently, MkDocs Material, pytest, ruff, mypy.

---

## Task 1: Shared Schema And Kwargs Helpers

**Files:**
- Create: `src/lumosai/schema.py`
- Test: `tests/test_schema.py`

- [ ] Add helpers for selecting target/features/time columns, validating categorical subsets, and allowlisting kwargs.
- [ ] Add tests for target-first ordering, missing columns, duplicate target in features, categorical subset validation, and unsupported kwargs.

## Task 2: Profile API Expansion

**Files:**
- Modify: `src/lumosai/data/profiling.py`
- Modify: `tests/data/test_profiling.py`
- Modify: `docs/api.md`

- [ ] Add `target`, `feature_columns`, `categorical_columns`, `report_name`, and `ydata_kwargs`.
- [ ] Pass `report_name` as ydata `title`; reject `title`/`minimal` in `ydata_kwargs`.
- [ ] Select profiled columns with target first and preserve time column for temporal sampling.
- [ ] Add tests for target-first profile data, allowed ydata kwargs, unsupported kwargs, and categorical metadata.

## Task 3: Drift API Expansion

**Files:**
- Modify: `src/lumosai/data/drift.py`
- Modify: `tests/data/test_drift.py`
- Modify: `docs/api.md`

- [ ] Add `feature_columns`, `categorical_columns`, `report_name`, and `evidently_kwargs`.
- [ ] Pass allowed preset/report kwargs; reject unsupported kwargs.
- [ ] Pass current Evidently categorical overrides via `Dataset.from_pandas(..., DataDefinition(...))`.
- [ ] Pass `report_name` to current Evidently run names when supported.
- [ ] Add tests for feature column filtering, categorical overrides, report/preset kwargs, report name, and invalid kwargs.

## Task 4: Model Report Metadata Expansion

**Files:**
- Modify: `src/lumosai/model/performance.py`
- Modify: `src/lumosai/model/bias.py`
- Modify: `tests/model/test_performance.py`
- Modify: `tests/model/test_bias.py`
- Modify: `docs/api.md`

- [ ] Add `report_name`, `feature_columns`, and `categorical_columns`.
- [ ] Validate feature/categorical columns and store them in metadata.
- [ ] Add tests for metadata and validation.

## Task 5: MkDocs Local Setup

**Files:**
- Create: `mkdocs.yml`
- Modify: `pyproject.toml`
- Modify: `docs/development.md`
- Modify: `README.md`

- [ ] Add MkDocs Material dev dependencies.
- [ ] Add local nav in `mkdocs.yml`.
- [ ] Document `uv run mkdocs serve` and `uv run mkdocs build --strict`.
- [ ] Verify `uv sync`, `uv run mkdocs build --strict`, `uv run ruff check .`, `uv run pytest -v`, and `uv run mypy src/lumosai`.
