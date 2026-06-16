# Pima Notebook Monitoring Capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the Pima walkthrough notebook so it demonstrates importance-aware drift and adaptive performance drift reports with inline HTML outputs.

**Architecture:** Keep the existing notebook flow intact and add one focused monitoring capability section after drift windows. Reuse existing report objects, scored windows, and `display_html_artifact()` helper instead of restructuring the walkthrough.

**Tech Stack:** Jupyter notebook JSON, `lumosai.data.drift_report`, `lumosai.model.performance_drift_report`, `uv run`, standard library JSON tooling.

---

### Task 1: Add Monitoring Capability Cells

**Files:**
- Modify: `examples/notebooks/pima_diabetes_walkthrough.ipynb`

- [ ] **Step 1: Update imports**

Add `performance_drift_report` to the model imports:

```python
from lumosai.model import (
    bias_report,
    calibration_report,
    feature_importance,
    performance_drift_report,
    performance_report,
)
```

- [ ] **Step 2: Make drift reports importance-aware**

Pass the existing `importance` result into both drift reports:

```python
importance_result=importance,
```

- [ ] **Step 3: Add adaptive performance drift cells**

Insert a markdown cell and code cell after the drift report cell. The code cell creates:

```python
score_drift = performance_drift_report(...)
labeled_performance_drift = performance_drift_report(...)
```

Then display selected metrics, flags, and both HTML artifacts inline.

### Task 2: Execute and Verify Notebook

**Files:**
- Modify: `examples/notebooks/pima_diabetes_walkthrough.ipynb`

- [ ] **Step 1: Run notebook execution**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/notebooks/pima_diabetes_walkthrough.ipynb
```

Expected: command exits `0`.

- [ ] **Step 2: Run focused verification**

Run:

```bash
uv run ruff check src tests docs
uv run pytest tests/model/test_performance_drift.py tests/data/test_drift.py::test_drift_report_extracts_top_n_permutation_features -v
uv run mkdocs build --strict
```

Expected: all commands exit `0`.

### Self-Review

- Spec coverage: The plan updates imports, importance-aware drift usage, prediction-only performance drift, labeled performance drift, and inline HTML display.
- Placeholder scan: No TODO/TBD placeholders.
- Type consistency: Function names match existing public API.
