# Performance Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `performance_drift_report()` for adaptive prediction-score drift, labeled performance drift, and residual drift.

**Architecture:** Add a focused `src/lumosai/model/performance_drift.py` module for PSI, residual construction, metric deltas, validation, and report assembly. Reuse existing ingestion, score normalization, metric calculation, threshold comparison, artifact, MLflow, and plot helpers. Add HTML rendering helpers to `src/lumosai/model/plots.py` and export the new report from `lumosai.model` and top-level `lumosai`.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn metrics through existing helpers, matplotlib HTML plots, Pydantic settings, pytest, uv, Ruff.

---

## File Structure

- Modify `src/lumosai/settings.py`: add `model.performance_drift_psi_threshold`.
- Create `src/lumosai/model/performance_drift.py`: validation, PSI, residuals, metric drift, result assembly.
- Modify `src/lumosai/model/plots.py`: add `performance_drift_html()`.
- Modify `src/lumosai/model/__init__.py` and `src/lumosai/__init__.py`: export `performance_drift_report`.
- Create `tests/model/test_performance_drift.py`: focused behavior tests.
- Modify `tests/test_public_api.py`: export test.
- Modify `docs/api.md` and add `docs/recipes/performance-drift.md`: user docs.

### Task 1: Settings And PSI Core

**Files:**
- Modify: `src/lumosai/settings.py`
- Create: `tests/model/test_performance_drift.py`
- Create: `src/lumosai/model/performance_drift.py`

- [ ] **Step 1: Write failing tests**

Create `tests/model/test_performance_drift.py` with:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.performance_drift import performance_drift_report
from lumosai.settings import Settings, settings


def test_performance_drift_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = Settings()
    assert loaded.model.performance_drift_psi_threshold == 0.2

    monkeypatch.setenv("LUMOSAI_MODEL__PERFORMANCE_DRIFT_PSI_THRESHOLD", "0.35")
    loaded = Settings()
    assert loaded.model.performance_drift_psi_threshold == 0.35


def test_prediction_only_score_psi_flags_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.model, "performance_drift_psi_threshold", 0.01)
    baseline = pd.DataFrame({"score": [0.01, 0.02, 0.03, 0.04, 0.05]})
    current = pd.DataFrame({"score": [0.95, 0.96, 0.97, 0.98, 0.99]})

    result = performance_drift_report(
        baseline,
        current,
        prediction_score="score",
        include_plots=False,
    )

    assert result.metadata["mode"] == "prediction_only"
    assert result.metrics["performance_drift/baseline/score_psi"] > 0.01
    assert result.summary["score"]["columns"] == ["score"]
    assert result.flagged == [
        {
            "comparison": "baseline",
            "metric": "score_psi",
            "value": result.metrics["performance_drift/baseline/score_psi"],
            "threshold": 0.01,
        }
    ]


def test_performance_drift_requires_signal() -> None:
    frame = pd.DataFrame({"score": [0.1, 0.2]})

    with pytest.raises(LumosValidationError, match="requires prediction_score or target"):
        performance_drift_report(frame, frame, include_plots=False)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/model/test_performance_drift.py -v
```

Expected: import failure because `lumosai.model.performance_drift` does not exist.

- [ ] **Step 3: Add settings and minimal report**

Add to `ModelSettings` in `src/lumosai/settings.py`:

```python
performance_drift_psi_threshold: float = Field(default=0.2, ge=0.0)
```

Create `src/lumosai/model/performance_drift.py` with PSI and prediction-only support.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/model/test_performance_drift.py -v
```

Expected: three tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lumosai/settings.py src/lumosai/model/performance_drift.py tests/model/test_performance_drift.py
git commit -m "Add prediction score drift report"
```

### Task 2: Labeled Classification And Regression Drift

**Files:**
- Modify: `src/lumosai/model/performance_drift.py`
- Modify: `tests/model/test_performance_drift.py`

- [ ] **Step 1: Add failing labeled tests**

Append tests for:

```python
def test_labeled_classification_adds_metric_and_residual_drift() -> None:
    baseline = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1, 1, 0],
            "prediction": [0, 0, 1, 1, 1, 0],
            "score": [0.05, 0.1, 0.8, 0.85, 0.9, 0.2],
        }
    )
    current = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1, 1, 0],
            "prediction": [1, 1, 0, 0, 1, 1],
            "score": [0.75, 0.8, 0.25, 0.3, 0.65, 0.9],
        }
    )

    result = performance_drift_report(
        baseline,
        current,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        include_plots=False,
    )

    assert result.metadata["mode"] == "labeled"
    assert result.metrics["performance_drift/baseline/baseline/accuracy"] == 1.0
    assert result.metrics["performance_drift/baseline/current/accuracy"] < 1.0
    assert "performance_drift/baseline/residual_psi" in result.metrics
    assert result.summary["residual"]["kind"] == "classification_probability"
    assert any(flag["metric"] == "metric_drift" for flag in result.flagged)


def test_labeled_regression_adds_metric_and_residual_drift() -> None:
    baseline = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.0, 3.0]})
    current = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [2.0, 3.0, 4.0]})

    result = performance_drift_report(
        baseline,
        current,
        target="actual",
        prediction="prediction",
        task_type="regression",
        include_plots=False,
    )

    assert result.metadata["mode"] == "labeled"
    assert result.metrics["performance_drift/baseline/current/rmse"] > 0.0
    assert "performance_drift/baseline/residual_psi" in result.metrics
    assert result.summary["residual"]["kind"] == "regression"
```

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest tests/model/test_performance_drift.py::test_labeled_classification_adds_metric_and_residual_drift tests/model/test_performance_drift.py::test_labeled_regression_adds_metric_and_residual_drift -v
```

Expected: missing metric/residual assertions fail.

- [ ] **Step 3: Implement labeled mode**

Use `get_metrics()`, `compare_metric()`, and existing score normalization. Add residual PSI for regression and classification probability residuals.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest tests/model/test_performance_drift.py -v
```

Expected: all performance drift tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lumosai/model/performance_drift.py tests/model/test_performance_drift.py
git commit -m "Add labeled performance drift"
```

### Task 3: Plots And Public API

**Files:**
- Modify: `src/lumosai/model/plots.py`
- Modify: `src/lumosai/model/performance_drift.py`
- Modify: `src/lumosai/model/__init__.py`
- Modify: `src/lumosai/__init__.py`
- Modify: `tests/model/test_performance_drift.py`
- Modify: `tests/test_public_api.py`

- [ ] **Step 1: Add failing tests**

Add tests asserting default HTML artifact is produced and top-level/domain exports work.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest tests/model/test_performance_drift.py::test_performance_drift_creates_default_html_artifact tests/test_public_api.py::test_performance_drift_public_api -v
```

Expected: export or artifact assertions fail.

- [ ] **Step 3: Implement plots and exports**

Add `performance_drift_html()` with metric delta table, score distribution, residual distribution, and residual scatter. Export `performance_drift_report`.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest tests/model/test_performance_drift.py tests/test_public_api.py -v
```

Expected: tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/lumosai/model/plots.py src/lumosai/model/performance_drift.py src/lumosai/model/__init__.py src/lumosai/__init__.py tests/model/test_performance_drift.py tests/test_public_api.py
git commit -m "Add performance drift plots and exports"
```

### Task 4: Docs And Final Verification

**Files:**
- Modify: `docs/api.md`
- Create: `docs/recipes/performance-drift.md`
- Modify: `mkdocs.yml` if recipes are explicitly listed.

- [ ] **Step 1: Add docs**

Document API, modes, PSI metrics, residuals, flags, and example calls.

- [ ] **Step 2: Verify**

Run:

```bash
uv run ruff check src tests docs
uv run pytest -v
uv run mkdocs build --strict
```

Expected: all commands pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/api.md docs/recipes/performance-drift.md mkdocs.yml
git commit -m "Document performance drift report"
```

---

## Self-Review

- Spec coverage: API, settings, PSI, labeled metrics, residuals, plots, validation, exports, and docs are covered.
- Placeholder scan: no placeholder markers remain.
- Type consistency: `performance_drift_report`, `performance_drift_psi_threshold`, and metric names match the design spec.
