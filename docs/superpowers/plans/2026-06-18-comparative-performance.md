# Comparative Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-vs-holdout comparative performance metrics so overfitting/generalization gaps are visible in MLflow without requiring train plots.

**Architecture:** Extend `performance_report()` with an optional scored train dataframe and optional train plot flag. When train data is provided, emit split-aware `performance/train/*`, `performance/holdout/*`, `performance/gap/*`, and `performance/ratio/*` metrics. Update `training_report()` to pass scored train data when prediction columns exist on train, while keeping plots focused on holdout by default.

**Tech Stack:** Python, pandas, pytest, existing LumosResult/MLflow artifact helpers.

---

### Task 1: Low-Level Comparative Performance

**Files:**
- Modify: `src/lumosai/model/performance.py`
- Test: `tests/model/test_performance.py`

- [ ] **Step 1: Write the failing low-level test**

Add a test that calls `performance_report(holdout, train=train)` and expects split metrics, gap metrics, ratio metrics, and no legacy unsplit metrics.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `uv run pytest tests/model/test_performance.py::test_performance_report_adds_train_holdout_gap_metrics -v`

Expected: FAIL because `performance_report()` does not accept `train`.

- [ ] **Step 3: Implement minimal comparative metrics**

Update `performance_report()` to accept `train: Any | None = None` and `include_train_plots: bool = False`. Normalize and validate the train frame with the same target, prediction, prediction_score, score_labels, task type, feature, and categorical handling as holdout. Compute holdout metrics and train metrics independently. If train is provided, emit namespaced metrics:

```text
performance/holdout/<metric>
performance/train/<metric>
performance/gap/<metric>
performance/ratio/<metric>
```

For higher-is-better metrics, gap is `train - holdout`. For lower-is-better metrics, gap is `holdout - train`. Ratio is `holdout / train` when higher is better and `holdout / train` when lower is better as well, preserving the direct observed ratio.

- [ ] **Step 4: Run focused performance tests**

Run: `uv run pytest tests/model/test_performance.py -v`

Expected: PASS.

### Task 2: Training Bundle Integration

**Files:**
- Modify: `src/lumosai/bundles.py`
- Test: `tests/test_bundles.py`

- [ ] **Step 1: Write failing bundle tests**

Add a test that monkeypatches `performance_report()` and verifies `training_report()` passes `train=<train_df>` and `include_train_plots=False` when the train dataframe has the prediction columns.

- [ ] **Step 2: Run the focused bundle test and verify it fails**

Run: `uv run pytest tests/test_bundles.py::test_training_report_passes_train_frame_to_performance_report -v`

Expected: FAIL because the bundle does not pass `train`.

- [ ] **Step 3: Implement bundle integration**

Update `_preflight_training_report()` so when performance is expected, train and holdout both require `target`, `prediction`, and `prediction_score` if provided. Update the `training_report()` call to:

```python
performance_report(
    holdout_pd,
    target=target,
    prediction=prediction,
    prediction_score=prediction_score,
    train=train_pd,
    include_train_plots=False,
    ...
)
```

- [ ] **Step 4: Run focused bundle tests**

Run: `uv run pytest tests/test_bundles.py -v`

Expected: PASS.

### Task 3: Documentation And Verification

**Files:**
- Modify: `docs/api.md`
- Modify: `docs/recipes/training-pipeline-reporting.md`

- [ ] **Step 1: Document the new API**

Update the `performance_report()` signature and notes to explain `train` and `include_train_plots`.

- [ ] **Step 2: Document training usage**

Show that scored train and scored holdout inputs produce train, holdout, gap, and ratio metrics for overfitting checks.

- [ ] **Step 3: Run full verification**

Run:

```bash
uv run ruff check src tests docs
uv run pytest -v
uv run mkdocs build --strict
```

Expected: all pass.
