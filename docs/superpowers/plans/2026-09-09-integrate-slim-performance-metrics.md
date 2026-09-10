# Slim Performance Metrics Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the reusable metric-selection and MLflow fold-validation features from PR #4 into the current typed performance-plot implementation on PR #10.

**Architecture:** Keep `PerformancePlot` and `plots=[...]` as the plot-selection API. Add typed metric selection in `model.metrics`, add a `profile="metrics_only"` default profile for fold/tuning loops, and thread `mlflow_step`/`log_dict` through existing MLflow logging. Explicit plot/logging arguments override profile defaults. Do not carry the unrelated tabbed-report or bias-residual stack from PR #4.

**Tech Stack:** Python 3.12, pandas, scikit-learn, MLflow, pytest, Ruff, MkDocs.

**Spec:** PR #4 slim-performance design, reconciled with the approved PR #10 `PerformancePlot` design.

## Global Constraints

- Preserve existing metric key names.
- Preserve `plots: list[PerformancePlot] | None` from PR #10.
- `profile="standard"` preserves existing rich-report behavior.
- `profile="metrics_only"` defaults to no plots, no lift metrics, and no JSON result artifact, while scalar MLflow metric logging remains enabled.
- Explicit `plots`, `include_plots`, `include_lift`, and `log_dict` override profile defaults.
- Do not include PR #4's tabbed HTML or bias-residual changes.

---

### Task 1: Typed metric selection

**Files:** `src/lumosai/model/metrics.py`, `src/lumosai/settings.py`, metric/settings tests.

- [ ] Add failing tests for `metrics="default"`, `metrics="all"`, explicit metric lists, invalid metrics, score-required metrics, task mismatch, and custom metric collisions.
- [ ] Add typed metric aliases/constants and runtime selection/validation.
- [ ] Include `log_loss` in default classification probability metrics and lower-is-better thresholds.
- [ ] Verify focused metric/settings tests.

### Task 2: Metrics-only report profile and MLflow steps

**Files:** `src/lumosai/model/performance.py`, `src/lumosai/mlflow.py`, `src/lumosai/artifacts.py`, performance/logging tests.

- [ ] Add failing tests for selected metrics, metrics-only defaults, explicit plot overrides, train/holdout filtering, `mlflow_step`, and `log_dict`.
- [ ] Reconcile profile defaults with `plots=[...]`: explicit `plots` wins; otherwise explicit `include_plots` wins; otherwise profile decides.
- [ ] Thread selected metrics through current/train metric computation.
- [ ] Pass `mlflow_step` to `mlflow.log_metrics(..., step=...)` and store it in metadata.
- [ ] Suppress JSON result logging by default in `metrics_only`.
- [ ] Verify focused performance/MLflow/artifact tests.

### Task 3: Public exports and docs

**Files:** `src/lumosai/model/__init__.py`, `src/lumosai/__init__.py`, `docs/api.md`, `docs/recipes/tuning-and-final-training.md`, public API tests.

- [ ] Export metric aliases/constants without removing `PerformancePlot`.
- [ ] Document `metrics`, `profile`, `mlflow_step`, `log_dict`, and precedence with `plots`/`include_plots`.
- [ ] Update fold-validation recipe to log folds as MLflow steps with `profile="metrics_only"`.
- [ ] Verify public API/docs diff.

### Task 4: Final verification

- [ ] Inspect PR #10 diff for accidental tabbed-report/bias-residual changes.
- [ ] Run available tests/lint/docs build; if execution remains unavailable, document the limitation and verify through repository diff plus targeted numerical/API inspection.
