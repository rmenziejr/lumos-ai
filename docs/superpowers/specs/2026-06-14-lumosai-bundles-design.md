# Lumosai Bundle Reports Design

## Purpose

`lumosai` should support both notebook development and scheduled production monitoring.
The low-level report functions remain the development and testing surface. Bundle APIs provide the production surface by composing those same primitives with consistent settings, validation, result grouping, and MLflow behavior.

The product principle is:

> Develop with primitives. Promote to bundles. Govern with settings.

This keeps the package from becoming an orchestrator while still reducing repetitive monitoring glue code.

## Current API Position

The current package has useful primitives:

- `profile()` for ydata profiling.
- `drift_report()` for Evidently drift.
- `build_sample()` for train benchmark, holdout, and monitoring window samples.
- `performance_report()` for labeled model performance.
- `bias_report()` for group-wise behavior.
- `feature_importance()` for permutation or SHAP importance.
- `LumosResult` as the common result shape.
- MLflow helpers for metrics, JSON summaries, report artifacts, and sample metadata.

These functions should stay stable and directly callable. They are the right interface for notebooks, exploratory analysis, tests, custom pipelines, and debugging.

## Design Goals

1. Add production-friendly bundle APIs without hiding the primitives.
2. Fail fast when a bundle is explicitly asked to run a report and required inputs are missing.
3. Keep settings useful but small. Defaults should make common use easy; users should not need a large config file.
4. Validate the bundle before running expensive report work.
5. Return a grouped result object that can be logged, inspected, serialized, and tested.
6. Keep orchestration, scheduling, late label joins, data storage, retries, and promotion gates outside `lumosai`.

## Non-Goals

- Do not add a scheduler or monitoring window manager.
- Do not manage model registry state.
- Do not own late-arriving label joins.
- Do not auto-detect production intent from arbitrary columns.
- Do not replace MLflow, Evidently, ydata-profiling, or orchestration systems.
- Do not require users to configure many bundle settings before first use.

## Core Concept: Intent From Arguments

Bundle arguments define report intent.

If an optional input is provided, the corresponding report is expected and must run successfully. Missing required columns or dependencies should raise a `LumosValidationError` before any expensive reports run.

Examples:

- Passing `benchmark` means benchmark drift is expected.
- Passing `previous_window` means previous-window drift is expected.
- Passing both `target` and `prediction` means performance is expected.
- Passing `protected_attribute` means bias is expected.
- Passing `model` and `feature_columns` with `include_feature_importance=True` means feature importance is expected.

The bundle should not silently decide to run performance just because columns named `target` and `prediction` happen to exist. It should run reports based on explicit arguments and settings.

## Proposed Public APIs

### `LumosRun`

`LumosRun` groups multiple `LumosResult` objects from one logical workflow execution.

```python
@dataclass(slots=True)
class LumosRun:
    run_type: str
    results: dict[str, LumosResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, float]: ...

    @property
    def flagged(self) -> list[dict[str, Any]]: ...

    def to_dict(self) -> dict[str, Any]: ...
```

Expected behavior:

- `metrics` merges metrics from child results.
- `flagged` concatenates flagged findings from child results and annotates each item with the result key.
- `to_dict()` returns JSON-safe data for logging or API responses.
- Child result keys should be stable names such as `train_sample`, `holdout_sample`, `drift_benchmark`, `drift_previous_window`, `performance`, `bias`, and `feature_importance`.

`LumosRun` should not own MLflow sessions initially. Bundle functions should call primitives inside one active MLflow run when needed.

### `training_report(...)`

Builds the standard post-training report bundle.

```python
def training_report(
    train: Any,
    holdout: Any,
    *,
    target: str,
    prediction: str | None = None,
    prediction_score: str | None = None,
    model: Any | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    protected_attribute: str | list[str] | dict[str, list[float]] | None = None,
    time_column: str | None = None,
    sample_size: int | None = None,
    include_profile: bool | None = None,
    include_performance: bool | None = None,
    include_bias: bool | None = None,
    include_feature_importance: bool | None = None,
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosRun:
```

Default behavior:

- Always build a `train_sample` from `train` with `role="train_benchmark"`.
- Always build a `holdout_sample` from `holdout` with `role="holdout"` when `prediction` is provided; otherwise build a sample without prediction.
- Run performance when `prediction` is provided or `include_performance=True`.
- Run bias when `protected_attribute` is provided or `include_bias=True`.
- Run feature importance when `model` is provided and `include_feature_importance` is not `False`.
- Run profile only when `include_profile=True` by default, because profiling can be expensive.

Fail-fast rules:

- `target` is required.
- If performance is enabled, `prediction` is required and both `target` and `prediction` must exist in the holdout frame.
- If bias is enabled, `protected_attribute`, `target`, and `prediction` are required.
- If feature importance is enabled, `model` and non-empty `feature_columns` are required.
- If samples need temporal behavior, `time_column` must exist.
- If `categorical_columns` are provided, they must be included in the relevant analysis columns.

### `monitoring_report(...)`

Builds the standard scheduled monitoring report bundle.

```python
def monitoring_report(
    current: Any,
    *,
    benchmark: Any,
    previous_window: Any | None = None,
    target: str | None = None,
    prediction: str | None = None,
    prediction_score: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    protected_attribute: str | list[str] | dict[str, list[float]] | None = None,
    temporal_features: list[str] | None = None,
    time_column: str | None = None,
    sample_size: int | None = None,
    include_performance: bool | None = None,
    include_bias: bool | None = None,
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosRun:
```

Default behavior:

- Always build a `monitoring_window` sample from `current`.
- Always run drift against `benchmark` as `drift_benchmark`.
- Run previous-window drift when `previous_window` is provided.
- Run performance when `target` and `prediction` are provided or `include_performance=True`.
- Run bias when `protected_attribute` is provided or `include_bias=True`.
- Do not run feature importance in monitoring by default; feature importance is generally a training or evaluation report.

Fail-fast rules:

- `benchmark` is required.
- Drift requires `temporal_features`. If omitted and `time_column` is provided, use `[time_column]`. If both are omitted, raise.
- If previous-window drift is expected, `previous_window` must contain the same drift analysis columns as `current`.
- If performance is enabled, `target` and `prediction` are required.
- If bias is enabled, `protected_attribute`, `target`, and `prediction` are required.

## Bundle Settings

Add a small `BundleSettings` group under `settings`.

```python
class BundleSettings(BaseModel):
    include_profile_in_training: bool = False
    include_feature_importance_in_training: bool = True
    include_previous_window_drift: bool = True
    fail_fast: bool = True
```

Settings should remain defaults only. Function arguments override settings for one call.

Avoid a large settings surface. Do not add per-report nested settings until real user workflows require it. Existing settings already cover most defaults:

- `settings.data.default_sample_size`
- `settings.data.drift_share_threshold`
- `settings.data.log_sample_metadata`
- `settings.data.log_sample_artifacts`
- `settings.model.metric_thresholds`
- `settings.model.shap_sample_size`
- `settings.mlflow.default_experiment_name`
- `settings.artifacts.keep_local`

## Validation Strategy

Each bundle should run a preflight validation phase before running any report.

Preflight should:

- convert input frames to pandas once per input;
- validate required columns for all expected reports;
- validate non-empty `feature_columns` where needed;
- validate required optional dependencies for enabled reports where practical;
- validate that no expected report is partially configured;
- produce clear `LumosValidationError` messages.

Preflight should not do expensive statistical work. Its job is to catch missing inputs before generating reports.

## MLflow Behavior

Bundles should group report calls under one MLflow run when `experiment_name` is provided.

Implementation approach:

- Use the existing `mlflow_run()` context manager around the primitive calls.
- Primitive calls already respect an active MLflow run and should log into that active run.
- The bundle can log a combined `lumosai_run.json` after all child results complete.

Metric namespaces already prevent overlap:

- `drift/benchmark/...`
- `drift/previous_window/...`
- `performance/...`
- `bias/...`
- `importance/...`

## Error Handling

Use `LumosValidationError` for invalid user inputs and missing columns.
Use `LumosOptionalDependencyError` for missing optional dependencies.

Bundle errors should identify:

- bundle name;
- expected child report;
- missing parameter or column;
- how to disable the report if it was not intended.

Example:

```text
monitoring_report expected performance because include_performance=True, but prediction is missing.
Pass prediction=... or set include_performance=False.
```

## Testing Strategy

Unit tests should cover:

- `LumosRun.metrics`, `LumosRun.flagged`, and `LumosRun.to_dict()`.
- `training_report()` happy path with samples, performance, and feature importance.
- `training_report()` fail-fast behavior for missing prediction when performance is enabled.
- `training_report()` skips optional profile by default.
- `monitoring_report()` happy path with benchmark drift and performance.
- `monitoring_report()` previous-window drift when `previous_window` is provided.
- `monitoring_report()` fail-fast behavior when drift temporal features are missing.
- Settings override behavior.
- Explicit function arguments overriding bundle settings.
- MLflow grouping under one active run using fakes.

Integration tests should stay focused. Avoid slow ydata, Evidently, and SHAP integration in the default suite unless the existing fake patterns can keep tests fast.

## Documentation Strategy

Docs should present the promotion path:

1. Develop with primitives in notebooks.
2. Stabilize sample and report arguments.
3. Promote the same arguments into `training_report()` or `monitoring_report()`.
4. Govern defaults through settings.

Add recipes:

- Training bundle report.
- Monitoring bundle report.
- How to fail fast in production jobs.
- How to override settings for one bundle call.

## Implementation Decisions

Use these decisions for the first implementation plan:

- `training_report()` requires both `train` and `holdout`. A train-only bundle can be added later if real workflows need it.
- Bundle functions accept raw dataframe-like inputs only in the first version. They should not accept prebuilt sample results yet because that complicates validation and result lineage.
- `LumosRun` should not include a `warnings` field initially. Skipped optional reports should appear in `metadata["skipped_reports"]` with clear reasons.
- Add `LumosRun` to `src/lumosai/results.py` alongside `LumosResult`.
- Add bundle functions to a new `src/lumosai/bundles.py` module and export them from the top-level package.
- Add combined run logging as `log_run()` in `src/lumosai/mlflow.py`, because MLflow concerns already live there.

## Recommendation

Implement this in two increments:

1. Add `LumosRun`, `BundleSettings`, and `monitoring_report()`.
2. Add `training_report()` and docs recipes.

`monitoring_report()` should come first because it demonstrates the strongest production value: one call that runs benchmark drift, optional previous-window drift, performance when labels arrive, and bias when explicitly requested.
