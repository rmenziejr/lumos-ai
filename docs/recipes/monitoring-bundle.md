# Monitoring Bundle

Use `monitoring_report()` when a scheduled monitoring job is ready to move from individual report calls to one production bundle. The bundle still uses the same low-level primitives internally, but it validates the full request up front, groups child results in a `LumosRun`, and logs one combined MLflow run when MLflow is enabled.

## Minimal Drift Bundle

At minimum, pass the current production window, a benchmark frame, and the temporal drift columns. The bundle builds a representative monitoring sample and runs benchmark drift.

```python
from lumosai import monitoring_report

run = monitoring_report(
    current_window,
    benchmark=train_benchmark,
    feature_columns=["tenure", "plan", "monthly_spend"],
    categorical_columns=["plan"],
    temporal_features=["event_date"],
    sample_size=10_000,
    report_name="Churn Monitoring",
    experiment_name="model-monitoring",
)

sample = run.results["monitoring_window"].artifacts["sample"]
drift_metrics = run.results["drift_benchmark"].metrics
```

`feature_columns` are the analysis columns for drift and downstream model reports. Temporal columns should be passed through `temporal_features` or `time_column`; do not include them in `feature_columns`.

`monitoring_report()` does not automatically attach feature-importance results yet. For importance-aware drift alerts, call `drift_report()` directly with `important_features` or `importance_result` in the monitoring step where that information is available.

If you have one time column, `time_column` is enough:

```python
run = monitoring_report(
    current_window,
    benchmark=train_benchmark,
    feature_columns=["tenure", "plan", "monthly_spend"],
    categorical_columns=["plan"],
    time_column="event_date",
)
```

## Performance And Bias

Pass `target` and `prediction` when labels are available for the current window. Add `protected_attribute` when the monitoring purpose permits group-wise checks.

```python
run = monitoring_report(
    current_window_with_labels,
    benchmark=train_benchmark,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    protected_attribute=["region", "segment"],
    feature_columns=["tenure", "plan", "monthly_spend"],
    categorical_columns=["plan"],
    time_column="event_date",
    report_name="Churn Monitoring With Labels",
    experiment_name="model-monitoring",
)

performance = run.results["performance"]
bias = run.results["bias"]
```

Performance runs when both `target` and `prediction` are provided. Bias runs when `protected_attribute` is provided and also requires `target` and `prediction`.

You can make intent explicit with `include_performance=True` or `include_bias=True`. If the required columns or parameters are missing, preflight validation raises before any report work starts.

## Previous Window Drift

Pass `previous_window` to compare the current production window with the immediately prior production window.

```python
run = monitoring_report(
    current_window,
    benchmark=train_benchmark,
    previous_window=last_week_window,
    feature_columns=["tenure", "plan", "monthly_spend"],
    categorical_columns=["plan"],
    time_column="event_date",
    experiment_name="model-monitoring",
)

previous_drift = run.results.get("drift_previous_window")
```

Previous-window drift only runs when both conditions are true:

- `previous_window` is provided.
- `settings.bundles.include_previous_window_drift` is `True`.

When `previous_window` is provided but the setting is disabled, the bundle skips `drift_previous_window` and records the reason in `run.metadata["skipped_reports"]`.
Provided previous-window data still goes through preflight column validation before the report is skipped.

```python
from lumosai.settings import Settings

loaded_settings = Settings(
    bundles={"include_previous_window_drift": False},
)

run = monitoring_report(
    current_window,
    benchmark=train_benchmark,
    previous_window=last_week_window,
    feature_columns=["tenure", "plan", "monthly_spend"],
    time_column="event_date",
    loaded_settings=loaded_settings,
)

assert "drift_previous_window" not in run.results
assert run.metadata["skipped_reports"]["drift_previous_window"] == (
    "previous_window drift disabled by settings"
)
```

## Fail-Fast Preflight

`monitoring_report()` validates the full bundle request before starting expensive report generation. It checks required drift columns, temporal columns, previous-window drift columns, performance inputs, bias inputs, and categorical column consistency.

This catches partial production configuration early. For example, `include_bias=True` without `protected_attribute`, or `include_performance=True` without `prediction`, raises a `LumosValidationError` before drift or sampling starts.

The current implementation requires fail-fast mode. Passing `loaded_settings` with `bundles.fail_fast=False` raises a `LumosValidationError`.

## MLflow Logging

When `experiment_name` is provided, or `loaded_settings.mlflow.default_experiment_name` is set, the bundle opens one MLflow run around all child reports.

During the bundle, primitive child JSON dictionary logging is suppressed so each child does not write its own result dictionary artifact. Child metrics and configured report artifacts still log through the active run. After the child reports finish, the bundle logs one combined `lumosai_run.json` artifact when MLflow dictionary logging is enabled.

```python
run = monitoring_report(
    current_window,
    benchmark=train_benchmark,
    feature_columns=["tenure", "plan", "monthly_spend"],
    time_column="event_date",
    experiment_name="model-monitoring",
)

payload = run.to_dict()
```

Use `run.to_dict()` for the same grouped shape that is logged to `lumosai_run.json`.
