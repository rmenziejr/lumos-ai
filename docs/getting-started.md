# Getting Started

## Install

Install the package from the repository during local development:

```bash
uv sync
```

For MLflow logging support, install the optional extra when packaging or consuming the project:

```bash
pip install "lumosai[mlflow]"
```

## Profile A Dataset

```python
from lumosai.data import profile

result = profile(feature_table, time_column="event_date")

print(result.summary)
print(result.artifacts)
```

`profile()` creates a ydata-profiling report. Without MLflow logging, the HTML artifact is kept locally according to artifact settings.
Use `target` and `feature_columns` to focus the report and move the outcome column to the first position.

For a profile dry run, pass `log_analysis=False`. This skips profile artifact generation and MLflow logging for the call, even when a default MLflow experiment is configured.

```python
profile(
    feature_table,
    target="churned",
    feature_columns=["tenure", "plan", "day_of_week"],
    categorical_columns=["plan", "day_of_week"],
    report_name="Training Feature Profile",
)
```

```python
profile(feature_table, time_column="event_date", log_analysis=False)
```

## Check Data Drift

```python
from lumosai.data import drift_report

result = drift_report(
    reference=train_benchmark,
    current=current_window,
    temporal_features=["event_date"],
    feature_columns=["tenure", "plan", "day_of_week"],
    categorical_columns=["plan", "day_of_week"],
    comparison="benchmark",
    report_name="Benchmark Drift",
)

print(result.metrics)
print(result.flagged)
```

Temporal features are excluded from drift calculations after validation.

## Build Representative Samples

```python
from lumosai.data import build_sample

sample = build_sample(
    feature_table,
    role="train_benchmark",
    target="churned",
    feature_columns=["tenure", "plan", "day_of_week"],
    categorical_columns=["plan", "day_of_week"],
    time_column="event_date",
    sample_size=10000,
    artifact_path="artifacts/train_benchmark",
    experiment_name="model-training",
)

print(sample.summary)
print(sample.metadata)
```

When `artifact_path` has no suffix, `settings.data.sample_artifact_format` controls the file format. With MLflow, sample metadata logging follows settings and raw sample artifact logging is opt-in.

## Check Model Performance

```python
from lumosai.model import performance_report

result = performance_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    report_name="Holdout Performance",
)

print(result.metrics)
```

## Check Feature Importance

```python
from lumosai.model import feature_importance

result = feature_importance(
    model,
    validation_frame,
    target="actual",
    feature_columns=["tenure", "plan_code", "day_of_week"],
    method="permutation",
    report_name="Holdout Feature Importance",
    experiment_name="model-training",
)

print(result.summary["methods"]["permutation"]["features"])
```

## Set Shared Defaults

Use `lumosai.settings` as the control point for standards that should be reused across scripts, scheduled jobs, and notebooks. Settings can define MLflow behavior, artifact handling, sample defaults, metric thresholds, and bundle behavior without passing the same options to every report call.

```bash
export LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME=model-monitoring
export LUMOSAI_DATA__DEFAULT_SAMPLE_SIZE=25000
export LUMOSAI_MODEL__METRIC_THRESHOLDS__RECALL__VALUE=0.9
export LUMOSAI_BUNDLES__FAIL_FAST=true
```

Nested environment variables use the `LUMOSAI_` prefix and double underscores between settings groups.

## Check Bias Across Groups

```python
from lumosai.model import bias_report

result = bias_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    protected_attribute=["region", "segment"],
    report_name="Holdout Bias",
)

print(result.summary)
print(result.flagged)
```

## MLflow Logging

Pass `experiment_name` to log metrics, summaries, and supported artifacts to MLflow:

```python
performance_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    experiment_name="model-monitoring",
)
```

You can also set `settings.mlflow.default_experiment_name` with `LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME` so repeated report calls log without passing `experiment_name` each time. Results stay local only when both the function argument and default setting are absent.

When no MLflow run is active, each report call creates its own run. Start an MLflow run around multiple calls when they should belong to the same scheduled execution.
