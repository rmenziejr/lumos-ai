# Representative Samples

Representative samples are bounded dataframes for profiling, drift, performance, and audit workflows. They give later reports a stable, documented slice of a larger training, holdout, or production window without making `lumosai` responsible for upstream storage.

Raw samples are not sent to MLflow by default. When `experiment_name` is provided, `build_sample()` logs sample metadata by default and logs the raw sample artifact only when enabled by settings or `log_artifact=True`.

## Training Benchmark

Use a training benchmark as the stable reference for later drift checks. It should represent the model features and target distribution used during training. Temporal columns are included only when you pass `time_column` or `temporal_columns`; they are excluded from the model feature set by default.

```python
from lumosai.data import build_sample

train_sample = build_sample(
    training_frame,
    role="train_benchmark",
    target="churned",
    feature_columns=["tenure", "plan", "monthly_spend", "day_of_week"],
    categorical_columns=["plan", "day_of_week"],
    time_column="event_date",
    sample_size=10000,
    artifact_path="artifacts/samples/train_benchmark",
    experiment_name="model-training",
)
```

If `artifact_path` has an explicit suffix, that suffix controls the local file format. If it has no suffix, `settings.data.sample_artifact_format` is used.

## Holdout

Use a holdout sample to document the evaluated population and keep target, prediction, feature, and time columns together for later review.

```python
from lumosai.data import build_sample

holdout_sample = build_sample(
    validation_scored,
    role="holdout",
    target="actual",
    prediction="prediction",
    feature_columns=["tenure", "plan", "monthly_spend", "day_of_week"],
    time_column="event_date",
    sample_size=5000,
    experiment_name="model-training",
)
```

## Monitoring Window

Use a monitoring window sample to capture the production frame being evaluated. This is useful before drift checks, and it can also support performance and bias reporting when labels arrive.

```python
from lumosai.data import build_sample

window_sample = build_sample(
    current_window,
    role="monitoring_window",
    feature_columns=["tenure", "plan", "monthly_spend", "day_of_week"],
    categorical_columns=["plan", "day_of_week"],
    time_column="event_date",
    sample_size=5000,
    experiment_name="model-monitoring",
)
```

## MLflow Guidance

Prefer metadata-first logging for representative samples. The metadata includes the role, strategy, row counts, selected columns, schema summary, digest, and optional time range. This makes MLflow useful for lineage without copying raw records into the artifact store.

Raw artifact logging is opt-in. Enable it only when the artifact store is approved for representative datasets:

```python
build_sample(
    current_window,
    role="monitoring_window",
    feature_columns=["tenure", "plan", "monthly_spend"],
    artifact_path="artifacts/samples/current_window.csv",
    experiment_name="model-monitoring",
    log_artifact=True,
)
```
