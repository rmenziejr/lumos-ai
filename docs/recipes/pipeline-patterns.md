# Pipeline Patterns

`lumosai` is a reporting SDK. The orchestrator owns scheduling, joins, storage, retries, credentials, and promotion rules. Pass prepared dataframes to `lumosai` at the points where you want profiling, drift, performance, bias, sample, or importance results.

Use the low-level primitives directly while developing, debugging, testing, or building custom pipeline steps. They make each report boundary explicit and are easy to unit test. When the monitoring flow is stable enough for scheduled production runs, promote the same prepared inputs to `monitoring_report()` so sampling, drift, optional performance, optional bias, fail-fast validation, result grouping, and MLflow bundle logging happen consistently.

## Data Pipeline

Use data pipeline jobs to validate and profile the feature table that downstream training or scoring jobs will consume.

```python
from lumosai.data import build_sample, profile

sample = build_sample(
    feature_table,
    role="train_benchmark",
    target="churned",
    feature_columns=["tenure", "plan", "monthly_spend"],
    time_column="event_date",
    experiment_name="feature-pipeline",
)

profile(
    sample.artifacts["sample"],
    target="churned",
    feature_columns=["tenure", "plan", "monthly_spend"],
    categorical_columns=["plan"],
    report_name="Training Feature Profile",
    experiment_name="feature-pipeline",
)
```

The pipeline still owns where the full feature table and sampled artifact are stored. `lumosai` records the report output and sample metadata.

## Training Pipeline

Use training jobs to create a benchmark sample, evaluate holdout performance, check permitted bias slices, and record feature importance.

```python
from lumosai.data import build_sample
from lumosai.model import bias_report, feature_importance, performance_report

train_sample = build_sample(
    training_frame,
    role="train_benchmark",
    target="churned",
    feature_columns=feature_columns,
    time_column="event_date",
    experiment_name="model-training",
)

holdout_sample = build_sample(
    validation_scored,
    role="holdout",
    target="actual",
    prediction="prediction",
    feature_columns=feature_columns,
    time_column="event_date",
    experiment_name="model-training",
)

performance_report(
    holdout_sample.artifacts["sample"],
    target="actual",
    prediction="prediction",
    feature_columns=feature_columns,
    experiment_name="model-training",
)

bias_report(
    holdout_sample.artifacts["sample"],
    target="actual",
    prediction="prediction",
    protected_attribute=["region", "segment"],
    experiment_name="model-training",
)

feature_importance(
    model,
    validation_scored,
    target="actual",
    feature_columns=feature_columns,
    report_name="Holdout Feature Importance",
    experiment_name="model-training",
)
```

## Ongoing Monitoring Pipeline

Use monitoring jobs to compare each production window against the training benchmark. Optionally compare the same current window against the previous production window to distinguish long-term drift from short-term movement.

During development, the primitive calls keep each report isolated:

```python
from lumosai.data import drift_report
from lumosai.model import bias_report, performance_report

drift_report(
    reference=train_benchmark,
    current=current_window,
    temporal_features=["event_date"],
    feature_columns=feature_columns,
    comparison="benchmark",
    experiment_name="model-monitoring",
)

drift_report(
    reference=previous_window,
    current=current_window,
    temporal_features=["event_date"],
    feature_columns=feature_columns,
    comparison="previous_window",
    experiment_name="model-monitoring",
)

performance_report(
    current_window_with_labels,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    feature_columns=feature_columns,
    experiment_name="model-monitoring",
)

bias_report(
    current_window_with_labels,
    target="actual",
    prediction="prediction",
    protected_attribute=["region", "segment"],
    experiment_name="model-monitoring",
)
```

Drift uses comparison labels such as `benchmark` and `previous_window` in metric namespaces. Performance and bias trend through MLflow over time as each scheduled run logs another result. Run performance when labels arrive, and run bias only when protected attributes are available and permitted for the monitoring purpose.

For the scheduled production version, use `monitoring_report()` to run the same monitoring shape as one bundle:

```python
from lumosai import monitoring_report

run = monitoring_report(
    current_window_with_labels,
    benchmark=train_benchmark,
    previous_window=previous_window,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    protected_attribute=["region", "segment"],
    feature_columns=feature_columns,
    temporal_features=["event_date"],
    report_name="Churn Monitoring",
    experiment_name="model-monitoring",
)
```

The bundle returns a grouped `LumosRun` and logs one combined run artifact when MLflow dictionary logging is enabled.
