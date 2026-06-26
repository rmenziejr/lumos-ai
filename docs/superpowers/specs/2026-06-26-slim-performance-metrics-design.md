# Slim Performance Metrics Design

## Goal

Allow users to run lightweight performance reporting for fold validation, tuning, and other repeated evaluation loops. These runs should log only the metrics the user chooses, optionally with an MLflow step such as the fold index, while avoiding repeated HTML and JSON artifacts unless explicitly requested.

## User Experience

Fold validation should be concise:

```python
performance_report(
    fold_scored,
    target=target,
    prediction="prediction",
    prediction_score="prediction_score",
    score_labels=list(model.classes_),
    metrics=["f1", "roc_auc", "pr_auc"],
    profile="metrics_only",
    mlflow_step=fold_index,
    report_name="Fold Validation",
    experiment_name=EXPERIMENT_NAME,
)
```

MLflow receives stable metric names such as `performance/f1`, `performance/roc_auc`, and `performance/pr_auc` with `step=fold_index`. This lets users chart fold behavior without creating one large artifact set per fold.

## Public Metric Types

Expose typed metric aliases and constants from `lumosai.model.metrics`:

```python
ClassificationMetric = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "log_loss",
]

RegressionMetric = Literal["mae", "rmse", "r2"]
PerformanceMetric = ClassificationMetric | RegressionMetric
MetricPreset = Literal["default", "all"]
```

Runtime constants should mirror these types:

- `CLASSIFICATION_METRICS`
- `CLASSIFICATION_PROBABILITY_METRICS`
- `REGRESSION_METRICS`
- `PERFORMANCE_METRICS`

The type aliases give users autocomplete and static checking. Runtime validation still protects notebook users and dynamically typed code.

## `metrics` Argument

Add a `metrics` argument to `get_metrics()` and `performance_report()`:

```python
metrics: MetricPreset | list[PerformanceMetric] = "default"
```

Behavior:

- `metrics="default"` uses the existing settings defaults:
  - `settings.model.classification_metrics`
  - `settings.model.classification_probability_metrics`
  - `settings.model.regression_metrics`
- `metrics="all"` computes all supported built-in metrics for the resolved task.
- `metrics=[...]` computes exactly those built-in metrics.
- `metrics=[]` is valid when the user only wants `custom_metrics`.
- Unsupported metric names raise `LumosValidationError`.
- Task-mismatched metrics raise `LumosValidationError`.
- Score-required metrics raise `LumosValidationError` when `prediction_score` is missing.

Score-required classification metrics are:

- `roc_auc`
- `pr_auc`
- `log_loss`

Custom metrics remain separate:

```python
performance_report(
    scored,
    target="actual",
    prediction="prediction",
    metrics=["f1"],
    custom_metrics=[("business_value", business_value)],
)
```

Custom metric names must not collide with built-in metrics or with each other. Collisions raise `LumosValidationError`.

## Metrics-Only Profile

Add a reporting profile argument to `performance_report()`:

```python
profile: Literal["standard", "metrics_only"] = "standard"
```

`profile="standard"` preserves current behavior.

`profile="metrics_only"` changes default behavior for that call:

- `include_plots=False`
- `include_lift=False`
- result dictionary artifact logging disabled
- scalar MLflow metric logging remains enabled
- the returned `LumosResult` still includes metrics, summary, metadata, and flags

Explicit arguments override profile defaults. For example, `profile="metrics_only", include_plots=True` still writes HTML.

## MLflow Step Support

Add optional `mlflow_step` support to result logging:

```python
performance_report(..., mlflow_step=fold_index)
```

Thread this through:

- `performance_report()`
- `log_result()`
- `log_result_with_html_artifact()`

When provided, metric logging should call:

```python
mlflow.log_metrics(result.metrics, step=mlflow_step)
```

`mlflow_step` should be stored in `result.metadata["mlflow_step"]` when supplied.

## Metric Namespacing

The current namespacing remains:

- single-split metrics: `performance/<metric>`
- train/holdout metrics: `performance/train/<metric>` and `performance/holdout/<metric>`
- train/holdout comparison metrics: `performance/gap/<metric>` and `performance/ratio/<metric>`

Fold validation should prefer MLflow steps over embedding fold numbers in metric keys. This keeps keys stable and makes MLflow charts more useful.

## Train/Holdout Interaction

When `train` is provided to `performance_report()`, the same `metrics` selection applies to both train and holdout. Gap and ratio metrics are computed only for selected metrics present in both splits.

Examples:

- `metrics=["f1"]` logs `performance/train/f1`, `performance/holdout/f1`, `performance/gap/f1`, and `performance/ratio/f1`.
- `metrics=["roc_auc"]` requires `prediction_score` for both train and holdout.

## Error Handling

Validation should fail before expensive work when possible:

- unknown metric name;
- classification metric requested for regression;
- regression metric requested for classification;
- score metric requested without scores;
- custom metric name collision.

Errors should use `LumosValidationError` with actionable messages that include the invalid metric names.

## Documentation

Update:

- API reference for `performance_report()`, `get_metrics()`, public metric aliases, `metrics`, `profile`, and `mlflow_step`.
- tuning/fold validation recipe to show `profile="metrics_only"`, `metrics=[...]`, and `mlflow_step=fold_index`.
- settings docs to explain that `metrics="default"` reads the existing model metric settings.

## Testing

Add tests for:

- default settings-driven metric selection;
- explicit `metrics=[...]` filtering;
- `metrics="all"`;
- empty built-in metrics with custom metrics;
- invalid metric names;
- task-mismatched metrics;
- score metrics without `prediction_score`;
- custom metric collisions;
- `profile="metrics_only"` suppressing plots and result dictionary logging while still logging scalar metrics;
- `mlflow_step` being passed to `mlflow.log_metrics`;
- train/holdout comparison respecting selected metrics.

## Out Of Scope

This design does not add a dedicated `fold_performance_report()` helper. The primitive `performance_report()` remains the fold-validation surface, with a slim profile and explicit metric selection.

This design does not change metric key names for existing standard reports.
