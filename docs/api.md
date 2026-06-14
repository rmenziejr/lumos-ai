# API Reference

## `LumosResult`

All report functions return `LumosResult`.

Fields:

- `metrics`: numeric metrics suitable for MLflow logging.
- `summary`: JSON-like structured report details.
- `flagged`: threshold or disparity findings.
- `artifacts`: local paths or external artifact references.
- `report`: underlying report object when applicable.
- `metadata`: report type, task type, comparison name, report name, and schema metadata.

Use `result.to_dict()` for JSON-safe output.

## Shared Report Parameters

Most report functions accept:

- `report_name`: a human-readable report name. Stored in `result.metadata["report_name"]` and passed to supported engines as a title/name.
- `feature_columns`: columns to treat as model features or drift/profile analysis columns.
- `categorical_columns`: semantic categorical overrides for columns such as numeric-coded `day_of_week`, `month`, `store_id`, or `zip3`.

For model reports, `target` is the outcome column. For profiling, `target` is also the outcome column and is moved to the first profiled column.

## Data APIs

### `profile(...)`

```python
profile(
    df,
    target=None,
    feature_columns=None,
    categorical_columns=None,
    time_column=None,
    freq="M",
    sample_size=None,
    min_per_period=1,
    minimal=None,
    log_analysis=None,
    report_name=None,
    ydata_kwargs=None,
    experiment_name=None,
)
```

Generates a ydata-profiling report.

- Accepts Narwhals-compatible dataframe inputs.
- Uses full data when `time_column` is omitted.
- Uses temporal sampling when `time_column` is provided.
- Profiles only `target + feature_columns` when `feature_columns` is provided.
- Moves `target` to the first profiled column when provided.
- Returns an HTML artifact when analysis logging is enabled.

Supported `ydata_kwargs`:

- `explorative`
- `dark_mode`
- `config_file`
- `vars`
- `sort`
- `sensitive`

`title` and `minimal` are not accepted in `ydata_kwargs`; use `report_name` and `minimal` instead.

### `temporal_sample(...)`

```python
temporal_sample(df, time_column, freq="M", sample_size=1000, min_per_period=1)
```

Samples rows per time period.

- Requires a valid non-null timestamp column.
- Rejects `sample_size < 1` and `min_per_period < 1`.
- Returns a pandas dataframe.

### `drift_report(...)`

```python
drift_report(
    reference,
    current,
    temporal_features,
    feature_columns=None,
    categorical_columns=None,
    column_mapping=None,
    comparison="benchmark",
    report_name=None,
    evidently_kwargs=None,
    experiment_name=None,
)
```

Runs an Evidently data drift report.

- Excludes `temporal_features` from drift calculations.
- Runs only over `feature_columns` when provided.
- Treats `categorical_columns` as semantic categorical overrides where the installed Evidently API supports it.
- Namespaces metrics under `drift/<comparison>/...`.
- Flags when drift share exceeds `settings.data.drift_share_threshold`.
- Supports current Evidently APIs and legacy report payloads.

Supported `evidently_kwargs`:

```python
{
    "preset": {
        "drift_share": 0.4,
        "num_threshold": 0.01,
    },
    "report": {
        "tags": ["production"],
    },
}
```

Allowed `preset` keys:

- `columns`
- `drift_share`
- `method`
- `cat_method`
- `num_method`
- `text_method`
- `threshold`
- `cat_threshold`
- `num_threshold`
- `text_threshold`
- `per_column_threshold`

Allowed `report` keys:

- `metadata`
- `tags`
- `include_tests`
- `model_id`
- `reference_id`
- `batch_size`
- `dataset_id`

When `feature_columns` is provided, `evidently_kwargs["preset"]["columns"]` must match it.

## Model APIs

### `get_metrics(...)`

```python
get_metrics(y_true, y_pred, y_score=None, task_type=None, custom_metrics=None)
```

Computes classification or regression metrics.

- Auto-detects task type when not provided.
- Supports classification metrics such as accuracy, precision, recall, F1, and ROC AUC when scores are supplied.
- Supports regression metrics such as MAE, RMSE, and R2.

### `performance_report(...)`

```python
performance_report(
    current,
    target,
    prediction,
    prediction_score=None,
    task_type=None,
    custom_metrics=None,
    report_name=None,
    feature_columns=None,
    categorical_columns=None,
    experiment_name=None,
)
```

Computes current-window model performance.

- `target` is the outcome column.
- `prediction` is the predicted label or value column.
- `prediction_score` is an optional score/probability column.
- Returns namespaced metrics under `performance/...`.
- Stores `feature_columns` and `categorical_columns` in metadata when provided.

### `bias_report(...)`

```python
bias_report(
    current,
    target,
    prediction,
    protected_attribute,
    prediction_score=None,
    task_type=None,
    custom_metrics=None,
    report_name=None,
    feature_columns=None,
    categorical_columns=None,
    experiment_name=None,
)
```

Computes group-wise model behavior across protected attributes.

- `target` is the outcome column.
- `protected_attribute` can be a list of columns or a mapping of column names to numeric bins.
- Classification reports include positive prediction rate for binary labels.
- Regression reports include residual and error summaries.
- Missing and out-of-bin protected values are retained as explicit groups.
- Stores `feature_columns` and `categorical_columns` in metadata when provided.

## Settings

Import global settings:

```python
from lumosai import settings
```

Settings use nested Pydantic models and environment variables with the `LUMOSAI_` prefix.

Examples:

- `LUMOSAI_MLFLOW__TRACKING_URI`
- `LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME`
- `LUMOSAI_ARTIFACTS__KEEP_LOCAL`
- `LUMOSAI_DATA__DRIFT_SHARE_THRESHOLD`
- `LUMOSAI_MODEL__METRIC_THRESHOLDS__RMSE__VALUE`

The main settings groups are:

- `settings.artifacts`
- `settings.mlflow`
- `settings.data`
- `settings.model`
