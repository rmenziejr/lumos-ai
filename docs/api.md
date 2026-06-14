# API Reference

## `LumosResult`

All report functions return `LumosResult`.

Fields:

- `metrics`: numeric metrics suitable for MLflow logging.
- `summary`: JSON-like structured report details.
- `flagged`: threshold or disparity findings.
- `artifacts`: local paths or external artifact references.
- `report`: underlying report object when applicable.
- `metadata`: report type, task type, comparison name, and logging metadata.

Use `result.to_dict()` for JSON-safe output.

## Data APIs

### `profile(df, time_column=None, freq="M", sample_size=None, min_per_period=1, minimal=None, log_analysis=None, experiment_name=None)`

Generates a ydata-profiling report.

- Accepts Narwhals-compatible dataframe inputs.
- Uses full data when `time_column` is omitted.
- Uses temporal sampling when `time_column` is provided.
- Returns an HTML artifact when analysis logging is enabled.

### `temporal_sample(df, time_column, freq="M", sample_size=1000, min_per_period=1)`

Samples rows per time period.

- Requires a valid non-null timestamp column.
- Rejects `sample_size < 1` and `min_per_period < 1`.
- Returns a pandas dataframe.

### `drift_report(reference, current, temporal_features, column_mapping=None, comparison="benchmark", experiment_name=None)`

Runs an Evidently data drift report.

- Excludes `temporal_features` from drift calculations.
- Namespaces metrics under `drift/<comparison>/...`.
- Flags when drift share exceeds `settings.data.drift_share_threshold`.
- Supports current Evidently APIs and legacy report payloads.

## Model APIs

### `get_metrics(y_true, y_pred, y_score=None, task_type=None, custom_metrics=None)`

Computes classification or regression metrics.

- Auto-detects task type when not provided.
- Supports classification metrics such as accuracy, precision, recall, F1, and ROC AUC when scores are supplied.
- Supports regression metrics such as MAE, RMSE, and R2.

### `performance_report(current, target, prediction, prediction_score=None, task_type=None, custom_metrics=None, experiment_name=None)`

Computes current-window model performance.

- Validates target and prediction columns.
- Returns namespaced metrics under `performance/...`.
- Flags metrics based on configured thresholds.

### `bias_report(current, target, prediction, protected_attribute, prediction_score=None, task_type=None, custom_metrics=None, experiment_name=None)`

Computes group-wise model behavior across protected attributes.

- `protected_attribute` can be a list of columns or a mapping of column names to numeric bins.
- Classification reports include positive prediction rate for binary labels.
- Regression reports include residual and error summaries.
- Missing and out-of-bin protected values are retained as explicit groups.

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
