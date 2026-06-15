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

## `LumosRun`

Bundle functions return `LumosRun`.

Fields:

- `run_type`: logical bundle type, such as `"monitoring"`.
- `results`: child `LumosResult` objects keyed by stable report names.
- `metadata`: bundle-level details such as `report_name` and skipped reports.

Properties and methods:

- `run.metrics`: merged metrics from all child results.
- `run.flagged`: flagged findings from all child results, annotated with `result_key`.
- `run.to_dict()`: JSON-safe grouped output for logging, APIs, or persisted summaries.

## Shared Report Parameters

Most report functions accept:

- `report_name`: a human-readable report name. Stored in `result.metadata["report_name"]` and passed to supported engines as a title/name.
- `feature_columns`: columns to treat as model features or drift/profile analysis columns.
- `categorical_columns`: semantic categorical overrides for columns such as numeric-coded `day_of_week`, `month`, `store_id`, or `zip3`.

For model reports, `target` is the outcome column. For profiling, `target` is also the outcome column and is moved to the first profiled column.

`categorical_columns` is a semantic override, not a dataframe type conversion. Use it when a column's storage type is numeric or string-like but the report should treat it as categorical.

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
- `log_analysis=False` disables profile artifact generation and MLflow logging for that call, even when an experiment is configured.

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

### `build_sample(...)`

```python
build_sample(
    data,
    *,
    role,
    sample_size=None,
    strategy="auto",
    target=None,
    prediction=None,
    feature_columns=None,
    categorical_columns=None,
    time_column=None,
    temporal_columns=None,
    stratify_by=None,
    random_state=42,
    artifact_path=None,
    experiment_name=None,
    log_metadata=None,
    log_artifact=None,
)
```

Builds a bounded representative sample for training benchmarks, holdouts, or monitoring windows.

- `role` must be `"train_benchmark"`, `"holdout"`, or `"monitoring_window"`.
- `strategy` may be `"auto"`, `"random"`, `"stratified"`, `"temporal_recent"`, or `"temporal_bucket"`.
- `feature_columns` selects model features; `target`, `prediction`, and `time_column` are included when required by the sample role.
- For `role="train_benchmark"`, `time_column` and `temporal_columns` are validated when provided but excluded from the sampled output.
- `categorical_columns` marks semantic categorical features and must be included in the sampled columns.
- Returns the sampled pandas dataframe as `result.artifacts["sample"]`.
- Adds sample role, strategy, row counts, columns, schema summary, digest, categorical columns, and optional time range to metadata/summary.
- Writes a local artifact when `artifact_path` is provided.
- An explicit `artifact_path` suffix controls the format; suffixless paths use `settings.data.sample_artifact_format`.
- When `experiment_name` is provided, or `settings.mlflow.default_experiment_name` is configured, `log_metadata=None` defers to `settings.data.log_sample_metadata`, which defaults to `True`.
- Raw sample artifact logging is opt-in: `log_artifact=None` defers to `settings.data.log_sample_artifacts`, which defaults to `False`; pass `log_artifact=True` to log the raw artifact for that call.

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

## Bundle APIs

### `monitoring_report(...)`

```python
monitoring_report(
    current,
    *,
    benchmark,
    previous_window=None,
    target=None,
    prediction=None,
    prediction_score=None,
    feature_columns=None,
    categorical_columns=None,
    protected_attribute=None,
    temporal_features=None,
    time_column=None,
    sample_size=None,
    include_performance=None,
    include_bias=None,
    report_name=None,
    experiment_name=None,
    loaded_settings=settings,
)
```

Builds the standard production monitoring bundle and returns a `LumosRun`.

- Always builds a `monitoring_window` sample from `current`.
- Always runs benchmark drift from `benchmark` to `current` as `drift_benchmark`.
- Runs previous-window drift as `drift_previous_window` when `previous_window` is provided and `settings.bundles.include_previous_window_drift` is `True`.
- Runs performance when both `target` and `prediction` are provided, or when `include_performance=True`.
- Runs bias when `protected_attribute` is provided, or when `include_bias=True`.
- Uses `prediction_score` for supported performance and bias metrics when provided.
- Uses `feature_columns` as drift and report analysis columns. Pass temporal columns with `temporal_features` or `time_column`, not in `feature_columns`.
- Uses `categorical_columns` as semantic categorical overrides and requires them to be included in `feature_columns` when `feature_columns` is provided.
- Uses `sample_size` for the monitoring window sample.
- Uses `report_name` as the base display name for child reports.
- Uses `experiment_name`, or `loaded_settings.mlflow.default_experiment_name`, to log the whole bundle in one MLflow run.
- Uses `loaded_settings` for bundle and MLflow settings. By default this is the global `settings` object.

Fail-fast validation runs before report generation. The bundle validates required input columns, temporal drift columns, performance inputs, and bias inputs before starting expensive report work. `monitoring_report()` currently requires `loaded_settings.bundles.fail_fast=True`.

### `training_report(...)`

```python
training_report(
    train,
    holdout,
    *,
    target,
    prediction=None,
    prediction_score=None,
    model=None,
    feature_columns=None,
    categorical_columns=None,
    protected_attribute=None,
    time_column=None,
    sample_size=None,
    include_profile=None,
    include_performance=None,
    include_bias=None,
    include_feature_importance=None,
    report_name=None,
    experiment_name=None,
    loaded_settings=settings,
)
```

Builds the standard post-training bundle and returns a `LumosRun`.

- Always builds `train_sample` from `train` with `role="train_benchmark"`.
- Always builds `holdout_sample` from `holdout` with `role="holdout"`.
- Runs profile as `profile` only when `include_profile=True`.
- Runs performance as `performance` when `prediction` is provided, or when `include_performance=True`.
- Runs bias as `bias` when `protected_attribute` is provided, or when `include_bias=True`.
- Runs feature importance as `feature_importance` when `model` is provided and `include_feature_importance` is not `False`, unless `settings.bundles.include_feature_importance_in_training` disables the default.
- Uses `sample_size` for train and holdout samples, and for profile temporal sampling when profile is enabled.
- Uses `experiment_name`, or `loaded_settings.mlflow.default_experiment_name`, to log the whole bundle in one MLflow run.

Fail-fast validation runs before report generation. The bundle validates required target, prediction, bias, and feature-importance inputs before starting expensive report work. `training_report()` currently requires `loaded_settings.bundles.fail_fast=True`.

## Model APIs

### `get_metrics(...)`

```python
get_metrics(
    y_true,
    y_pred,
    y_score=None,
    score_labels=None,
    task_type=None,
    custom_metrics=None,
)
```

Computes classification or regression metrics.

- Auto-detects task type when not provided.
- Supports classification metrics such as accuracy, precision, recall, F1, ROC AUC, and log loss when probability-like scores are supplied.
- `score_labels` defines probability order for binary or multiclass scores. For 1D binary scores, the score is interpreted as the probability of `score_labels[-1]`.
- Supports regression metrics such as MAE, RMSE, and R2.

### `performance_report(...)`

```python
performance_report(
    current,
    target,
    prediction,
    prediction_score=None,
    score_labels=None,
    task_type=None,
    custom_metrics=None,
    include_lift=None,
    report_name=None,
    feature_columns=None,
    categorical_columns=None,
    experiment_name=None,
)
```

Computes current-window model performance.

- `target` is the outcome column.
- `prediction` is the predicted label or value column.
- `prediction_score` is an optional score/probability column, a column of probability arrays, or a mapping of labels to probability columns.
- `score_labels` defines probability order for binary or multiclass arrays. Pass `list(model.classes_)` for sklearn-style classifiers.
- When multiclass array scores omit `score_labels`, labels are inferred by sorting observed target/prediction labels and warning metadata is recorded.
- Classification reports include log loss when probability-like scores are supplied.
- Pass `include_lift=True` to add decile lift metrics under `performance/lift/<class>/...`.
- Returns namespaced metrics under `performance/...`.
- Stores `feature_columns` and `categorical_columns` in metadata when provided.

### `calibration_report(...)`

```python
calibration_report(
    current,
    target,
    prediction_score,
    *,
    score_labels=None,
    n_bins=10,
    strategy="uniform",
    report_name=None,
    experiment_name=None,
)
```

Computes probability calibration for classification models.

- Accepts sklearn-style probability arrays through a `prediction_score` column.
- Also accepts `prediction_score={"label": "probability_column"}` mappings.
- `score_labels` defines probability order. Pass `list(model.classes_)` for sklearn-style classifiers.
- Binary calibration evaluates the positive class.
- Multiclass calibration runs one-vs-rest calibration per class.
- Returns Brier score and expected calibration error metrics under `calibration/<class>/...`.
- Returns macro calibration metrics under `calibration/macro_brier` and `calibration/macro_ece`.
- Stores bin tables in `result.summary["calibration"]`.
- Uses `report_name` in result metadata and MLflow logging when provided.

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

### `feature_importance(...)`

```python
feature_importance(
    model,
    data,
    *,
    target,
    feature_columns,
    method="permutation",
    scoring=None,
    n_repeats=5,
    sample_size=None,
    random_state=42,
    report_name=None,
    experiment_name=None,
)
```

Computes model feature importance after training or evaluation.

- `method="permutation"` is the default and uses scikit-learn permutation importance.
- `method="shap"` computes SHAP mean absolute importance and requires the optional `lumosai[importance]` dependency.
- `feature_columns` must contain at least one feature and all features must exist in `data`.
- `sample_size` optionally samples rows before computing importance.
- `scoring`, `n_repeats`, and `random_state` apply to permutation importance.
- Returns metrics under `importance/<feature>` and sorted feature rows in `result.summary["features"]`.
- Stores method, feature columns, and optional `report_name` in metadata.

## Settings

Import global settings:

```python
from lumosai import settings
```

The settings API is the package control point for repeated use. Put shared standards in environment variables once, such as MLflow destinations, artifact retention, sample sizes, metric thresholds, and bundle behavior, then keep individual report calls focused on the data and columns for that run.

Relevant sample defaults live under `settings.data`:

- `default_sample_size`: default sample size for temporal bucket sampling.
- `sample_artifact_format`: format for suffixless sample artifact paths, either `"parquet"` or `"csv"`.
- `log_sample_metadata`: default metadata logging behavior for `build_sample()`.
- `log_sample_artifacts`: default raw sample artifact logging behavior for `build_sample()`.

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
- `settings.bundles`
