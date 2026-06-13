# lumosai Package Design

## Purpose

`lumosai` is an opinionated Python package for ML monitoring and reporting. It uses established libraries such as Evidently and ydata-profiling underneath, but its value is not being a faster wrapper. Its value is a smaller, consistent workflow layer for common ML monitoring jobs:

- accept common dataframe objects through Narwhals;
- expose simple function-first APIs;
- return stable structured results;
- add bias and fairness summaries that are not just pass-through Evidently reports;
- make MLflow logging predictable;
- document how to fit the functions into data, training, and ongoing monitoring pipelines.

The package does not train models, tune hyperparameters, orchestrate pipelines, manage monitoring windows, or own late-arriving label joins.

## Scope

### v0.1

The first release includes:

- package scaffold using `uv`, `src/` layout, pytest, and ruff;
- required Narwhals dataframe ingestion with pandas as the internal execution type;
- nested Pydantic settings using `pydantic-settings`;
- a shared `LumosResult` dataclass return type;
- optional MLflow logging with automatic run handling by default;
- local and temporary artifact handling;
- data functions: `profile()` and `drift_report()`;
- model functions: `get_metrics()`, `performance_report()`, and `bias_report()`;
- documentation recipes for data pipeline monitoring, training pipeline reporting, ongoing monitoring, and MLflow layout.

### v0.2 / Future

Later releases may add:

- `feature_importance()` using permutation importance and SHAP;
- `build_sample()` for representative benchmark and holdout samples;
- optional dataset metadata logging and explicit sample artifact logging;
- richer alerting helpers if real usage shows a need.

The package should not add a monitoring window manager unless repeated user workflows justify it.

## Package Structure

The package is organized around product concepts rather than one file per function.

```text
src/lumosai/
  __init__.py
  settings.py
  mlflow.py
  results.py
  exceptions.py

  data/
    __init__.py
    ingest.py
    validation.py
    profiling.py
    drift.py

  model/
    __init__.py
    validation.py
    metrics.py
    performance.py
    bias.py
```

`lumosai.data` owns dataframe normalization, profiling, drift checks, temporal exclusions, and data validation.

`lumosai.model` owns task detection, model-output metrics, performance summaries, and bias/fairness reporting.

`lumosai.mlflow` owns optional MLflow imports, connection setup, run handling, metric logging, artifact logging, and optional dependency errors.

`lumosai.settings` owns runtime defaults and environment variable parsing.

## Public API

Key methods are available both at the package top level and through domain modules.

```python
import lumosai as la

la.profile(...)
la.drift_report(...)
la.performance_report(...)
la.bias_report(...)
la.get_metrics(...)
```

Domain imports are also supported:

```python
from lumosai.data import profile, drift_report
from lumosai.model import performance_report, bias_report, get_metrics
```

Helper APIs such as dataframe normalization and task detection stay domain-scoped:

```python
from lumosai.data.ingest import to_pandas
from lumosai.model.metrics import detect_task_type
```

## Dependencies

Core dependencies:

- `pandas`
- `narwhals`
- `pydantic`
- `pydantic-settings`
- `ydata-profiling`
- `evidently`
- `scikit-learn`
- `matplotlib`

Optional dependency groups:

```toml
[project.optional-dependencies]
mlflow = ["mlflow>=2.18"]
importance = ["shap"]
dev = ["pytest", "ruff", "mypy"]
```

`scikit-learn` remains core for v0.1 because metric computation is part of the core package. SHAP is not core because feature importance is deferred to v0.2.

## Dataframe Inputs

Public functions accept Narwhals-compatible dataframe inputs and convert them to pandas at function boundaries.

```python
from narwhals.typing import IntoDataFrame

def drift_report(reference: IntoDataFrame, current: IntoDataFrame, ...):
    reference_pd = to_pandas(reference)
    current_pd = to_pandas(current)
```

Internal execution uses pandas. This keeps the implementation compatible with ydata-profiling, Evidently, scikit-learn, matplotlib, and MLflow.

Documentation must state that conversion can collect lazy or distributed data into memory. For large Spark, Dask, DuckDB, or warehouse-backed workflows, callers should pass bounded materialized monitoring frames or samples.

## Settings

Settings use nested `pydantic-settings` models with the `LUMOSAI_` prefix and `__` nested delimiter.

Example environment variables:

```bash
LUMOSAI_MLFLOW__TRACKING_URI=http://localhost:5000
LUMOSAI_MLFLOW__RUN_MODE=auto
LUMOSAI_ARTIFACTS__KEEP_LOCAL=true
LUMOSAI_ARTIFACTS__LOCAL_DIR=./lumosai-artifacts
LUMOSAI_MODEL__SHAP_SAMPLE_SIZE=500
```

Settings model shape:

```python
class ArtifactSettings(BaseModel):
    keep_local: bool = False
    local_dir: Path | None = None


class MLflowSettings(BaseModel):
    tracking_uri: str | None = None
    username: str | None = None
    password: str | None = None
    default_experiment_name: str | None = None
    run_mode: Literal["auto", "require_active"] = "auto"
    log_artifacts: bool = True
    log_dicts: bool = True


class DataSettings(BaseModel):
    drift_share_threshold: float = 0.1
    profile_minimal_default: bool = True
    log_analysis: bool = True


class MetricThreshold(BaseModel):
    mode: Literal["relative", "absolute"] = "relative"
    value: float
    greater_is_better: bool = True


class ModelSettings(BaseModel):
    classification_metrics: list[str] = ["accuracy", "precision", "recall", "f1"]
    classification_probability_metrics: list[str] = ["roc_auc"]
    regression_metrics: list[str] = ["mae", "rmse", "r2"]
    metric_thresholds: dict[str, MetricThreshold] = {
        "precision": MetricThreshold(mode="relative", value=0.8, greater_is_better=True),
        "recall": MetricThreshold(mode="relative", value=0.8, greater_is_better=True),
        "f1": MetricThreshold(mode="relative", value=0.8, greater_is_better=True),
        "positive_prediction_rate": MetricThreshold(
            mode="relative",
            value=0.8,
            greater_is_better=True,
        ),
        "mae": MetricThreshold(mode="relative", value=1.25, greater_is_better=False),
        "rmse": MetricThreshold(mode="relative", value=1.25, greater_is_better=False),
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMOSAI_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    artifacts: ArtifactSettings = ArtifactSettings()
    mlflow: MLflowSettings = MLflowSettings()
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
```

Metric thresholds are metric-specific. A single global bias threshold is not expressive enough because some metrics are higher-is-better, some are lower-is-better, and some require absolute rather than relative comparisons.

## Result Type

All public report functions return `LumosResult`.

```python
@dataclass
class LumosResult:
    metrics: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    report: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "summary": self.summary,
            "flagged": self.flagged,
            "artifacts": json_safe_artifacts(self.artifacts),
            "metadata": self.metadata,
        }
```

`metrics`, `summary`, `flagged`, and `metadata` must be JSON-safe. `report` may contain rich objects such as Evidently reports or ydata-profiling reports. `artifacts` may contain local paths, MLflow artifact metadata, or in-memory figures, but `to_dict()` must expose only JSON-safe artifact metadata.

## MLflow Behavior

Every report function accepts `experiment_name: str | None = None`.

Resolution:

1. If `experiment_name` is provided, use it.
2. Else use `settings.mlflow.default_experiment_name`.
3. If neither resolves, do not log.

If logging is requested and MLflow is not installed, raise a clear optional dependency error.

Run behavior:

- If an MLflow run is active, log into it.
- If no run is active and `settings.mlflow.run_mode == "auto"`, start and close a managed run.
- If no run is active and `settings.mlflow.run_mode == "require_active"`, raise a clear error.

Artifact behavior:

- If logging to MLflow and `settings.artifacts.keep_local` is `False`, write artifacts to a temporary directory and clean it up after logging.
- If logging to MLflow and `settings.artifacts.keep_local` is `True`, write artifacts under `settings.artifacts.local_dir` or `./lumosai-artifacts`.
- If not logging to MLflow, preserve local artifacts and return their paths in `result.artifacts`.

MLflow metric namespaces:

- performance metrics use `performance/<metric>`;
- bias metrics use `bias/...`;
- drift metrics use `drift/<comparison>/...`.

Performance and bias do not use reference/current names. Trending belongs to MLflow runs over time. Drift needs an explicit `comparison` label because a single monitoring run may compare the current window against both a benchmark and a previous window.

## Functions

### `profile()`

```python
profile(
    df,
    time_column: str | None = None,
    freq: str = "M",
    sample_size: int | None = None,
    min_per_period: int = 1,
    minimal: bool | None = None,
    log_analysis: bool | None = None,
    experiment_name: str | None = None,
) -> LumosResult
```

If `time_column` is not provided, profile the full dataframe. `minimal` defaults to `settings.data.profile_minimal_default`.

If `time_column` is provided, create a temporal sample before profiling. The sample should represent each time bucket instead of using a flat random sample that can miss sparse periods. Temporal samples are useful for processed-data exploration, but temporal columns themselves should not be treated as drift-comparable features by default.

The result includes the profile report object in `report`, sample metadata in `summary`, and HTML artifact metadata in `artifacts`.

### `drift_report()`

```python
drift_report(
    reference,
    current,
    temporal_features: list[str],
    column_mapping=None,
    comparison: str = "benchmark",
    experiment_name: str | None = None,
) -> LumosResult
```

`temporal_features` is required and explicit. Temporal columns drift by construction between time windows and should be excluded from ordinary data drift comparisons. Pass an empty list when there are none.

`comparison` is used only for metric and artifact namespacing. Examples:

- `benchmark`
- `previous_window`
- `previous_day`
- `train_baseline`

Metrics are logged as:

```text
drift/benchmark/share_drifted_columns
drift/benchmark/n_drifted_columns
drift/previous_window/share_drifted_columns
```

Artifacts are logged under:

```text
drift/benchmark/evidently_report.html
drift/previous_window/evidently_report.html
```

The result includes the underlying Evidently report object, a JSON-safe drift summary, and flagged drift information.

### `get_metrics()`

```python
get_metrics(
    y_true,
    y_pred,
    y_score=None,
    task_type: Literal["classification", "regression"] | None = None,
    custom_metrics: list[tuple[str, Callable]] | None = None,
) -> dict[str, float]
```

`get_metrics()` is pure. It has no MLflow logging, report generation, or artifact output.

Classification metrics that require labels use `y_pred`. Probability metrics such as ROC AUC use `y_score`. This distinction must be explicit so score-based metrics are not computed accidentally from hard labels.

Task type can be auto-detected, but callers can override it.

### `performance_report()`

```python
performance_report(
    current,
    target: str,
    prediction: str,
    prediction_score: str | None = None,
    task_type: Literal["classification", "regression"] | None = None,
    custom_metrics: list[tuple[str, Callable]] | None = None,
    experiment_name: str | None = None,
) -> LumosResult
```

`performance_report()` evaluates the current dataframe. It does not need a reference dataframe in v0.1. MLflow run history provides trends over time.

The function computes configured metrics through `get_metrics()`, optionally builds an Evidently report, logs namespaced metrics and artifacts, and returns `LumosResult`.

### `bias_report()`

```python
bias_report(
    current,
    target: str,
    prediction: str,
    protected_attribute: list[str] | dict[str, list[float] | None],
    prediction_score: str | None = None,
    task_type: Literal["classification", "regression"] | None = None,
    custom_metrics: list[tuple[str, Callable]] | None = None,
    experiment_name: str | None = None,
) -> LumosResult
```

`bias_report()` is the main differentiator for v0.1. It computes group-wise metrics and cross-group disparity comparisons.

Protected attributes can be:

- a list of categorical column names;
- a dict of column name to bins, where bins are passed to `pd.cut`;
- a dict value of `None` for already-categorical columns.

Each protected attribute is analyzed independently for v0.1. Intersectional fairness is future scope.

For classification, compute group metrics such as precision, recall, F1, and positive prediction rate. For regression, compute residual metrics such as mean residual, mean absolute residual, and RMSE.

Flagging uses metric-specific thresholds from settings. Higher-is-better and lower-is-better metrics must be handled differently. Relative and absolute threshold modes are both supported.

## Sampling And Dataset Logging

Sampling helpers are deferred to v0.2, but the intended direction should be documented in v0.1.

`build_sample()` should eventually support roles such as:

- `drift_benchmark`, usually training data or a stable historical reference;
- `performance_benchmark`, usually a holdout or test set with labels;
- `monitoring_window`, if users choose to build bounded production samples themselves.

The package should not manage windows or schedules. Users and orchestration tools decide what frames represent benchmark, previous, and current periods.

For MLflow, the default recommendation is metadata-only dataset logging. Lumosai should not send raw representative datasets to MLflow by default. It should log dataset name, role, row count, schema, digest, time range, sampling strategy, and storage URI/path. Raw sample artifact logging should require explicit opt-in because MLflow artifact storage may not be approved for sensitive data.

## Documentation Recipes

The documentation should include practical recipes rather than expanding the package into an orchestrator:

- data pipeline monitoring: profile feature tables and run drift checks after feature creation;
- training pipeline reporting: evaluate holdout performance and bias after model scoring;
- ongoing monitoring pipeline: run drift for every production window, run performance when labels arrive, and run bias when protected attributes are available and permitted;
- MLflow layout: recommended experiments, run structure, metric namespaces, and artifact conventions.

The ongoing monitoring recipe should show both benchmark and previous-window drift:

```python
drift_report(
    reference=train_benchmark,
    current=current_window,
    temporal_features=["event_date", "event_month"],
    comparison="benchmark",
    experiment_name="model-monitoring",
)

drift_report(
    reference=previous_window,
    current=current_window,
    temporal_features=["event_date", "event_month"],
    comparison="previous_window",
    experiment_name="model-monitoring",
)
```

Performance and bias are shown as current-window reports:

```python
performance_report(
    current_window,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    experiment_name="model-monitoring",
)

bias_report(
    current_window,
    target="actual",
    prediction="prediction",
    protected_attribute=["gender", "region"],
    experiment_name="model-monitoring",
)
```

## Testing Strategy

Tests should cover:

- settings parsing from nested environment variables;
- Narwhals conversion with pandas and at least one alternate dataframe library in optional tests;
- validation errors for missing columns, empty frames, duplicate columns, bad bins, and invalid temporal features;
- task detection and metric computation;
- explicit `y_pred` versus `y_score` handling;
- metric threshold comparison behavior for higher-is-better, lower-is-better, relative, and absolute thresholds;
- bias grouping and flagging with synthetic datasets;
- JSON safety of `LumosResult.to_dict()`;
- MLflow behavior with no experiment, active run, managed auto run, and missing optional dependency.

Heavy Evidently and ydata-profiling integration tests can be marked separately if runtime becomes expensive.

## Open Implementation Risks

- Evidently APIs have changed across versions, so the implementation must pin and target a specific supported version range.
- ydata-profiling can be expensive on large frames, so defaults should stay conservative.
- Narwhals conversion can materialize lazy/distributed frames in memory, so docs and validation should be clear.
- Classification score semantics must be explicit. Hard labels and probabilities should not be conflated.
- MLflow dataset APIs should be revisited before v0.2 sample logging implementation.
