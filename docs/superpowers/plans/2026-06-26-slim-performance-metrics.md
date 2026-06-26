# Slim Performance Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add typed metric selection, metrics-only performance reporting, and MLflow step logging for fold validation and tuning loops.

**Architecture:** Keep the primitive `performance_report()` as the fold-validation surface. Move built-in metric selection into `lumosai.model.metrics`, thread the selected metrics through train/holdout comparison, and add MLflow logging controls through the existing `log_result()` and `log_result_with_html_artifact()` adapters.

**Tech Stack:** Python 3.12, pandas, scikit-learn metrics, Pydantic settings, MLflow adapter, pytest, Ruff, MkDocs.

---

## File Structure

- `src/lumosai/model/metrics.py`: owns metric name types, runtime constants, metric validation, and built-in metric computation.
- `src/lumosai/model/performance.py`: owns `performance_report()` API behavior, profile defaults, train/holdout metric filtering, and `mlflow_step` metadata.
- `src/lumosai/mlflow.py`: owns result metric logging without HTML artifacts.
- `src/lumosai/artifacts.py`: owns result metric logging when HTML artifacts are also logged.
- `src/lumosai/model/__init__.py`: exports typed metric aliases and constants through the existing lazy module pattern.
- `src/lumosai/__init__.py`: exports typed metric aliases and constants through the existing lazy top-level API pattern.
- `tests/model/test_metrics.py`: covers metric validation and filtered built-in metric computation.
- `tests/model/test_performance.py`: covers filtered performance reporting, metrics-only profile behavior, and train/holdout filtering.
- `tests/test_mlflow.py`: covers `mlflow_step` and dictionary logging controls for non-HTML result logging.
- `tests/test_artifacts.py`: covers `mlflow_step` and dictionary logging controls for HTML result logging.
- `docs/api.md`: documents public metric types, `metrics`, `profile`, and `mlflow_step`.
- `docs/recipes/tuning-and-final-training.md`: updates fold validation examples to use metrics-only fold reporting.

---

### Task 1: Typed Built-In Metric Selection

**Files:**
- Modify: `src/lumosai/model/metrics.py`
- Modify: `tests/model/test_metrics.py`

- [ ] **Step 1: Write failing tests for built-in metric filtering**

Add these tests to `tests/model/test_metrics.py`:

```python
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.metrics import (
    CLASSIFICATION_METRICS,
    CLASSIFICATION_PROBABILITY_METRICS,
    PERFORMANCE_METRICS,
    REGRESSION_METRICS,
    get_metrics,
)


def test_metric_constants_list_supported_names() -> None:
    assert CLASSIFICATION_METRICS == ("accuracy", "precision", "recall", "f1")
    assert CLASSIFICATION_PROBABILITY_METRICS == ("roc_auc", "pr_auc", "log_loss")
    assert REGRESSION_METRICS == ("mae", "rmse", "r2")
    assert "f1" in PERFORMANCE_METRICS
    assert "rmse" in PERFORMANCE_METRICS


def test_get_metrics_filters_classification_metrics() -> None:
    metrics = get_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        y_score=[0.1, 0.9, 0.4, 0.2],
        task_type="classification",
        metrics=["f1", "roc_auc"],
    )

    assert set(metrics) == {"f1", "roc_auc"}
    assert metrics["f1"] < 1.0
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_get_metrics_all_includes_supported_classification_metrics() -> None:
    metrics = get_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        y_score=[0.1, 0.9, 0.4, 0.2],
        task_type="classification",
        metrics="all",
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
    }


def test_get_metrics_filters_regression_metrics() -> None:
    metrics = get_metrics(
        [1.0, 2.0, 3.0],
        [1.1, 1.8, 2.9],
        task_type="regression",
        metrics=["rmse"],
    )

    assert set(metrics) == {"rmse"}
    assert metrics["rmse"] > 0


def test_get_metrics_rejects_unknown_metric() -> None:
    with pytest.raises(LumosValidationError, match="Unsupported metrics: banana"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["banana"])


def test_get_metrics_rejects_task_mismatched_metric() -> None:
    with pytest.raises(LumosValidationError, match="not valid for classification"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["rmse"])


def test_get_metrics_rejects_score_metric_without_scores() -> None:
    with pytest.raises(LumosValidationError, match="require prediction scores"):
        get_metrics([0, 1], [0, 1], task_type="classification", metrics=["roc_auc"])


def test_get_metrics_accepts_empty_builtin_metrics_with_custom_metric() -> None:
    metrics = get_metrics(
        [0, 1],
        [0, 1],
        task_type="classification",
        metrics=[],
        custom_metrics=[("business_value", lambda y_true, y_pred: 42.0)],
    )

    assert metrics == {"business_value": 42.0}


def test_get_metrics_rejects_custom_metric_collisions() -> None:
    with pytest.raises(LumosValidationError, match="Custom metric names collide"):
        get_metrics(
            [0, 1],
            [0, 1],
            task_type="classification",
            metrics=["f1"],
            custom_metrics=[("f1", lambda y_true, y_pred: 1.0)],
        )

    with pytest.raises(LumosValidationError, match="Duplicate custom metric names"):
        get_metrics(
            [0, 1],
            [0, 1],
            task_type="classification",
            metrics=[],
            custom_metrics=[
                ("business_value", lambda y_true, y_pred: 1.0),
                ("business_value", lambda y_true, y_pred: 2.0),
            ],
        )
```

In the existing log-loss-oriented tests in `tests/model/test_metrics.py`, request log-loss explicitly so those tests keep validating probability-loss behavior after `metrics="default"` starts reading settings:

- In `test_get_metrics_skips_log_loss_for_1d_decision_scores_outside_probability_range()`, add the keyword argument `metrics="all"` to the existing metric calculation call.
- In `test_get_metrics_mixed_explicit_score_labels_do_not_raise_raw_type_error()`, add the keyword argument `metrics="all"` to the existing metric calculation call.
- In `test_get_metrics_adds_log_loss_for_binary_probability_matrix()`, add the keyword argument `metrics=["log_loss"]` to the existing metric calculation call.
- In `test_get_metrics_adds_log_loss_for_multiclass_probability_matrix()`, add the keyword argument `metrics=["roc_auc", "pr_auc", "log_loss"]` to the existing metric calculation call.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/model/test_metrics.py -v
```

Expected: FAIL because metric constants, `metrics`, and validation do not exist yet.

- [ ] **Step 3: Implement metric types, constants, and filtering**

In `src/lumosai/model/metrics.py`, add imports:

```python
from typing import Any, Literal, TypeAlias

from lumosai.exceptions import LumosValidationError
```

Define public types and constants after `TaskType`:

```python
ClassificationMetric: TypeAlias = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "log_loss",
]
RegressionMetric: TypeAlias = Literal["mae", "rmse", "r2"]
PerformanceMetric: TypeAlias = ClassificationMetric | RegressionMetric
MetricPreset: TypeAlias = Literal["default", "all"]

CLASSIFICATION_METRICS: tuple[ClassificationMetric, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
)
CLASSIFICATION_PROBABILITY_METRICS: tuple[ClassificationMetric, ...] = (
    "roc_auc",
    "pr_auc",
    "log_loss",
)
REGRESSION_METRICS: tuple[RegressionMetric, ...] = ("mae", "rmse", "r2")
PERFORMANCE_METRICS: tuple[PerformanceMetric, ...] = (
    *CLASSIFICATION_METRICS,
    *CLASSIFICATION_PROBABILITY_METRICS,
    *REGRESSION_METRICS,
)
_SCORE_REQUIRED_METRICS = frozenset(CLASSIFICATION_PROBABILITY_METRICS)
```

Add helpers before `get_metrics()`:

```python
def _settings_default_metrics(task_type: TaskType) -> list[str]:
    if task_type == "classification":
        return [
            *settings.model.classification_metrics,
            *settings.model.classification_probability_metrics,
        ]
    return list(settings.model.regression_metrics)


def _all_metrics(task_type: TaskType) -> list[str]:
    if task_type == "classification":
        return [*CLASSIFICATION_METRICS, *CLASSIFICATION_PROBABILITY_METRICS]
    return list(REGRESSION_METRICS)


def _resolve_metric_names(
    *,
    metrics: MetricPreset | list[PerformanceMetric],
    task_type: TaskType,
    has_scores: bool,
) -> list[str]:
    if metrics == "default":
        requested = _settings_default_metrics(task_type)
    elif metrics == "all":
        requested = _all_metrics(task_type)
    else:
        requested = list(metrics)

    supported = set(PERFORMANCE_METRICS)
    unknown = sorted(set(requested).difference(supported))
    if unknown:
        msg = "Unsupported metrics: " + ", ".join(unknown)
        raise LumosValidationError(msg)

    valid_for_task = set(_all_metrics(task_type))
    mismatched = sorted(set(requested).difference(valid_for_task))
    if mismatched:
        msg = "Metrics are not valid for "
        msg += f"{task_type}: " + ", ".join(mismatched)
        raise LumosValidationError(msg)

    score_required = sorted(set(requested).intersection(_SCORE_REQUIRED_METRICS))
    if score_required and not has_scores:
        msg = "Metrics require prediction scores: " + ", ".join(score_required)
        raise LumosValidationError(msg)

    return requested


def _validate_custom_metrics(
    *,
    requested_metrics: list[str],
    custom_metrics: list[tuple[str, Callable[..., float]]] | None,
) -> None:
    custom_names = [name for name, _metric_func in custom_metrics or []]
    duplicate_custom = sorted(
        name for name in set(custom_names) if custom_names.count(name) > 1
    )
    if duplicate_custom:
        msg = "Duplicate custom metric names: " + ", ".join(duplicate_custom)
        raise LumosValidationError(msg)

    built_in_collisions = sorted(set(custom_names).intersection(PERFORMANCE_METRICS))
    requested_collisions = sorted(set(custom_names).intersection(requested_metrics))
    collisions = sorted(set(built_in_collisions + requested_collisions))
    if collisions:
        msg = "Custom metric names collide with built-in metrics: "
        msg += ", ".join(collisions)
        raise LumosValidationError(msg)
```

Change `get_metrics()` signature:

```python
def get_metrics(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series | None = None,
    score_labels: Sequence[Any] | None = None,
    task_type: TaskType | None = None,
    metrics: MetricPreset | list[PerformanceMetric] = "default",
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
) -> dict[str, float]:
```

Replace the built-in metric body with selected metric computation:

```python
    resolved_task = task_type or detect_task_type(y_true, y_pred)
    requested_metrics = _resolve_metric_names(
        metrics=metrics,
        task_type=resolved_task,
        has_scores=y_score is not None,
    )
    _validate_custom_metrics(
        requested_metrics=requested_metrics,
        custom_metrics=custom_metrics,
    )
    computed: dict[str, float] = {}

    if resolved_task == "classification":
        average = "weighted"
        zero_division = 0
        if "accuracy" in requested_metrics:
            computed["accuracy"] = float(accuracy_score(y_true, y_pred))
        if "precision" in requested_metrics:
            computed["precision"] = float(
                precision_score(y_true, y_pred, average=average, zero_division=zero_division)
            )
        if "recall" in requested_metrics:
            computed["recall"] = float(
                recall_score(y_true, y_pred, average=average, zero_division=zero_division)
            )
        if "f1" in requested_metrics:
            computed["f1"] = float(
                f1_score(y_true, y_pred, average=average, zero_division=zero_division)
            )
        if y_score is not None and "roc_auc" in requested_metrics:
            computed["roc_auc"] = _roc_auc(y_true, y_score, score_labels)
        if y_score is not None and "pr_auc" in requested_metrics:
            computed["pr_auc"] = _pr_auc(y_true, y_score, score_labels)
        if y_score is not None and "log_loss" in requested_metrics:
            log_loss_value = _log_loss(y_true, y_score, score_labels)
            if log_loss_value is not None:
                computed["log_loss"] = log_loss_value
    else:
        if "mae" in requested_metrics:
            computed["mae"] = float(mean_absolute_error(y_true, y_pred))
        if "rmse" in requested_metrics:
            computed["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        if "r2" in requested_metrics:
            computed["r2"] = float(r2_score(y_true, y_pred))

    for name, metric_func in custom_metrics or []:
        computed[name] = float(metric_func(y_true, y_pred))

    return computed
```

- [ ] **Step 4: Run metric tests**

Run:

```bash
uv run pytest tests/model/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit metric filtering**

Run:

```bash
git add src/lumosai/model/metrics.py tests/model/test_metrics.py
git commit -m "Add typed performance metric selection"
```

---

### Task 2: Performance Report Profile And MLflow Step Logging

**Files:**
- Modify: `src/lumosai/model/performance.py`
- Modify: `src/lumosai/mlflow.py`
- Modify: `src/lumosai/artifacts.py`
- Modify: `tests/model/test_performance.py`
- Modify: `tests/test_mlflow.py`
- Modify: `tests/test_artifacts.py`

- [ ] **Step 1: Write failing performance report tests**

Add these tests to `tests/model/test_performance.py`:

```python
def test_performance_report_filters_selected_metrics() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "prediction_score": [0.1, 0.9, 0.4, 0.2],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
        metrics=["f1", "roc_auc"],
        include_plots=False,
    )

    assert set(result.metrics) == {"performance/f1", "performance/roc_auc"}
    assert set(result.summary["metrics"]) == {"f1", "roc_auc"}


def test_performance_report_metrics_only_profile_suppresses_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "prediction_score": [0.1, 0.9, 0.4, 0.2],
        }
    )
    logged: dict[str, object] = {}

    def fake_log_result(result, *, experiment_name=None, loaded_settings=None, log_dict=None, mlflow_step=None):
        logged["log_dict"] = log_dict
        logged["mlflow_step"] = mlflow_step
        result.metadata["logged_to_mlflow"] = True
        return result

    monkeypatch.setattr("lumosai.model.performance.log_result", fake_log_result)

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
        metrics=["f1"],
        profile="metrics_only",
        mlflow_step=2,
        experiment_name="experiment",
    )

    assert result.artifacts == {}
    assert result.metrics == {"performance/f1": result.summary["metrics"]["f1"]}
    assert logged == {"log_dict": False, "mlflow_step": 2}
    assert result.metadata["profile"] == "metrics_only"
    assert result.metadata["mlflow_step"] == 2


def test_performance_report_metrics_only_profile_allows_explicit_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "prediction_score": [0.1, 0.9, 0.4, 0.2],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
        metrics=["f1"],
        profile="metrics_only",
        include_plots=True,
    )

    assert Path(result.artifacts["html"]).exists()
    assert result.metadata["profile"] == "metrics_only"


def test_performance_report_train_comparison_respects_selected_metrics() -> None:
    train = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "prediction_score": [0.1, 0.9, 0.8, 0.2],
        }
    )
    holdout = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "prediction_score": [0.1, 0.9, 0.4, 0.2],
        }
    )

    result = performance_report(
        holdout,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        train=train,
        task_type="classification",
        metrics=["f1"],
        include_plots=False,
    )

    assert set(result.metrics) == {
        "performance/train/f1",
        "performance/holdout/f1",
        "performance/gap/f1",
        "performance/ratio/f1",
    }
    assert set(result.summary["train_metrics"]) == {"f1"}
    assert set(result.summary["holdout_metrics"]) == {"f1"}
```

Update the existing `test_performance_report_adds_train_holdout_gap_metrics()` call in `tests/model/test_performance.py` to request all built-ins because it asserts log-loss gap behavior:

```python
    result = performance_report(
        holdout,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        train=train,
        task_type="classification",
        metrics="all",
        include_plots=False,
    )
```

- [ ] **Step 2: Write failing MLflow step tests**

In `tests/test_mlflow.py`, update the existing `FakeMlflow.log_metrics()` helper to accept `step`:

```python
def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    self.metrics.update(metrics)
    self.metric_step = step
```

Add this test:

```python
def test_log_result_passes_mlflow_step_and_honors_log_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeMlflow()
    monkeypatch.setattr(mlflow_adapter, "require_mlflow", lambda: fake)
    result = LumosResult(metrics={"performance/f1": 0.8})

    logged = log_result(
        result,
        experiment_name="experiment",
        mlflow_step=3,
        log_dict=False,
    )

    assert logged.metadata["logged_to_mlflow"] is True
    assert logged.metadata["mlflow_step"] == 3
    assert fake.metrics == {"performance/f1": 0.8}
    assert fake.metric_step == 3
    assert fake.dicts == {}
```

In `tests/test_artifacts.py`, add these imports:

```python
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from lumosai.artifacts import log_result_with_html_artifact
from lumosai.results import LumosResult
```

Then add this fake helper:

```python
class FakeMlflow:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.metric_step: int | None = None
        self.dicts: list[tuple[dict[str, Any], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.experiment_name: str | None = None

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def active_run(self) -> Any | None:
        return SimpleNamespace(info=SimpleNamespace(run_id="active-run"))

    def start_run(self) -> Any:
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="started-run")))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.update(metrics)
        self.metric_step = step

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.dicts.append((payload, artifact_file))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))
```

Then add this test:

```python
def test_log_result_with_html_artifact_passes_mlflow_step_and_honors_log_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    fake = FakeMlflow()
    monkeypatch.setattr("lumosai.artifacts.require_mlflow", lambda: fake)

    result = LumosResult(metrics={"performance/f1": 0.8})
    logged = log_result_with_html_artifact(
        result,
        html_path=html_path,
        artifact_path="performance",
        experiment_name="experiment",
        mlflow_step=4,
        log_dict=False,
    )

    assert logged.metadata["logged_to_mlflow"] is True
    assert logged.metadata["mlflow_step"] == 4
    assert fake.metrics == {"performance/f1": 0.8}
    assert fake.metric_step == 4
    assert fake.dicts == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/model/test_performance.py tests/test_mlflow.py tests/test_artifacts.py -v
```

Expected: FAIL because `metrics`, `profile`, `mlflow_step`, and `log_dict` are not threaded through yet.

- [ ] **Step 4: Implement logging controls**

In `src/lumosai/mlflow.py`, change `log_result()` signature:

```python
def log_result(
    result: LumosResult,
    *,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
    log_dict: bool | None = None,
    mlflow_step: int | None = None,
) -> LumosResult:
```

Inside the MLflow context, before logging metrics:

```python
        if mlflow_step is not None:
            result.metadata["mlflow_step"] = mlflow_step
        if result.metrics:
            mlflow.log_metrics(result.metrics, step=mlflow_step)
        should_log_dict = loaded_settings.mlflow.log_dicts if log_dict is None else log_dict
        if should_log_dict:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
```

In `src/lumosai/artifacts.py`, change `log_result_with_html_artifact()` signature to include:

```python
    log_dict: bool | None = None,
    mlflow_step: int | None = None,
```

Inside its MLflow logging block, mirror the same behavior:

```python
        if mlflow_step is not None:
            result.metadata["mlflow_step"] = mlflow_step
        if result.metrics:
            mlflow.log_metrics(result.metrics, step=mlflow_step)
        should_log_dict = loaded_settings.mlflow.log_dicts if log_dict is None else log_dict
        if should_log_dict:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
```

- [ ] **Step 5: Implement `performance_report()` profile and metric selection**

In `src/lumosai/model/performance.py`, update imports:

```python
from typing import Any, Literal, cast
```

Update the metrics import:

```python
from lumosai.model.metrics import (
    MetricPreset,
    PerformanceMetric,
    TaskType,
    detect_task_type,
    get_metrics,
)
```

Change `performance_report()` signature:

```python
def performance_report(
    current: Any,
    target: str,
    prediction: str,
    prediction_score: ScoreInput | None = None,
    score_labels: list[Any] | None = None,
    train: Any | None = None,
    task_type: TaskType | None = None,
    metrics: MetricPreset | list[PerformanceMetric] = "default",
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    include_lift: bool | None = None,
    report_name: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    include_plots: bool | None = None,
    include_train_plots: bool = False,
    profile: Literal["standard", "metrics_only"] = "standard",
    mlflow_step: int | None = None,
    log_dict: bool | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
```

At the top of the function after the docstring:

```python
    if profile not in {"standard", "metrics_only"}:
        msg = "profile must be 'standard' or 'metrics_only'"
        raise LumosValidationError(msg)
    if include_plots is None:
        resolved_include_plots = profile != "metrics_only"
    else:
        resolved_include_plots = include_plots

    if include_lift is None:
        resolved_include_lift = False if profile == "metrics_only" else None
    else:
        resolved_include_lift = include_lift

    if log_dict is None:
        resolved_log_dict = False if profile == "metrics_only" else None
    else:
        resolved_log_dict = log_dict
```

Pass selected metrics into both `get_metrics()` calls:

```python
        metrics=metrics,
        custom_metrics=custom_metrics,
```

Change lift branches to use `resolved_include_lift` and `resolved_include_plots`.

Add metadata:

```python
    metadata: dict[str, Any] = {
        "report_type": "performance",
        "task_type": resolved_task,
        "profile": profile,
        "metrics_argument": metrics,
    }
    if mlflow_step is not None:
        metadata["mlflow_step"] = mlflow_step
```

Use `resolved_include_plots` for HTML generation. Pass logging controls into both logging paths:

```python
            return log_result_with_html_artifact(
                result,
                html_path=html_path,
                artifact_path="performance",
                experiment_name=experiment_name,
                log_dict=resolved_log_dict,
                mlflow_step=mlflow_step,
            )
```

And:

```python
    log_result(
        result,
        experiment_name=experiment_name,
        log_dict=resolved_log_dict,
        mlflow_step=mlflow_step,
    )
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/model/test_performance.py tests/test_mlflow.py tests/test_artifacts.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit performance and logging controls**

Run:

```bash
git add src/lumosai/model/performance.py src/lumosai/mlflow.py src/lumosai/artifacts.py tests/model/test_performance.py tests/test_mlflow.py tests/test_artifacts.py
git commit -m "Add metrics-only performance reporting"
```

---

### Task 3: Public Exports And Documentation

**Files:**
- Modify: `src/lumosai/model/__init__.py`
- Modify: `src/lumosai/__init__.py`
- Modify: `tests/test_public_api.py`
- Modify: `docs/api.md`
- Modify: `docs/recipes/tuning-and-final-training.md`

- [ ] **Step 1: Write failing public API tests**

Add to `tests/test_public_api.py`:

```python
def test_metric_types_and_constants_public_api() -> None:
    import lumosai
    from lumosai.model import (
        CLASSIFICATION_METRICS,
        CLASSIFICATION_PROBABILITY_METRICS,
        PERFORMANCE_METRICS,
        REGRESSION_METRICS,
    )

    assert lumosai.CLASSIFICATION_METRICS is CLASSIFICATION_METRICS
    assert lumosai.CLASSIFICATION_PROBABILITY_METRICS is CLASSIFICATION_PROBABILITY_METRICS
    assert lumosai.REGRESSION_METRICS is REGRESSION_METRICS
    assert lumosai.PERFORMANCE_METRICS is PERFORMANCE_METRICS
```

- [ ] **Step 2: Run public API test to verify it fails**

Run:

```bash
uv run pytest tests/test_public_api.py::test_metric_types_and_constants_public_api -v
```

Expected: FAIL because the constants are not exported yet.

- [ ] **Step 3: Export metric constants and type aliases**

In `src/lumosai/model/__init__.py`, add metric names to `__all__`:

```python
    "CLASSIFICATION_METRICS",
    "CLASSIFICATION_PROBABILITY_METRICS",
    "ClassificationMetric",
    "MetricPreset",
    "PERFORMANCE_METRICS",
    "PerformanceMetric",
    "REGRESSION_METRICS",
    "RegressionMetric",
```

Update `__getattr__` to return them from `lumosai.model.metrics`:

```python
if name in {
    "CLASSIFICATION_METRICS",
    "CLASSIFICATION_PROBABILITY_METRICS",
    "ClassificationMetric",
    "MetricPreset",
    "PERFORMANCE_METRICS",
    "PerformanceMetric",
    "REGRESSION_METRICS",
    "RegressionMetric",
}:
    from lumosai.model import metrics as metrics_module

    return getattr(metrics_module, name)
```

In `src/lumosai/__init__.py`, add the same names to `__all__` and route `__getattr__` to `lumosai.model`.

- [ ] **Step 4: Update API docs**

In `docs/api.md`, update the `performance_report(...)` signature block to include:

```python
    metrics="default",
    profile="standard",
    mlflow_step=None,
    log_dict=None,
```

Add a short section under model performance:

```markdown
Supported built-in performance metrics are exposed as constants:

- `CLASSIFICATION_METRICS`: `("accuracy", "precision", "recall", "f1")`
- `CLASSIFICATION_PROBABILITY_METRICS`: `("roc_auc", "pr_auc", "log_loss")`
- `REGRESSION_METRICS`: `("mae", "rmse", "r2")`
- `PERFORMANCE_METRICS`: all built-in names

Use `metrics="default"` to read model metric settings, `metrics="all"` to compute every built-in metric for the task, or `metrics=[...]` to track an explicit subset. Probability metrics require `prediction_score`.

Use `profile="metrics_only"` for fold validation and tuning loops. It suppresses plots, lift, and per-result JSON artifact logging by default while still logging scalar metrics. Pass `mlflow_step=<fold_index>` to log fold metrics as MLflow metric steps.
```

- [ ] **Step 5: Update fold validation recipe**

In `docs/recipes/tuning-and-final-training.md`, change the fold-level `performance_report()` example to:

```python
                result = performance_report(
                    fold_scored,
                    target=target,
                    prediction="prediction",
                    prediction_score="prediction_score",
                    score_labels=list(model.classes_),
                    metrics=["f1", "roc_auc", "pr_auc"],
                    profile="metrics_only",
                    mlflow_step=fold_index,
                    feature_columns=feature_columns,
                    report_name="Fold Validation",
                    experiment_name=EXPERIMENT_NAME,
                )
                fold_scores.append(result.metrics["performance/f1"])
```

Update the surrounding prose to explain that fold-level metrics are logged as MLflow steps and that final training reports should use the standard profile for rich artifacts.

- [ ] **Step 6: Run docs and public API tests**

Run:

```bash
uv run pytest tests/test_public_api.py -v
uv run mkdocs build --strict
```

Expected: PASS. MkDocs may print the existing Material warning and existing nav notices.

- [ ] **Step 7: Commit exports and docs**

Run:

```bash
git add src/lumosai/model/__init__.py src/lumosai/__init__.py tests/test_public_api.py docs/api.md docs/recipes/tuning-and-final-training.md
git commit -m "Document slim performance metrics"
```

---

### Task 4: Full Verification

**Files:**
- No additional source files.

- [ ] **Step 1: Run lint**

Run:

```bash
uv run ruff check src tests docs
```

Expected: PASS with `All checks passed!`.

- [ ] **Step 2: Run full tests**

Run:

```bash
uv run pytest -v
```

Expected: PASS with all tests passing.

- [ ] **Step 3: Run docs build**

Run:

```bash
uv run mkdocs build --strict
```

Expected: PASS. Existing Material/MkDocs warning and existing nav notices are acceptable.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: working tree clean and new task commits visible.
