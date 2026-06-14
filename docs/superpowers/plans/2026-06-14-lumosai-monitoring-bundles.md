# Lumosai Monitoring Bundles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first production bundle layer: `LumosRun`, bundle settings, MLflow run logging, and `monitoring_report()`.

**Architecture:** Keep primitive report functions unchanged and compose them in a new `lumosai.bundles` module. Add `LumosRun` beside `LumosResult` as a grouped result container, add small `BundleSettings` defaults under `settings`, and add `log_run()` to the existing MLflow adapter. `monitoring_report()` performs fail-fast preflight validation, then calls existing primitives inside one MLflow run when logging is enabled.

**Tech Stack:** Python 3.11+, pandas, Pydantic settings, existing Narwhals ingestion, existing `LumosResult`, existing MLflow helper patterns, pytest, ruff, mypy, MkDocs.

---

## File Structure

Create:
- `src/lumosai/bundles.py`: `monitoring_report()` and bundle preflight helpers.
- `tests/test_bundles.py`: monitoring bundle behavior, fail-fast validation, settings overrides, MLflow grouping.

Modify:
- `src/lumosai/results.py`: add `LumosRun`.
- `tests/test_results.py`: add `LumosRun` aggregation tests.
- `src/lumosai/settings.py`: add `BundleSettings` and `settings.bundles`.
- `tests/test_settings.py`: add bundle setting defaults and env override test.
- `src/lumosai/mlflow.py`: add `log_run()`.
- `tests/test_mlflow.py`: add combined run logging test.
- `src/lumosai/__init__.py`: export `LumosRun` and `monitoring_report`.
- `tests/test_public_api.py`: verify top-level exports.
- `docs/api.md`: add `LumosRun` and `monitoring_report()` API docs.
- `docs/recipes/pipeline-patterns.md`: mention bundle promotion path.
- `docs/recipes/monitoring-bundle.md`: new monitoring bundle recipe.
- `mkdocs.yml`: add monitoring bundle recipe to nav.

Do not implement `training_report()` in this plan.

---

### Task 1: Add `LumosRun`

**Files:**
- Modify: `src/lumosai/results.py`
- Modify: `tests/test_results.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_results.py`:

```python
from lumosai.results import LumosResult, LumosRun


def test_lumos_run_aggregates_metrics_and_flagged_items() -> None:
    run = LumosRun(
        run_type="monitoring",
        results={
            "drift_benchmark": LumosResult(
                metrics={"drift/benchmark/share": 0.5},
                flagged=[{"metric": "share", "value": 0.5}],
            ),
            "performance": LumosResult(metrics={"performance/f1": 0.8}),
        },
        metadata={"model": "churn"},
    )

    assert run.metrics == {
        "drift/benchmark/share": 0.5,
        "performance/f1": 0.8,
    }
    assert run.flagged == [
        {"metric": "share", "value": 0.5, "result_key": "drift_benchmark"}
    ]


def test_lumos_run_to_dict_is_json_safe() -> None:
    run = LumosRun(
        run_type="monitoring",
        results={
            "sample": LumosResult(
                artifacts={"frame": pd.DataFrame({"x": [1, 2]})},
                metadata={"report_type": "sample"},
            )
        },
        metadata={"skipped_reports": {"bias": "protected_attribute not provided"}},
    )

    payload = run.to_dict()

    assert payload == {
        "run_type": "monitoring",
        "metrics": {},
        "flagged": [],
        "metadata": {"skipped_reports": {"bias": "protected_attribute not provided"}},
        "results": {
            "sample": {
                "metrics": {},
                "summary": {},
                "flagged": [],
                "artifacts": {"frame": "<DataFrame shape=(2, 1)>"},
                "metadata": {"report_type": "sample"},
            }
        },
    }
```

Ensure `pandas as pd` is imported in `tests/test_results.py` if it is not already.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_results.py -v`

Expected: FAIL with missing `LumosRun`.

- [ ] **Step 3: Implement `LumosRun`**

Add to `src/lumosai/results.py` after `LumosResult`:

```python
@dataclass(slots=True)
class LumosRun:
    """Grouped result returned by lumosai bundle functions."""

    run_type: str
    results: dict[str, LumosResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for result in self.results.values():
            merged.update(result.metrics)
        return merged

    @property
    def flagged(self) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for key, result in self.results.items():
            for item in result.flagged:
                findings.append({**item, "result_key": key})
        return findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_type": self.run_type,
            "metrics": self.metrics,
            "flagged": self.flagged,
            "metadata": self.metadata,
            "results": {key: result.to_dict() for key, result in self.results.items()},
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_results.py -v`

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/results.py tests/test_results.py
uv run ruff check src/lumosai/results.py tests/test_results.py
git add src/lumosai/results.py tests/test_results.py
git commit -m "feat: add lumos run result"
```

---

### Task 2: Add Bundle Settings

**Files:**
- Modify: `src/lumosai/settings.py`
- Modify: `tests/test_settings.py`

- [ ] **Step 1: Write failing settings tests**

Add to `tests/test_settings.py`:

```python
def test_bundle_settings_defaults() -> None:
    loaded = Settings()

    assert loaded.bundles.include_profile_in_training is False
    assert loaded.bundles.include_feature_importance_in_training is True
    assert loaded.bundles.include_previous_window_drift is True
    assert loaded.bundles.fail_fast is True


def test_bundle_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LUMOSAI_BUNDLES__INCLUDE_PREVIOUS_WINDOW_DRIFT", "false")
    monkeypatch.setenv("LUMOSAI_BUNDLES__FAIL_FAST", "false")

    loaded = Settings()

    assert loaded.bundles.include_previous_window_drift is False
    assert loaded.bundles.fail_fast is False
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_settings.py -v`

Expected: FAIL with missing `Settings.bundles`.

- [ ] **Step 3: Implement settings**

Add to `src/lumosai/settings.py` after `ModelSettings`:

```python
class BundleSettings(BaseModel):
    include_profile_in_training: bool = False
    include_feature_importance_in_training: bool = True
    include_previous_window_drift: bool = True
    fail_fast: bool = True
```

Add to `Settings`:

```python
bundles: BundleSettings = Field(default_factory=BundleSettings)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_settings.py -v`

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/settings.py tests/test_settings.py
uv run ruff check src/lumosai/settings.py tests/test_settings.py
git add src/lumosai/settings.py tests/test_settings.py
git commit -m "feat: add bundle settings"
```

---

### Task 3: Add `log_run()`

**Files:**
- Modify: `src/lumosai/mlflow.py`
- Modify: `tests/test_mlflow.py`

- [ ] **Step 1: Write failing MLflow test**

Add to imports in `tests/test_mlflow.py`:

```python
from lumosai.results import LumosResult, LumosRun
```

Replace the existing `LumosResult`-only import rather than duplicating it.

Add to `tests/test_mlflow.py`:

```python
def test_log_run_logs_metrics_and_combined_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setattr(mlflow_adapter, "require_mlflow", lambda: fake_mlflow)
    run = LumosRun(
        run_type="monitoring",
        results={
            "performance": LumosResult(metrics={"performance/f1": 0.8}),
            "drift_benchmark": LumosResult(metrics={"drift/benchmark/share": 0.2}),
        },
    )

    logged = mlflow_adapter.log_run(run, experiment_name="experiment")

    assert logged.metadata["logged_to_mlflow"] is True
    assert logged.metadata["mlflow_run_id"] == "active-run"
    assert fake_mlflow.metrics == {
        "performance/f1": 0.8,
        "drift/benchmark/share": 0.2,
    }
    assert fake_mlflow.dicts[-1][1] == "lumosai_run.json"
    assert fake_mlflow.dicts[-1][0]["run_type"] == "monitoring"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_mlflow.py::test_log_run_logs_metrics_and_combined_payload -v`

Expected: FAIL with missing `log_run`.

- [ ] **Step 3: Implement `log_run()`**

Modify imports in `src/lumosai/mlflow.py`:

```python
from lumosai.results import LumosResult, LumosRun
```

Add after `log_result()`:

```python
def log_run(
    run: LumosRun,
    *,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> LumosRun:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        run.metadata["logged_to_mlflow"] = False
        return run

    with mlflow_run(resolved, loaded_settings) as (mlflow, run_id):
        if mlflow is None:
            run.metadata["logged_to_mlflow"] = False
            return run
        run.metadata["logged_to_mlflow"] = True
        run.metadata["mlflow_run_id"] = run_id
        if run.metrics:
            mlflow.log_metrics(run.metrics)
        if loaded_settings.mlflow.log_dicts:
            mlflow.log_dict(run.to_dict(), "lumosai_run.json")
    return run
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlflow.py -v`

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/mlflow.py tests/test_mlflow.py
uv run ruff check src/lumosai/mlflow.py tests/test_mlflow.py
git add src/lumosai/mlflow.py tests/test_mlflow.py
git commit -m "feat: log lumos runs to mlflow"
```

---

### Task 4: Add Monitoring Bundle Preflight

**Files:**
- Create: `src/lumosai/bundles.py`
- Create: `tests/test_bundles.py`

- [ ] **Step 1: Write failing preflight tests**

Create `tests/test_bundles.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lumosai.bundles import monitoring_report
from lumosai.exceptions import LumosValidationError


def make_monitoring_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "amount": [10, 12, 14, 16, 18, 20],
            "age": [30, 31, 32, 33, 34, 35],
            "target": [0, 1, 0, 1, 0, 1],
            "prediction": [0, 1, 0, 0, 0, 1],
            "region": ["a", "a", "b", "b", "a", "b"],
        }
    )


def test_monitoring_report_requires_temporal_features_for_drift() -> None:
    with pytest.raises(LumosValidationError, match="temporal_features"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            feature_columns=["amount", "age"],
        )


def test_monitoring_report_requires_prediction_when_performance_enabled() -> None:
    with pytest.raises(LumosValidationError, match="prediction"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            target="target",
            include_performance=True,
            feature_columns=["amount", "age"],
        )


def test_monitoring_report_requires_protected_attribute_when_bias_enabled() -> None:
    with pytest.raises(LumosValidationError, match="protected_attribute"):
        monitoring_report(
            make_monitoring_frame(),
            benchmark=make_monitoring_frame(),
            temporal_features=["event_date"],
            target="target",
            prediction="prediction",
            include_bias=True,
            feature_columns=["amount", "age"],
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_bundles.py -v`

Expected: FAIL with missing `lumosai.bundles`.

- [ ] **Step 3: Implement preflight helpers and stub result**

Create `src/lumosai/bundles.py`:

```python
from __future__ import annotations

from typing import Any

import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.results import LumosRun


def _resolve_temporal_features(
    temporal_features: list[str] | None,
    time_column: str | None,
) -> list[str]:
    if temporal_features is not None:
        return list(temporal_features)
    if time_column is not None:
        return [time_column]
    msg = "monitoring_report requires temporal_features or time_column for drift"
    raise LumosValidationError(msg)


def _performance_expected(
    *,
    target: str | None,
    prediction: str | None,
    include_performance: bool | None,
) -> bool:
    return include_performance is True or (target is not None and prediction is not None)


def _bias_expected(
    *,
    protected_attribute: str | list[str] | dict[str, list[float]] | None,
    include_bias: bool | None,
) -> bool:
    return include_bias is True or protected_attribute is not None


def _protected_columns(
    protected_attribute: str | list[str] | dict[str, list[float]],
) -> list[str]:
    if isinstance(protected_attribute, str):
        return [protected_attribute]
    if isinstance(protected_attribute, dict):
        return list(protected_attribute)
    return list(protected_attribute)


def _preflight_monitoring_report(
    *,
    current: pd.DataFrame,
    benchmark: pd.DataFrame,
    previous_window: pd.DataFrame | None,
    target: str | None,
    prediction: str | None,
    prediction_score: str | None,
    feature_columns: list[str] | None,
    categorical_columns: list[str] | None,
    protected_attribute: str | list[str] | dict[str, list[float]] | None,
    temporal_features: list[str],
    include_performance: bool | None,
    include_bias: bool | None,
) -> None:
    drift_columns = list(feature_columns or current.columns)
    require_columns(current, drift_columns)
    require_columns(benchmark, drift_columns)
    require_columns(current, temporal_features)
    require_columns(benchmark, temporal_features)
    if previous_window is not None:
        require_columns(previous_window, drift_columns)
        require_columns(previous_window, temporal_features)
    if categorical_columns is not None:
        missing_categorical = [column for column in categorical_columns if column not in drift_columns]
        if missing_categorical:
            msg = "categorical_columns must be included in feature_columns: "
            msg += ", ".join(missing_categorical)
            raise LumosValidationError(msg)
    if _performance_expected(
        target=target,
        prediction=prediction,
        include_performance=include_performance,
    ):
        if target is None:
            msg = "monitoring_report expected performance but target is missing"
            raise LumosValidationError(msg)
        if prediction is None:
            msg = "monitoring_report expected performance but prediction is missing"
            raise LumosValidationError(msg)
        required = [target, prediction]
        if prediction_score is not None:
            required.append(prediction_score)
        require_columns(current, required)
    if _bias_expected(protected_attribute=protected_attribute, include_bias=include_bias):
        if protected_attribute is None:
            msg = "monitoring_report expected bias but protected_attribute is missing"
            raise LumosValidationError(msg)
        if target is None or prediction is None:
            msg = "monitoring_report expected bias but target and prediction are required"
            raise LumosValidationError(msg)
        require_columns(current, [target, prediction, *_protected_columns(protected_attribute)])


def monitoring_report(
    current: Any,
    *,
    benchmark: Any,
    previous_window: Any | None = None,
    target: str | None = None,
    prediction: str | None = None,
    prediction_score: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    protected_attribute: str | list[str] | dict[str, list[float]] | None = None,
    temporal_features: list[str] | None = None,
    time_column: str | None = None,
    sample_size: int | None = None,
    include_performance: bool | None = None,
    include_bias: bool | None = None,
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosRun:
    current_pd = to_pandas(current)
    benchmark_pd = to_pandas(benchmark)
    previous_pd = to_pandas(previous_window) if previous_window is not None else None
    resolved_temporal_features = _resolve_temporal_features(temporal_features, time_column)
    _preflight_monitoring_report(
        current=current_pd,
        benchmark=benchmark_pd,
        previous_window=previous_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        protected_attribute=protected_attribute,
        temporal_features=resolved_temporal_features,
        include_performance=include_performance,
        include_bias=include_bias,
    )
    return LumosRun(
        run_type="monitoring",
        results={},
        metadata={
            "report_name": report_name,
            "skipped_reports": {},
            "sample_size": sample_size,
            "experiment_name": experiment_name,
        },
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bundles.py -v`

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/bundles.py tests/test_bundles.py
uv run ruff check src/lumosai/bundles.py tests/test_bundles.py
git add src/lumosai/bundles.py tests/test_bundles.py
git commit -m "feat: add monitoring bundle preflight"
```

---

### Task 5: Implement `monitoring_report()`

**Files:**
- Modify: `src/lumosai/bundles.py`
- Modify: `tests/test_bundles.py`

- [ ] **Step 1: Add happy path and optional report tests**

Add to `tests/test_bundles.py`:

```python
def test_monitoring_report_runs_sample_drift_and_performance() -> None:
    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        target="target",
        prediction="prediction",
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
        sample_size=3,
        report_name="daily-monitoring",
    )

    assert result.run_type == "monitoring"
    assert set(result.results) == {"monitoring_window", "drift_benchmark", "performance"}
    assert result.results["monitoring_window"].metadata["sample_role"] == "monitoring_window"
    assert result.results["drift_benchmark"].metadata["comparison"] == "benchmark"
    assert result.results["performance"].metadata["report_type"] == "performance"
    assert result.metadata["report_name"] == "daily-monitoring"


def test_monitoring_report_runs_previous_window_drift_when_provided() -> None:
    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        previous_window=make_monitoring_frame(),
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
    )

    assert "drift_previous_window" in result.results
    assert result.results["drift_previous_window"].metadata["comparison"] == "previous_window"


def test_monitoring_report_runs_bias_when_protected_attribute_provided() -> None:
    result = monitoring_report(
        make_monitoring_frame(),
        benchmark=make_monitoring_frame(),
        target="target",
        prediction="prediction",
        protected_attribute="region",
        temporal_features=["event_date"],
        feature_columns=["amount", "age"],
    )

    assert "bias" in result.results
    assert result.results["bias"].metadata["report_type"] == "bias"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_bundles.py -v`

Expected: FAIL because `monitoring_report()` returns no child results.

- [ ] **Step 3: Implement bundle composition**

Update imports in `src/lumosai/bundles.py`:

```python
from lumosai.data.drift import drift_report
from lumosai.data.sampling import build_sample
from lumosai.mlflow import log_run, mlflow_run, resolve_experiment_name
from lumosai.model.bias import bias_report
from lumosai.model.performance import performance_report
from lumosai.settings import settings
```

Replace the stub return in `monitoring_report()` with:

```python
    results: dict[str, LumosResult] = {}
    skipped_reports: dict[str, str] = {}
    resolved_experiment = resolve_experiment_name(experiment_name)
    run_context = mlflow_run(resolved_experiment) if resolved_experiment else nullcontext((None, None))
    with run_context:
        results["monitoring_window"] = build_sample(
            current_pd,
            role="monitoring_window",
            sample_size=sample_size,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            time_column=time_column,
            experiment_name=resolved_experiment,
        )
        results["drift_benchmark"] = drift_report(
            benchmark_pd,
            current_pd,
            temporal_features=resolved_temporal_features,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            comparison="benchmark",
            report_name=f"{report_name} Benchmark Drift" if report_name else None,
            experiment_name=resolved_experiment,
        )
        if previous_pd is not None:
            results["drift_previous_window"] = drift_report(
                previous_pd,
                current_pd,
                temporal_features=resolved_temporal_features,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                comparison="previous_window",
                report_name=f"{report_name} Previous Window Drift" if report_name else None,
                experiment_name=resolved_experiment,
            )
        else:
            skipped_reports["drift_previous_window"] = "previous_window not provided"
        if _performance_expected(
            target=target,
            prediction=prediction,
            include_performance=include_performance,
        ):
            results["performance"] = performance_report(
                current_pd,
                target=target or "",
                prediction=prediction or "",
                prediction_score=prediction_score,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                report_name=f"{report_name} Performance" if report_name else None,
                experiment_name=resolved_experiment,
            )
        else:
            skipped_reports["performance"] = "target and prediction not provided"
        if _bias_expected(protected_attribute=protected_attribute, include_bias=include_bias):
            results["bias"] = bias_report(
                current_pd,
                target=target or "",
                prediction=prediction or "",
                protected_attribute=protected_attribute or "",
                prediction_score=prediction_score,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                report_name=f"{report_name} Bias" if report_name else None,
                experiment_name=resolved_experiment,
            )
        else:
            skipped_reports["bias"] = "protected_attribute not provided"
        run = LumosRun(
            run_type="monitoring",
            results=results,
            metadata={
                "report_name": report_name,
                "skipped_reports": skipped_reports,
            },
        )
        log_run(run, experiment_name=resolved_experiment)
        return run
```

Also import `nullcontext` and `LumosResult`:

```python
from contextlib import nullcontext
from lumosai.results import LumosResult, LumosRun
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bundles.py -v`

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/bundles.py tests/test_bundles.py
uv run ruff check src/lumosai/bundles.py tests/test_bundles.py
git add src/lumosai/bundles.py tests/test_bundles.py
git commit -m "feat: add monitoring report bundle"
```

---

### Task 6: Export Public API

**Files:**
- Modify: `src/lumosai/__init__.py`
- Modify: `tests/test_public_api.py`

- [ ] **Step 1: Write failing public API test**

Modify `tests/test_public_api.py`:

```python
from lumosai.results import LumosResult, LumosRun
```

Add:

```python
def test_bundle_public_api() -> None:
    import lumosai
    from lumosai.bundles import monitoring_report

    assert lumosai.monitoring_report is monitoring_report
    assert lumosai.LumosRun is LumosRun
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_public_api.py -v`

Expected: FAIL with missing top-level exports.

- [ ] **Step 3: Export bundle API**

Modify `src/lumosai/__init__.py`:

```python
from lumosai.results import LumosResult, LumosRun
```

Add to `__all__`:

```python
"LumosRun",
"monitoring_report",
```

Add branch to `__getattr__`:

```python
if name == "monitoring_report":
    from lumosai.bundles import monitoring_report

    return monitoring_report
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_public_api.py -v`

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/__init__.py tests/test_public_api.py
uv run ruff check src/lumosai/__init__.py tests/test_public_api.py
git add src/lumosai/__init__.py tests/test_public_api.py
git commit -m "feat: export monitoring bundle api"
```

---

### Task 7: Add Monitoring Bundle Docs

**Files:**
- Create: `docs/recipes/monitoring-bundle.md`
- Modify: `docs/api.md`
- Modify: `docs/recipes/pipeline-patterns.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Add API docs**

Add to `docs/api.md` after `LumosResult`:

````markdown
## `LumosRun`

Bundle functions return `LumosRun`.

Fields:

- `run_type`: bundle type such as `"monitoring"`.
- `results`: child `LumosResult` objects keyed by stable names.
- `metadata`: bundle metadata such as skipped reports and report name.

Properties:

- `metrics`: merged child metrics.
- `flagged`: flagged child findings annotated with `result_key`.
````

Add after Data APIs or before Model APIs:

````markdown
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
)
```

Runs a production-oriented monitoring bundle.

- Always builds a `monitoring_window` sample.
- Always runs benchmark drift.
- Runs previous-window drift when `previous_window` is provided.
- Runs performance when `target` and `prediction` are provided or `include_performance=True`.
- Runs bias when `protected_attribute` is provided or `include_bias=True`.
- Fails before running reports when expected inputs or columns are missing.
- Returns a `LumosRun`.
````

- [ ] **Step 2: Add monitoring bundle recipe**

Create `docs/recipes/monitoring-bundle.md`:

````markdown
# Monitoring Bundle

Use `monitoring_report()` when a scheduled job should run the standard production checks with one call.

```python
from lumosai import monitoring_report

run = monitoring_report(
    current_window,
    benchmark=train_benchmark,
    previous_window=previous_window,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    protected_attribute="region",
    temporal_features=["event_date"],
    feature_columns=["tenure", "plan", "monthly_spend"],
    categorical_columns=["plan"],
    sample_size=5000,
    report_name="Daily Monitoring",
    experiment_name="model-monitoring",
)
```

The bundle fails fast when expected inputs are missing. For example, passing `include_performance=True` without `prediction` raises before drift or sample reports run.

Use primitives during development:

```python
from lumosai.data import drift_report
from lumosai.model import performance_report
```

Promote the same arguments into `monitoring_report()` when the job is ready to run repeatedly.
````

- [ ] **Step 3: Update pipeline docs and nav**

In `docs/recipes/pipeline-patterns.md`, add a short section before the code-heavy monitoring example:

```markdown
For scheduled production jobs, prefer `monitoring_report()` once the primitive report arguments are stable. It runs the same checks through one fail-fast bundle and returns a `LumosRun`.
```

In `mkdocs.yml`, add:

```yaml
      - Monitoring Bundle: recipes/monitoring-bundle.md
```

under `Recipes`.

- [ ] **Step 4: Build docs**

Run: `uv run mkdocs build --strict`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add docs/api.md docs/recipes/monitoring-bundle.md docs/recipes/pipeline-patterns.md mkdocs.yml
git commit -m "docs: add monitoring bundle guide"
```

---

### Task 8: Full Verification

**Files:**
- No source edits unless verification exposes a defect.

- [ ] **Step 1: Format**

Run: `uv run ruff format .`

Expected: no formatting errors.

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`

Expected: PASS.

- [ ] **Step 3: Type check**

Run: `uv run mypy src/lumosai`

Expected: PASS.

- [ ] **Step 4: Test**

Run: `uv run pytest -v`

Expected: PASS.

- [ ] **Step 5: Build docs**

Run: `uv run mkdocs build --strict`

Expected: PASS.

- [ ] **Step 6: Commit final fixes if any**

If verification required changes:

```bash
git add <changed-files>
git commit -m "fix: address monitoring bundle verification issues"
```

If no files changed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- `LumosRun` is covered by Task 1.
- `BundleSettings` is covered by Task 2.
- `log_run()` and combined `lumosai_run.json` are covered by Task 3.
- Fail-fast preflight validation is covered by Task 4.
- `monitoring_report()` is covered by Task 5.
- Top-level public exports are covered by Task 6.
- Monitoring bundle docs are covered by Task 7.
- Full verification is covered by Task 8.

Scope:

- `training_report()` is intentionally excluded and should receive a separate plan after monitoring bundle behavior is reviewed.
- The plan does not add orchestration, scheduling, model registry behavior, or label joins.

Type consistency:

- `LumosRun` lives in `src/lumosai/results.py` and is exported top-level.
- `monitoring_report()` returns `LumosRun`.
- Child result keys match the design: `monitoring_window`, `drift_benchmark`, `drift_previous_window`, `performance`, and `bias`.
