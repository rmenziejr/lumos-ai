# lumosai Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v0.1 `lumosai` package as a function-first ML monitoring/reporting library with Narwhals ingestion, Pydantic settings, structured results, MLflow-aware logging, profiling, drift, performance, bias, and documentation recipes.

**Architecture:** Use a `src/` Python package split into `lumosai.data` and `lumosai.model`, with shared `settings`, `results`, `exceptions`, and `mlflow` modules. Public report functions normalize dataframes to pandas at boundaries, validate inputs early, compute JSON-safe summaries, and return `LumosResult`; MLflow logging is optional and routed through one adapter.

**Tech Stack:** Python 3.11+, uv, pandas, narwhals, pydantic, pydantic-settings, scikit-learn, matplotlib, ydata-profiling, evidently, pytest, ruff.

---

## File Structure

Create this structure:

```text
pyproject.toml
README.md
src/lumosai/
  __init__.py
  exceptions.py
  artifacts.py
  results.py
  settings.py
  mlflow.py
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
tests/
  conftest.py
  data/
    test_ingest.py
    test_validation.py
    test_profiling.py
    test_drift.py
  model/
    test_metrics.py
    test_bias.py
    test_performance.py
  test_public_api.py
  test_results.py
  test_settings.py
  test_mlflow.py
docs/
  recipes/
    data-pipeline-monitoring.md
    training-pipeline-reporting.md
    ongoing-monitoring-pipeline.md
    mlflow-layout.md
```

Responsibility map:

- `settings.py`: nested Pydantic settings and the global `settings` object.
- `results.py`: `LumosResult` and JSON-safe serialization helpers.
- `exceptions.py`: package-specific exception classes.
- `artifacts.py`: local and temporary artifact directory handling.
- `mlflow.py`: optional MLflow import, setup, run context, metric/artifact logging.
- `data/ingest.py`: Narwhals-to-pandas conversion.
- `data/validation.py`: dataframe and column validation helpers.
- `data/profiling.py`: temporal sampling and ydata-profiling wrapper.
- `data/drift.py`: Evidently drift wrapper and drift summary extraction.
- `model/validation.py`: model dataframe validation.
- `model/metrics.py`: task detection, metric calculation, threshold comparison.
- `model/performance.py`: current-window performance report.
- `model/bias.py`: group-wise metrics, disparity comparisons, and flags.

---

### Task 1: Scaffold Package Metadata

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/lumosai/__init__.py`
- Create: `src/lumosai/data/__init__.py`
- Create: `src/lumosai/model/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create package metadata**

Write `pyproject.toml`:

```toml
[project]
name = "lumosai"
version = "0.1.0"
description = "Opinionated ML monitoring and reporting helpers."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "evidently>=0.4.38",
  "matplotlib>=3.8",
  "narwhals>=1.0",
  "pandas>=2.1",
  "pydantic>=2.7",
  "pydantic-settings>=2.2",
  "scikit-learn>=1.4",
  "ydata-profiling>=4.8",
]

[project.optional-dependencies]
mlflow = ["mlflow>=2.18"]
importance = ["shap>=0.45"]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0",
  "ruff>=0.5",
  "mypy>=1.10",
  "polars>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lumosai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "integration: tests that exercise heavy optional integrations",
  "mlflow: tests that require mlflow",
]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "ANN", "S"]
ignore = ["ANN401", "S101"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["ANN"]

[tool.mypy]
python_version = "3.11"
packages = ["lumosai"]
strict = true
warn_unused_ignores = true
warn_return_any = true
```

- [ ] **Step 2: Create minimal README**

Write `README.md`:

```markdown
# lumosai

`lumosai` is an opinionated Python package for ML monitoring and reporting.
It provides simple function-first APIs for profiling, drift, performance, and bias checks while returning structured results and optionally logging to MLflow.
```

- [ ] **Step 3: Create package init files**

Write `src/lumosai/__init__.py`:

```python
"""Public API for lumosai."""

from lumosai.data import drift_report, profile
from lumosai.model import bias_report, get_metrics, performance_report
from lumosai.results import LumosResult
from lumosai.settings import settings

__all__ = [
    "LumosResult",
    "bias_report",
    "drift_report",
    "get_metrics",
    "performance_report",
    "profile",
    "settings",
]
```

Write `src/lumosai/data/__init__.py`:

```python
"""Data monitoring helpers."""

from lumosai.data.drift import drift_report
from lumosai.data.profiling import profile, temporal_sample

__all__ = ["drift_report", "profile", "temporal_sample"]
```

Write `src/lumosai/model/__init__.py`:

```python
"""Model monitoring helpers."""

from lumosai.model.bias import bias_report
from lumosai.model.metrics import get_metrics
from lumosai.model.performance import performance_report

__all__ = ["bias_report", "get_metrics", "performance_report"]
```

Write `tests/conftest.py`:

```python
from __future__ import annotations
```

- [ ] **Step 4: Run initial checks**

Run: `uv run pytest`

Expected: pytest starts and reports no tests collected or all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md src/lumosai/__init__.py src/lumosai/data/__init__.py src/lumosai/model/__init__.py tests/conftest.py
git commit -m "chore: scaffold lumosai package"
```

---

### Task 2: Add Exceptions And Result Type

**Files:**
- Create: `src/lumosai/exceptions.py`
- Create: `src/lumosai/results.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write result tests**

Write `tests/test_results.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lumosai.results import LumosResult, json_safe_artifacts


def test_lumos_result_to_dict_excludes_report_and_serializes_artifacts() -> None:
    result = LumosResult(
        metrics={"performance/f1": 0.8},
        summary={"rows": 10},
        flagged=[{"metric": "f1", "reason": "below_threshold"}],
        artifacts={"html": Path("report.html"), "frame": pd.DataFrame({"a": [1]})},
        report=object(),
        metadata={"report_type": "performance"},
    )

    payload = result.to_dict()

    assert payload == {
        "metrics": {"performance/f1": 0.8},
        "summary": {"rows": 10},
        "flagged": [{"metric": "f1", "reason": "below_threshold"}],
        "artifacts": {"html": "report.html", "frame": "<DataFrame shape=(1, 1)>"},
        "metadata": {"report_type": "performance"},
    }


def test_json_safe_artifacts_handles_nested_values() -> None:
    payload = json_safe_artifacts({"paths": [Path("a.html"), Path("b.json")]})

    assert payload == {"paths": ["a.html", "b.json"]}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_results.py -v`

Expected: FAIL with `ModuleNotFoundError` for `lumosai.results`.

- [ ] **Step 3: Implement exceptions and results**

Write `src/lumosai/exceptions.py`:

```python
from __future__ import annotations


class LumosError(Exception):
    """Base exception for lumosai errors."""


class LumosValidationError(LumosError, ValueError):
    """Raised when user-provided data or arguments are invalid."""


class LumosOptionalDependencyError(LumosError, ImportError):
    """Raised when an optional dependency is required but not installed."""


class LumosConfigurationError(LumosError, RuntimeError):
    """Raised when runtime configuration prevents a requested action."""
```

Write `src/lumosai/results.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def json_safe_artifacts(value: Any) -> Any:
    """Convert artifact metadata into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return f"<DataFrame shape={value.shape}>"
    if isinstance(value, dict):
        return {str(key): json_safe_artifacts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_artifacts(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe_artifacts(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return f"<{type(value).__name__}>"


@dataclass(slots=True)
class LumosResult:
    """Structured result returned by lumosai report functions."""

    metrics: dict[str, float] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    flagged: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    report: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe fields for logging, APIs, or persisted summaries."""
        return {
            "metrics": self.metrics,
            "summary": self.summary,
            "flagged": self.flagged,
            "artifacts": json_safe_artifacts(self.artifacts),
            "metadata": self.metadata,
        }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_results.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/exceptions.py src/lumosai/results.py tests/test_results.py
git commit -m "feat: add structured result type"
```

---

### Task 3: Add Nested Settings

**Files:**
- Create: `src/lumosai/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write settings tests**

Write `tests/test_settings.py`:

```python
from __future__ import annotations

from pathlib import Path

from lumosai.settings import MetricThreshold, Settings


def test_settings_parse_nested_environment(monkeypatch) -> None:
    monkeypatch.setenv("LUMOSAI_MLFLOW__TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("LUMOSAI_MLFLOW__RUN_MODE", "require_active")
    monkeypatch.setenv("LUMOSAI_ARTIFACTS__KEEP_LOCAL", "true")
    monkeypatch.setenv("LUMOSAI_ARTIFACTS__LOCAL_DIR", "./artifacts")
    monkeypatch.setenv("LUMOSAI_MODEL__SHAP_SAMPLE_SIZE", "250")

    loaded = Settings()

    assert loaded.mlflow.tracking_uri == "http://localhost:5000"
    assert loaded.mlflow.run_mode == "require_active"
    assert loaded.artifacts.keep_local is True
    assert loaded.artifacts.local_dir == Path("artifacts")
    assert loaded.model.shap_sample_size == 250


def test_metric_threshold_defaults_include_metric_direction() -> None:
    loaded = Settings()

    assert loaded.model.metric_thresholds["f1"] == MetricThreshold(
        mode="relative",
        value=0.8,
        greater_is_better=True,
    )
    assert loaded.model.metric_thresholds["rmse"] == MetricThreshold(
        mode="relative",
        value=1.25,
        greater_is_better=False,
    )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_settings.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing `Settings`.

- [ ] **Step 3: Implement settings**

Write `src/lumosai/settings.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    classification_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )
    classification_probability_metrics: list[str] = Field(default_factory=lambda: ["roc_auc"])
    regression_metrics: list[str] = Field(default_factory=lambda: ["mae", "rmse", "r2"])
    metric_thresholds: dict[str, MetricThreshold] = Field(
        default_factory=lambda: {
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
    )
    include_perm_importance: bool = True
    log_shap: bool = True
    shap_sample_size: int = 1000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMOSAI_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)


settings = Settings()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_settings.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/settings.py tests/test_settings.py
git commit -m "feat: add nested pydantic settings"
```

---

### Task 4: Add Data Ingestion And Validation

**Files:**
- Create: `src/lumosai/data/ingest.py`
- Create: `src/lumosai/data/validation.py`
- Create: `tests/data/test_ingest.py`
- Create: `tests/data/test_validation.py`

- [ ] **Step 1: Write ingestion tests**

Write `tests/data/test_ingest.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lumosai.data.ingest import to_pandas
from lumosai.exceptions import LumosValidationError


def test_to_pandas_returns_copy_for_pandas_dataframe() -> None:
    source = pd.DataFrame({"a": [1, 2]})

    result = to_pandas(source)
    result.loc[0, "a"] = 99

    assert isinstance(result, pd.DataFrame)
    assert source.loc[0, "a"] == 1


def test_to_pandas_rejects_empty_dataframe() -> None:
    with pytest.raises(LumosValidationError, match="must contain at least one row"):
        to_pandas(pd.DataFrame({"a": []}))
```

Write `tests/data/test_validation.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lumosai.data.validation import (
    require_columns,
    require_no_duplicate_columns,
    validate_temporal_features,
)
from lumosai.exceptions import LumosValidationError


def test_require_columns_reports_missing_columns() -> None:
    frame = pd.DataFrame({"a": [1]})

    with pytest.raises(LumosValidationError, match="missing required columns: b"):
        require_columns(frame, ["a", "b"])


def test_require_no_duplicate_columns_reports_duplicates() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(LumosValidationError, match="duplicate columns: a"):
        require_no_duplicate_columns(frame)


def test_validate_temporal_features_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"feature": [1]})

    with pytest.raises(LumosValidationError, match="temporal_features not found: event_date"):
        validate_temporal_features(frame, ["event_date"])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/data/test_ingest.py tests/data/test_validation.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Implement ingestion and validation**

Write `src/lumosai/data/ingest.py`:

```python
from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd

from lumosai.exceptions import LumosValidationError


def to_pandas(df: Any) -> pd.DataFrame:
    """Convert a Narwhals-compatible dataframe to a pandas DataFrame copy."""
    try:
        native = nw.from_native(df)
        pandas_df = native.to_pandas()
    except Exception as exc:
        msg = "df must be a Narwhals-compatible dataframe object"
        raise LumosValidationError(msg) from exc

    if not isinstance(pandas_df, pd.DataFrame):
        msg = "converted dataframe must be a pandas DataFrame"
        raise LumosValidationError(msg)
    if pandas_df.empty:
        msg = "dataframe must contain at least one row"
        raise LumosValidationError(msg)
    return pandas_df.copy()
```

Write `src/lumosai/data/validation.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from lumosai.exceptions import LumosValidationError


def require_no_duplicate_columns(df: pd.DataFrame) -> None:
    duplicates = sorted({column for column in df.columns if list(df.columns).count(column) > 1})
    if duplicates:
        msg = f"dataframe has duplicate columns: {', '.join(map(str, duplicates))}"
        raise LumosValidationError(msg)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"dataframe is missing required columns: {', '.join(missing)}"
        raise LumosValidationError(msg)


def validate_temporal_features(df: pd.DataFrame, temporal_features: list[str]) -> None:
    require_no_duplicate_columns(df)
    missing = [column for column in temporal_features if column not in df.columns]
    if missing:
        msg = f"temporal_features not found: {', '.join(missing)}"
        raise LumosValidationError(msg)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/data/test_ingest.py tests/data/test_validation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/data/ingest.py src/lumosai/data/validation.py tests/data/test_ingest.py tests/data/test_validation.py
git commit -m "feat: add dataframe ingestion and validation"
```

---

### Task 5: Add Model Metrics And Thresholds

**Files:**
- Create: `src/lumosai/model/metrics.py`
- Create: `src/lumosai/model/validation.py`
- Create: `tests/model/test_metrics.py`

- [ ] **Step 1: Write metric tests**

Write `tests/model/test_metrics.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lumosai.model.metrics import compare_metric, detect_task_type, get_metrics
from lumosai.settings import MetricThreshold


def test_detect_task_type_classification_for_low_cardinality_labels() -> None:
    assert detect_task_type(pd.Series([0, 1, 1, 0]), pd.Series([0, 1, 0, 0])) == "classification"


def test_detect_task_type_regression_for_float_continuous_values() -> None:
    assert detect_task_type(pd.Series([1.2, 2.5, 3.7, 4.1]), pd.Series([1.0, 2.0, 4.0, 4.5])) == "regression"


def test_get_metrics_classification_uses_scores_for_roc_auc() -> None:
    metrics = get_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 0],
        y_score=[0.1, 0.9, 0.8, 0.2],
        task_type="classification",
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_get_metrics_regression() -> None:
    metrics = get_metrics(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.5, 2.5],
        task_type="regression",
    )

    assert metrics["mae"] == pytest.approx(0.3333333333)
    assert metrics["rmse"] == pytest.approx(0.4082482904)
    assert metrics["r2"] == pytest.approx(0.75)


def test_compare_metric_higher_is_better_relative_flag() -> None:
    threshold = MetricThreshold(mode="relative", value=0.8, greater_is_better=True)

    comparison = compare_metric("f1", group_value=0.70, best_value=1.0, threshold=threshold)

    assert comparison["flagged"] is True
    assert comparison["ratio"] == pytest.approx(0.7)


def test_compare_metric_lower_is_better_relative_flag() -> None:
    threshold = MetricThreshold(mode="relative", value=1.25, greater_is_better=False)

    comparison = compare_metric("rmse", group_value=1.4, best_value=1.0, threshold=threshold)

    assert comparison["flagged"] is True
    assert comparison["ratio"] == pytest.approx(1.4)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/model/test_metrics.py -v`

Expected: FAIL with missing `lumosai.model.metrics`.

- [ ] **Step 3: Implement metrics and model validation**

Write `src/lumosai/model/validation.py`:

```python
from __future__ import annotations

import pandas as pd

from lumosai.data.validation import require_columns, require_no_duplicate_columns


def validate_prediction_frame(
    df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    prediction_score: str | None = None,
) -> None:
    required = [target, prediction]
    if prediction_score is not None:
        required.append(prediction_score)
    require_no_duplicate_columns(df)
    require_columns(df, required)
```

Write `src/lumosai/model/metrics.py`:

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_auc_score,
)

from lumosai.exceptions import LumosValidationError
from lumosai.settings import MetricThreshold, settings

TaskType = Literal["classification", "regression"]


def detect_task_type(y_true: Sequence[Any] | pd.Series, y_pred: Sequence[Any] | pd.Series) -> TaskType:
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    combined = pd.concat([y_true_series, y_pred_series], ignore_index=True).dropna()
    unique_count = combined.nunique()
    is_float = pd.api.types.is_float_dtype(combined)
    if is_float and unique_count > min(20, max(2, len(combined) // 10)):
        return "regression"
    return "classification"


def get_metrics(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
    y_score: Sequence[float] | pd.Series | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
) -> dict[str, float]:
    resolved_task = task_type or detect_task_type(y_true, y_pred)
    metrics: dict[str, float] = {}

    if resolved_task == "classification":
        average = "binary" if pd.Series(y_true).nunique(dropna=True) <= 2 else "weighted"
        zero_division = 0
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=zero_division))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=zero_division))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=zero_division))
        if y_score is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    for name, metric_func in custom_metrics or []:
        metrics[name] = float(metric_func(y_true, y_pred))

    return metrics


def compare_metric(
    metric: str,
    *,
    group_value: float,
    best_value: float,
    threshold: MetricThreshold | None = None,
) -> dict[str, float | bool | str]:
    resolved = threshold or settings.model.metric_thresholds.get(metric)
    if resolved is None:
        resolved = MetricThreshold(mode="relative", value=0.8, greater_is_better=True)

    if resolved.mode == "absolute":
        diff = group_value - best_value
        flagged = diff < -resolved.value if resolved.greater_is_better else diff > resolved.value
        ratio = np.nan
    else:
        if best_value == 0:
            ratio = np.inf if group_value != 0 else 1.0
        else:
            ratio = group_value / best_value
        flagged = ratio < resolved.value if resolved.greater_is_better else ratio > resolved.value
        diff = group_value - best_value

    return {
        "metric": metric,
        "group_value": float(group_value),
        "best_value": float(best_value),
        "diff": float(diff),
        "ratio": float(ratio),
        "threshold": float(resolved.value),
        "flagged": bool(flagged),
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/model/test_metrics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/model/metrics.py src/lumosai/model/validation.py tests/model/test_metrics.py
git commit -m "feat: add model metrics"
```

---

### Task 6: Add Performance Report

**Files:**
- Create: `src/lumosai/model/performance.py`
- Create: `tests/model/test_performance.py`

- [ ] **Step 1: Write performance tests**

Write `tests/model/test_performance.py`:

```python
from __future__ import annotations

import pandas as pd

from lumosai.model.performance import performance_report
from lumosai.results import LumosResult


def test_performance_report_returns_namespaced_metrics() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 1, 0],
            "prediction_score": [0.1, 0.9, 0.8, 0.2],
        }
    )

    result = performance_report(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        task_type="classification",
    )

    assert isinstance(result, LumosResult)
    assert result.metrics["performance/accuracy"] == 1.0
    assert result.metrics["performance/roc_auc"] == 1.0
    assert result.metadata["report_type"] == "performance"
    assert result.metadata["task_type"] == "classification"


def test_performance_report_to_dict_is_json_safe() -> None:
    frame = pd.DataFrame({"actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.5, 2.5]})

    result = performance_report(frame, target="actual", prediction="prediction", task_type="regression")

    assert result.to_dict()["metrics"]["performance/mae"] > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/model/test_performance.py -v`

Expected: FAIL with missing `performance_report`.

- [ ] **Step 3: Implement performance report**

Write `src/lumosai/model/performance.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lumosai.data.ingest import to_pandas
from lumosai.mlflow import log_result
from lumosai.model.metrics import TaskType, detect_task_type, get_metrics
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult


def performance_report(
    current: Any,
    target: str,
    prediction: str,
    prediction_score: str | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    current_pd = to_pandas(current)
    validate_prediction_frame(
        current_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
    )
    resolved_task = task_type or detect_task_type(current_pd[target], current_pd[prediction])
    raw_metrics = get_metrics(
        current_pd[target],
        current_pd[prediction],
        y_score=current_pd[prediction_score] if prediction_score else None,
        task_type=resolved_task,
        custom_metrics=custom_metrics,
    )
    metrics = {f"performance/{name}": value for name, value in raw_metrics.items()}
    result = LumosResult(
        metrics=metrics,
        summary={"rows": len(current_pd), "metrics": raw_metrics},
        metadata={"report_type": "performance", "task_type": resolved_task},
    )
    log_result(result, experiment_name=experiment_name)
    return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/model/test_performance.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/model/performance.py tests/model/test_performance.py
git commit -m "feat: add performance report"
```

---

### Task 7: Add Artifact Handling And MLflow Adapter

**Files:**
- Create: `src/lumosai/artifacts.py`
- Create: `src/lumosai/mlflow.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_mlflow.py`

- [ ] **Step 1: Write artifact and MLflow adapter tests**

Write `tests/test_artifacts.py`:

```python
from __future__ import annotations

from pathlib import Path

from lumosai.artifacts import artifact_workspace
from lumosai.settings import Settings


def test_artifact_workspace_uses_temp_dir_when_not_keeping_local() -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = False

    with artifact_workspace(loaded_settings=loaded) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")
        assert path.exists()

    assert not path.exists()


def test_artifact_workspace_keeps_configured_local_dir(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = True
    loaded.artifacts.local_dir = tmp_path

    with artifact_workspace(loaded_settings=loaded) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")

    assert path.exists()


def test_artifact_workspace_can_force_local_retention(tmp_path: Path) -> None:
    loaded = Settings()
    loaded.artifacts.keep_local = False
    loaded.artifacts.local_dir = tmp_path

    with artifact_workspace(loaded_settings=loaded, keep_local=True) as workspace:
        path = workspace / "report.html"
        path.write_text("ok")

    assert path.exists()
```

Write `tests/test_mlflow.py`:

```python
from __future__ import annotations

from lumosai.mlflow import resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.settings import Settings


def test_resolve_experiment_name_prefers_argument() -> None:
    loaded = Settings()

    assert resolve_experiment_name("explicit", loaded) == "explicit"


def test_resolve_experiment_name_uses_default_setting() -> None:
    loaded = Settings()
    loaded.mlflow.default_experiment_name = "default"

    assert resolve_experiment_name(None, loaded) == "default"


def test_log_result_without_experiment_returns_original_result() -> None:
    from lumosai.mlflow import log_result

    result = LumosResult(metrics={"performance/f1": 1.0})

    assert log_result(result, experiment_name=None) is result
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_artifacts.py tests/test_mlflow.py -v`

Expected: FAIL with missing `lumosai.artifacts` and `lumosai.mlflow`.

- [ ] **Step 3: Implement artifact handling and MLflow adapter**

Write `src/lumosai/artifacts.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from lumosai.settings import Settings, settings


@contextmanager
def artifact_workspace(
    loaded_settings: Settings = settings,
    *,
    keep_local: bool | None = None,
) -> Iterator[Path]:
    """Yield a directory for artifacts and clean it up unless local retention is enabled."""
    resolved_keep_local = loaded_settings.artifacts.keep_local if keep_local is None else keep_local
    if resolved_keep_local:
        directory = loaded_settings.artifacts.local_dir or Path("lumosai-artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        yield directory
        return

    with TemporaryDirectory(prefix="lumosai-") as temp_dir:
        yield Path(temp_dir)
```

Write `src/lumosai/mlflow.py`:

```python
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from lumosai.exceptions import LumosConfigurationError, LumosOptionalDependencyError
from lumosai.results import LumosResult
from lumosai.settings import Settings, settings


def resolve_experiment_name(experiment_name: str | None, loaded_settings: Settings = settings) -> str | None:
    return experiment_name or loaded_settings.mlflow.default_experiment_name


def require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        msg = "mlflow logging requires the optional dependency: pip install lumosai[mlflow]"
        raise LumosOptionalDependencyError(msg) from exc
    return mlflow


def configure_mlflow(mlflow: Any, loaded_settings: Settings = settings) -> None:
    if loaded_settings.mlflow.username is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = loaded_settings.mlflow.username
    if loaded_settings.mlflow.password is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = loaded_settings.mlflow.password
    if loaded_settings.mlflow.tracking_uri is not None:
        mlflow.set_tracking_uri(loaded_settings.mlflow.tracking_uri)


@contextmanager
def mlflow_run(
    experiment_name: str | None,
    loaded_settings: Settings = settings,
) -> Iterator[tuple[Any, str | None]]:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        yield None, None
        return

    mlflow = require_mlflow()
    configure_mlflow(mlflow, loaded_settings)
    mlflow.set_experiment(resolved)

    active_run = mlflow.active_run()
    if active_run is not None:
        yield mlflow, active_run.info.run_id
        return

    if loaded_settings.mlflow.run_mode == "require_active":
        msg = "MLflow logging requested but no active run exists and run_mode='require_active'"
        raise LumosConfigurationError(msg)

    with mlflow.start_run() as run:
        yield mlflow, run.info.run_id


def log_result(
    result: LumosResult,
    *,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> LumosResult:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        result.metadata["logged_to_mlflow"] = False
        return result

    context = mlflow_run(resolved, loaded_settings) if resolved else nullcontext((None, None))
    with context as (mlflow, run_id):
        if mlflow is None:
            result.metadata["logged_to_mlflow"] = False
            return result
        if result.metrics:
            mlflow.log_metrics(result.metrics)
        if loaded_settings.mlflow.log_dicts:
            mlflow.log_dict(result.to_dict(), "lumosai_result.json")
        result.metadata["logged_to_mlflow"] = True
        result.metadata["mlflow_run_id"] = run_id
    return result


def log_artifact_paths(
    paths: dict[str, Path],
    *,
    experiment_name: str | None = None,
    loaded_settings: Settings = settings,
) -> dict[str, str]:
    resolved = resolve_experiment_name(experiment_name, loaded_settings)
    if resolved is None:
        return {name: str(path) for name, path in paths.items()}

    with mlflow_run(resolved, loaded_settings) as (mlflow, _run_id):
        if mlflow is None:
            return {name: str(path) for name, path in paths.items()}
        for name, path in paths.items():
            mlflow.log_artifact(str(path), artifact_path=name)
    return {name: str(path) for name, path in paths.items()}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_artifacts.py tests/test_mlflow.py tests/model/test_performance.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/artifacts.py src/lumosai/mlflow.py tests/test_artifacts.py tests/test_mlflow.py src/lumosai/model/performance.py
git commit -m "feat: add artifact and mlflow adapters"
```

---

### Task 8: Add Bias Report

**Files:**
- Create: `src/lumosai/model/bias.py`
- Create: `tests/model/test_bias.py`

- [ ] **Step 1: Write bias tests**

Write `tests/model/test_bias.py`:

```python
from __future__ import annotations

import pandas as pd

from lumosai.model.bias import bias_report


def test_bias_report_flags_group_disparity_for_classification() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 1, 1, 1, 1, 1, 1],
            "prediction": [1, 1, 1, 1, 1, 1, 0, 0],
            "segment": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute=["segment"],
        task_type="classification",
    )

    assert result.metadata["report_type"] == "bias"
    assert result.metrics["bias/flags_count"] >= 1
    assert any(flag["protected_attribute"] == "segment" for flag in result.flagged)


def test_bias_report_bins_continuous_protected_attribute() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, 1, 0, 0],
            "prediction": [1, 0, 0, 0],
            "age": [22, 35, 62, 70],
        }
    )

    result = bias_report(
        frame,
        target="actual",
        prediction="prediction",
        protected_attribute={"age": [0, 40, 120]},
        task_type="classification",
    )

    assert "age" in result.summary["by_attribute"]
    assert len(result.summary["by_attribute"]["age"]["by_group"]) == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/model/test_bias.py -v`

Expected: FAIL with missing `bias_report`.

- [ ] **Step 3: Implement bias report**

Write `src/lumosai/model/bias.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.mlflow import log_result
from lumosai.model.metrics import TaskType, compare_metric, detect_task_type, get_metrics
from lumosai.model.validation import validate_prediction_frame
from lumosai.results import LumosResult
from lumosai.settings import settings


ProtectedAttribute = list[str] | dict[str, list[float] | None]


def _normalize_protected_attribute(protected_attribute: ProtectedAttribute) -> dict[str, list[float] | None]:
    if isinstance(protected_attribute, list):
        return {column: None for column in protected_attribute}
    return protected_attribute


def _group_series(df: pd.DataFrame, column: str, bins: list[float] | None) -> pd.Series:
    if bins is None:
        return df[column].astype("object")
    return pd.cut(df[column], bins=bins, include_lowest=True).astype("object")


def _best_value(values: list[float], *, greater_is_better: bool) -> float:
    return max(values) if greater_is_better else min(values)


def bias_report(
    current: Any,
    target: str,
    prediction: str,
    protected_attribute: ProtectedAttribute,
    prediction_score: str | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    current_pd = to_pandas(current)
    validate_prediction_frame(
        current_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score,
    )
    normalized = _normalize_protected_attribute(protected_attribute)
    require_columns(current_pd, normalized.keys())
    resolved_task = task_type or detect_task_type(current_pd[target], current_pd[prediction])

    summary: dict[str, Any] = {"by_attribute": {}}
    flagged: list[dict[str, Any]] = []

    for attribute, bins in normalized.items():
        groups = _group_series(current_pd, attribute, bins)
        working = current_pd.assign(_lumos_group=groups)
        by_group: list[dict[str, Any]] = []

        for group_name, group_df in working.groupby("_lumos_group", observed=False):
            metric_values = get_metrics(
                group_df[target],
                group_df[prediction],
                y_score=group_df[prediction_score] if prediction_score else None,
                task_type=resolved_task,
                custom_metrics=custom_metrics,
            )
            if resolved_task == "classification":
                metric_values["positive_prediction_rate"] = float(pd.Series(group_df[prediction]).mean())
            else:
                residual = group_df[prediction] - group_df[target]
                metric_values["mean_residual"] = float(residual.mean())
                metric_values["mean_absolute_residual"] = float(residual.abs().mean())

            by_group.append({"group": str(group_name), "count": int(len(group_df)), **metric_values})

        comparisons: list[dict[str, Any]] = []
        metric_names = [key for key in by_group[0] if key not in {"group", "count"}] if by_group else []
        for metric_name in metric_names:
            threshold = settings.model.metric_thresholds.get(metric_name)
            greater_is_better = threshold.greater_is_better if threshold is not None else True
            values = [float(row[metric_name]) for row in by_group]
            best = _best_value(values, greater_is_better=greater_is_better)
            for row in by_group:
                comparison = compare_metric(
                    metric_name,
                    group_value=float(row[metric_name]),
                    best_value=best,
                    threshold=threshold,
                )
                comparison["group"] = row["group"]
                comparison["protected_attribute"] = attribute
                comparisons.append(comparison)
                if comparison["flagged"]:
                    flagged.append(comparison)

        summary["by_attribute"][attribute] = {
            "by_group": by_group,
            "comparisons": comparisons,
        }

    result = LumosResult(
        metrics={"bias/flags_count": float(len(flagged))},
        summary=summary,
        flagged=flagged,
        metadata={"report_type": "bias", "task_type": resolved_task},
    )
    log_result(result, experiment_name=experiment_name)
    return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/model/test_bias.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/model/bias.py tests/model/test_bias.py
git commit -m "feat: add bias report"
```

---

### Task 9: Add Profiling

**Files:**
- Create: `src/lumosai/data/profiling.py`
- Create: `tests/data/test_profiling.py`

- [ ] **Step 1: Write profiling tests**

Write `tests/data/test_profiling.py`:

```python
from __future__ import annotations

import pandas as pd

from lumosai.data.profiling import profile, temporal_sample
from lumosai.results import LumosResult


def test_temporal_sample_includes_each_period() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-03-01"]),
            "value": [1, 2, 3, 4],
        }
    )

    sampled = temporal_sample(frame, time_column="event_date", freq="M", sample_size=1)

    assert sampled["event_date"].dt.to_period("M").nunique() == 3


def test_profile_returns_lumos_result_when_report_generation_disabled(monkeypatch) -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    class FakeProfileReport:
        def __init__(self, df, minimal):
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file):
            output_file.write_text("<html>profile</html>")

    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)

    result = profile(frame)

    assert isinstance(result, LumosResult)
    assert result.metadata["report_type"] == "profile"
    assert result.summary["rows"] == 3
    assert result.report.minimal is True
    assert "html" in result.artifacts
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/data/test_profiling.py -v`

Expected: FAIL with missing `profile` and `temporal_sample`.

- [ ] **Step 3: Implement profiling**

Write `src/lumosai/data/profiling.py`:

```python
from __future__ import annotations

from typing import Any

import pandas as pd
from ydata_profiling import ProfileReport

from lumosai.artifacts import artifact_workspace
from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.mlflow import log_artifact_paths, log_result, resolve_experiment_name
from lumosai.results import LumosResult
from lumosai.settings import settings


def temporal_sample(
    df: pd.DataFrame,
    time_column: str,
    freq: str = "M",
    sample_size: int = 1000,
    min_per_period: int = 1,
) -> pd.DataFrame:
    require_columns(df, [time_column])
    working = df.copy()
    working[time_column] = pd.to_datetime(working[time_column])
    periods = working[time_column].dt.to_period(freq)

    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        n = min(len(group), max(sample_size, min_per_period))
        return group.sample(n=n, random_state=42)

    return working.groupby(periods, group_keys=False).apply(sample_group).reset_index(drop=True)


def profile(
    df: Any,
    time_column: str | None = None,
    freq: str = "M",
    sample_size: int | None = None,
    min_per_period: int = 1,
    minimal: bool | None = None,
    log_analysis: bool | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    frame = to_pandas(df)
    if time_column is None:
        profiled = frame
        resolved_minimal = settings.data.profile_minimal_default if minimal is None else minimal
        sampling_summary = {"mode": "full"}
    else:
        rows_per_period = 1000 if sample_size is None else sample_size
        profiled = temporal_sample(frame, time_column, freq, rows_per_period, min_per_period)
        resolved_minimal = False if minimal is None else minimal
        sampling_summary = {
            "mode": "temporal",
            "time_column": time_column,
            "freq": freq,
            "sample_size_per_period": rows_per_period,
            "sampled_rows": len(profiled),
        }

    report = ProfileReport(profiled, minimal=resolved_minimal)
    should_log_analysis = settings.data.log_analysis if log_analysis is None else log_analysis
    artifacts = {}
    if should_log_analysis:
        logging_requested = resolve_experiment_name(experiment_name) is not None
        with artifact_workspace(keep_local=not logging_requested) as workspace:
            html_path = workspace / "profile.html"
            report.to_file(html_path)
            artifacts["html"] = str(html_path)
            log_artifact_paths({"profile": html_path}, experiment_name=experiment_name)

    result = LumosResult(
        summary={"rows": len(profiled), "columns": list(profiled.columns), "sampling": sampling_summary},
        artifacts=artifacts,
        report=report,
        metadata={"report_type": "profile", "minimal": resolved_minimal},
    )
    log_result(result, experiment_name=experiment_name)
    return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/data/test_profiling.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/data/profiling.py tests/data/test_profiling.py
git commit -m "feat: add profiling helpers"
```

---

### Task 10: Add Drift Report

**Files:**
- Create: `src/lumosai/data/drift.py`
- Create: `tests/data/test_drift.py`

- [ ] **Step 1: Write drift tests**

Write `tests/data/test_drift.py`:

```python
from __future__ import annotations

import pandas as pd

from lumosai.data.drift import drift_report, safe_comparison_name


def test_safe_comparison_name_normalizes_metric_path_component() -> None:
    assert safe_comparison_name("Previous Window!") == "previous_window"


def test_drift_report_returns_namespaced_metrics_without_evidently(monkeypatch) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [10.0]})

    class FakeReport:
        def __init__(self, metrics):
            self.metrics = metrics

        def run(self, reference_data, current_data, column_mapping=None):
            self.reference_data = reference_data
            self.current_data = current_data

        def as_dict(self):
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": True,
                            "number_of_drifted_columns": 1,
                            "share_of_drifted_columns": 1.0,
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        comparison="Previous Window!",
    )

    assert result.metrics["drift/previous_window/n_drifted_columns"] == 1.0
    assert result.metrics["drift/previous_window/share_drifted_columns"] == 1.0
    assert result.metadata["comparison"] == "previous_window"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/data/test_drift.py -v`

Expected: FAIL with missing `drift_report`.

- [ ] **Step 3: Implement drift report**

Write `src/lumosai/data/drift.py`:

```python
from __future__ import annotations

import re
from typing import Any

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_no_duplicate_columns, validate_temporal_features
from lumosai.mlflow import log_result
from lumosai.results import LumosResult
from lumosai.settings import settings


def safe_comparison_name(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "benchmark"


def _extract_drift_summary(report_payload: dict[str, Any]) -> dict[str, Any]:
    for metric in report_payload.get("metrics", []):
        result = metric.get("result", {})
        if "number_of_drifted_columns" in result or "share_of_drifted_columns" in result:
            return {
                "dataset_drift": bool(result.get("dataset_drift", False)),
                "n_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
                "share_drifted_columns": float(result.get("share_of_drifted_columns", 0.0)),
            }
    return {"dataset_drift": False, "n_drifted_columns": 0, "share_drifted_columns": 0.0}


def drift_report(
    reference: Any,
    current: Any,
    temporal_features: list[str],
    column_mapping: Any = None,
    comparison: str = "benchmark",
    experiment_name: str | None = None,
) -> LumosResult:
    reference_pd = to_pandas(reference)
    current_pd = to_pandas(current)
    require_no_duplicate_columns(reference_pd)
    require_no_duplicate_columns(current_pd)
    validate_temporal_features(reference_pd, temporal_features)
    validate_temporal_features(current_pd, temporal_features)

    excluded = set(temporal_features)
    reference_for_drift = reference_pd.drop(columns=list(excluded), errors="ignore")
    current_for_drift = current_pd.drop(columns=list(excluded), errors="ignore")

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_for_drift,
        current_data=current_for_drift,
        column_mapping=column_mapping,
    )
    summary = _extract_drift_summary(report.as_dict())
    safe_comparison = safe_comparison_name(comparison)
    metrics = {
        f"drift/{safe_comparison}/n_drifted_columns": float(summary["n_drifted_columns"]),
        f"drift/{safe_comparison}/share_drifted_columns": float(summary["share_drifted_columns"]),
    }
    flagged = []
    if summary["share_drifted_columns"] > settings.data.drift_share_threshold:
        flagged.append(
            {
                "comparison": safe_comparison,
                "metric": "share_drifted_columns",
                "value": summary["share_drifted_columns"],
                "threshold": settings.data.drift_share_threshold,
            }
        )

    result = LumosResult(
        metrics=metrics,
        summary=summary,
        flagged=flagged,
        report=report,
        metadata={"report_type": "drift", "comparison": safe_comparison},
    )
    log_result(result, experiment_name=experiment_name)
    return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/data/test_drift.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lumosai/data/drift.py tests/data/test_drift.py
git commit -m "feat: add drift report"
```

---

### Task 11: Verify Public API And Add Documentation Recipes

**Files:**
- Create: `tests/test_public_api.py`
- Create: `docs/recipes/data-pipeline-monitoring.md`
- Create: `docs/recipes/training-pipeline-reporting.md`
- Create: `docs/recipes/ongoing-monitoring-pipeline.md`
- Create: `docs/recipes/mlflow-layout.md`

- [ ] **Step 1: Write public API tests**

Write `tests/test_public_api.py`:

```python
from __future__ import annotations

import lumosai as la
from lumosai.data import drift_report, profile
from lumosai.model import bias_report, get_metrics, performance_report


def test_top_level_exports_match_domain_exports() -> None:
    assert la.profile is profile
    assert la.drift_report is drift_report
    assert la.performance_report is performance_report
    assert la.bias_report is bias_report
    assert la.get_metrics is get_metrics
```

- [ ] **Step 2: Write recipe docs**

Write `docs/recipes/data-pipeline-monitoring.md`:

```markdown
# Data Pipeline Monitoring

Run `profile()` after feature table creation to inspect the produced dataset.
Run `drift_report()` against a stable benchmark when a new feature table or production extract is available.

```python
from lumosai.data import drift_report, profile

profile(feature_table, time_column="event_date")

drift_report(
    reference=train_benchmark,
    current=current_feature_window,
    temporal_features=["event_date", "event_month"],
    comparison="benchmark",
    experiment_name="model-monitoring",
)
```
```

Write `docs/recipes/training-pipeline-reporting.md`:

```markdown
# Training Pipeline Reporting

After scoring a holdout or validation set, use `performance_report()` and `bias_report()` to record model behavior.

```python
from lumosai.model import bias_report, performance_report

performance_report(
    validation_scored,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    experiment_name="model-training",
)

bias_report(
    validation_scored,
    target="actual",
    prediction="prediction",
    protected_attribute=["region", "segment"],
    experiment_name="model-training",
)
```
```

Write `docs/recipes/ongoing-monitoring-pipeline.md`:

```markdown
# Ongoing Monitoring Pipeline

`lumosai` evaluates the frames passed to it. It does not schedule jobs, build monitoring windows, join late labels, or own orchestration.

For each production window, run data drift. Run performance when labels are available. Run bias when labels and permitted protected attributes are available.

```python
from lumosai.data import drift_report
from lumosai.model import bias_report, performance_report

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

performance_report(
    current_window_with_labels,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
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
```

Write `docs/recipes/mlflow-layout.md`:

```markdown
# MLflow Layout

Use one experiment per monitored model or model family.
Each scheduled monitoring execution can create one run.

Metric namespaces:

- `performance/<metric>`
- `bias/...`
- `drift/<comparison>/...`

Use `drift/benchmark/...` for drift against training or stable baseline data.
Use `drift/previous_window/...` for rolling comparisons.

By default, representative datasets should be logged as metadata or external references rather than raw MLflow artifacts.
```

- [ ] **Step 3: Run public API tests**

Run: `uv run pytest tests/test_public_api.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_public_api.py docs/recipes/data-pipeline-monitoring.md docs/recipes/training-pipeline-reporting.md docs/recipes/ongoing-monitoring-pipeline.md docs/recipes/mlflow-layout.md
git commit -m "docs: add monitoring recipes"
```

---

### Task 12: Full Verification And Cleanup

**Files:**
- Modify only files needed to fix verification failures.

- [ ] **Step 1: Run formatting and lint**

Run: `uv run ruff format .`

Expected: files are formatted.

Run: `uv run ruff check .`

Expected: PASS.

- [ ] **Step 2: Run unit tests**

Run: `uv run pytest -v`

Expected: PASS.

- [ ] **Step 3: Run type checks**

Run: `uv run mypy src/lumosai`

Expected: PASS. If third-party libraries lack types, add targeted `ignore_missing_imports` overrides in `pyproject.toml` for those packages only.

- [ ] **Step 4: Inspect git status**

Run: `git status --short`

Expected: only intentional formatting or verification fixes are present.

- [ ] **Step 5: Commit verification fixes**

If formatting, lint, or type-check fixes changed files, commit them:

```bash
git add pyproject.toml src tests docs
git commit -m "chore: verify lumosai package"
```

If no files changed, do not create an empty commit.

---

## Spec Coverage Review

This plan covers the approved design spec as follows:

- uv package scaffold and `src/` layout: Task 1.
- Required Narwhals ingestion and pandas internals: Task 4.
- Nested Pydantic settings: Task 3.
- `LumosResult`: Task 2.
- MLflow adapter and auto-run semantics: Task 7.
- Local/temp artifact policy: Task 7 implements `artifact_workspace()` and Task 9 uses it for profile HTML artifacts.
- `profile()` and temporal sampling: Task 9.
- `drift_report()` with `comparison` namespacing: Task 10.
- `get_metrics()` with explicit label/score semantics: Task 5.
- `performance_report()` current-window semantics: Task 6.
- `bias_report()` group metrics, bins, comparisons, and flags: Task 8.
- Top-level and domain exports: Tasks 1 and 11.
- Documentation recipes: Task 11.
- Verification: Task 12.

Feature importance, representative sample builders, and dataset logging are intentionally excluded from v0.1 implementation and remain documented future scope.
