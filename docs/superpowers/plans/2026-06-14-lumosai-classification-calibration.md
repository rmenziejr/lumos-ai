# Lumosai Classification Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-party classification score normalization, log loss, lift deciles, and a standalone calibration report.

**Architecture:** Add `lumosai.model.scores` as the shared normalization boundary for binary, multiclass, array, and mapping probability inputs. Keep `performance_report()` as the stable production metric API, extend it with optional lift and score-label metadata, and add `calibration_report()` as a separate training/evaluation primitive. Do not wrap Evidently classification quality in this iteration.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn metrics, existing `LumosResult`, existing MLflow helpers, pytest, ruff, mypy, MkDocs.

---

## File Structure

Create:

- `src/lumosai/model/scores.py`: shared classification score normalization and metric-safe label helpers.
- `src/lumosai/model/lift.py`: first-party decile lift computation.
- `src/lumosai/model/calibration.py`: standalone `calibration_report()` primitive.
- `tests/model/test_scores.py`: score normalization tests.
- `tests/model/test_lift.py`: lift decile tests.
- `tests/model/test_calibration.py`: calibration report tests.

Modify:

- `src/lumosai/model/metrics.py`: accept normalized score arrays, add log loss, preserve existing metrics.
- `src/lumosai/model/performance.py`: accept `score_labels`, `prediction_score` mappings, and `include_lift`.
- `src/lumosai/model/__init__.py`: export `calibration_report`.
- `src/lumosai/__init__.py`: top-level lazy export for `calibration_report`.
- `tests/model/test_metrics.py`: log-loss and normalized multiclass metric coverage.
- `tests/model/test_performance.py`: performance report score-label and lift coverage.
- `tests/test_public_api.py`: public API export coverage.
- `docs/api.md`: score labels, lift, and calibration API docs.
- `docs/recipes/training-pipeline-reporting.md`: calibration example.
- `docs/recipes/pipeline-patterns.md`: score array and calibration mention.

Do not modify Evidently drift wrappers except if imports need formatting.

---

### Task 1: Add Classification Score Normalization

**Files:**

- Create: `src/lumosai/model/scores.py`
- Create: `tests/model/test_scores.py`

- [ ] **Step 1: Write failing normalization tests**

Create `tests/model/test_scores.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.scores import normalize_classification_scores, safe_label


def test_binary_1d_scores_infer_positive_label_from_sorted_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
            "score": [0.1, 0.9, 0.4, 0.2],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
    )

    assert scores.labels == [0, 1]
    assert scores.labels_inferred is True
    assert scores.positive_label == 1
    assert scores.source == "column"
    np.testing.assert_allclose(
        scores.values,
        np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]]),
    )
    assert scores.metadata()["score_labels_inferred"] is True


def test_multiclass_array_scores_use_explicit_score_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        score_labels=["bronze", "silver", "gold"],
    )

    assert scores.labels == ["bronze", "silver", "gold"]
    assert scores.labels_inferred is False
    assert scores.positive_label is None
    assert scores.source == "array"
    np.testing.assert_allclose(scores.values[0], np.array([0.2, 0.7, 0.1]))


def test_multiclass_array_scores_infer_sorted_labels_with_warning_metadata() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score="score",
    )

    assert scores.labels == ["bronze", "gold", "silver"]
    assert scores.labels_inferred is True
    assert scores.metadata()["score_label_warning"] == (
        "Multiclass score_labels were inferred by sorted labels; "
        "pass score_labels to match model.classes_."
    )


def test_multiclass_array_scores_require_matching_width() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["a", "b", "c"],
            "prediction": ["a", "b", "c"],
            "score": [[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
        }
    )

    with pytest.raises(LumosValidationError, match="score width"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
            score_labels=["a", "b", "c"],
        )


def test_unsortable_labels_require_score_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": [1, "two"],
            "prediction": [1, "two"],
            "score": [[0.8, 0.2], [0.1, 0.9]],
        }
    )

    with pytest.raises(LumosValidationError, match="pass score_labels"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score="score",
        )


def test_probability_mapping_uses_mapping_keys_as_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.2, 0.8],
        }
    )

    scores = normalize_classification_scores(
        frame,
        target="actual",
        prediction="prediction",
        prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
    )

    assert scores.labels == ["bronze", "gold"]
    assert scores.labels_inferred is False
    assert scores.source == "mapping"
    np.testing.assert_allclose(scores.values, np.array([[0.8, 0.2], [0.2, 0.8]]))


def test_probability_mapping_rejects_mismatched_score_labels() -> None:
    frame = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.2, 0.8],
        }
    )

    with pytest.raises(LumosValidationError, match="must match prediction_score mapping"):
        normalize_classification_scores(
            frame,
            target="actual",
            prediction="prediction",
            prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
            score_labels=["gold", "bronze"],
        )


def test_safe_label_normalizes_metric_path_component() -> None:
    assert safe_label("Gold Plan") == "gold_plan"
    assert safe_label("A/B") == "a_b"
    assert safe_label("") == "empty"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/model/test_scores.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lumosai.model.scores'`.

- [ ] **Step 3: Implement score normalizer**

Create `src/lumosai/model/scores.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError

ScoreInput = str | dict[Any, str]

_INFERRED_LABEL_WARNING = (
    "Multiclass score_labels were inferred by sorted labels; "
    "pass score_labels to match model.classes_."
)


@dataclass(slots=True)
class ClassificationScores:
    values: np.ndarray
    labels: list[Any]
    labels_inferred: bool
    positive_label: Any | None
    source: Literal["column", "array", "mapping"]

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "score_labels": list(self.labels),
            "score_labels_inferred": self.labels_inferred,
        }
        if self.positive_label is not None:
            metadata["positive_label"] = self.positive_label
        if self.labels_inferred:
            metadata["score_label_warning"] = _INFERRED_LABEL_WARNING
        return metadata

    def label_index(self, label: Any) -> int:
        for index, candidate in enumerate(self.labels):
            if candidate == label:
                return index
        msg = f"label is not present in score_labels: {label!r}"
        raise LumosValidationError(msg)


def safe_label(label: Any) -> str:
    text = str(label).strip().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or "empty"


def _infer_sorted_labels(frame: pd.DataFrame, *, target: str, prediction: str) -> list[Any]:
    values = pd.concat([frame[target], frame[prediction]], ignore_index=True).dropna().unique()
    try:
        return sorted(values.tolist())
    except TypeError as exc:
        msg = "could not infer sorted score_labels for mixed or unsortable labels; pass score_labels"
        raise LumosValidationError(msg) from exc


def _as_score_matrix(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    if len(values) == 0:
        msg = "prediction_score must contain at least one row"
        raise LumosValidationError(msg)
    if series.map(lambda value: isinstance(value, list | tuple | np.ndarray)).all():
        try:
            matrix = np.asarray(series.tolist(), dtype=float)
        except (TypeError, ValueError) as exc:
            msg = "prediction_score arrays must contain numeric probabilities"
            raise LumosValidationError(msg) from exc
        if matrix.ndim != 2:
            msg = "prediction_score array values must form a two-dimensional matrix"
            raise LumosValidationError(msg)
        return matrix
    try:
        return np.asarray(values, dtype=float).reshape(-1, 1)
    except (TypeError, ValueError) as exc:
        msg = "prediction_score column must contain numeric probabilities or arrays"
        raise LumosValidationError(msg) from exc


def _labels_from_score_labels(score_labels: list[Any] | None) -> list[Any] | None:
    if score_labels is None:
        return None
    labels = list(score_labels)
    if len(labels) < 2:
        msg = "score_labels must contain at least two labels"
        raise LumosValidationError(msg)
    if len(set(map(repr, labels))) != len(labels):
        msg = "score_labels must not contain duplicates"
        raise LumosValidationError(msg)
    return labels


def _normalize_mapping_scores(
    frame: pd.DataFrame,
    *,
    prediction_score: dict[Any, str],
    score_labels: list[Any] | None,
) -> ClassificationScores:
    labels = list(prediction_score)
    explicit_labels = _labels_from_score_labels(score_labels)
    if explicit_labels is not None and explicit_labels != labels:
        msg = "score_labels must match prediction_score mapping keys and order"
        raise LumosValidationError(msg)
    columns = list(prediction_score.values())
    require_columns(frame, columns)
    try:
        values = frame[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        msg = "prediction_score mapping columns must contain numeric probabilities"
        raise LumosValidationError(msg) from exc
    positive_label = labels[-1] if len(labels) == 2 else None
    return ClassificationScores(
        values=values,
        labels=labels,
        labels_inferred=False,
        positive_label=positive_label,
        source="mapping",
    )


def normalize_classification_scores(
    frame: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    prediction_score: ScoreInput,
    score_labels: list[Any] | None = None,
) -> ClassificationScores:
    require_columns(frame, [target, prediction])
    if isinstance(prediction_score, dict):
        return _normalize_mapping_scores(
            frame,
            prediction_score=prediction_score,
            score_labels=score_labels,
        )

    require_columns(frame, [prediction_score])
    raw_values = _as_score_matrix(frame[prediction_score])
    explicit_labels = _labels_from_score_labels(score_labels)

    if raw_values.shape[1] == 1:
        labels = explicit_labels or _infer_sorted_labels(frame, target=target, prediction=prediction)
        if len(labels) != 2:
            msg = "binary 1D prediction_score requires exactly two resolved score labels"
            raise LumosValidationError(msg)
        positive = labels[-1]
        positive_scores = raw_values[:, 0]
        values = np.column_stack([1.0 - positive_scores, positive_scores])
        return ClassificationScores(
            values=values,
            labels=labels,
            labels_inferred=explicit_labels is None,
            positive_label=positive,
            source="column",
        )

    labels_inferred = explicit_labels is None
    labels = explicit_labels or _infer_sorted_labels(frame, target=target, prediction=prediction)
    if raw_values.shape[1] != len(labels):
        msg = (
            "prediction_score score width must match resolved score_labels: "
            f"width={raw_values.shape[1]}, labels={len(labels)}"
        )
        raise LumosValidationError(msg)
    return ClassificationScores(
        values=raw_values,
        labels=labels,
        labels_inferred=labels_inferred,
        positive_label=labels[-1] if len(labels) == 2 else None,
        source="array",
    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/model/test_scores.py -v
```

Expected: PASS.

- [ ] **Step 5: Format, lint, type-check, commit**

Run:

```bash
uv run ruff format src/lumosai/model/scores.py tests/model/test_scores.py
uv run ruff check src/lumosai/model/scores.py tests/model/test_scores.py
uv run mypy src/lumosai/model/scores.py
git add src/lumosai/model/scores.py tests/model/test_scores.py
git commit -m "feat: normalize classification scores"
```

Expected: ruff passes and mypy reports no issues.

---

### Task 2: Add Log Loss and Score Labels to Metrics

**Files:**

- Modify: `src/lumosai/model/metrics.py`
- Modify: `tests/model/test_metrics.py`

- [ ] **Step 1: Add failing metric tests**

Append to `tests/model/test_metrics.py`:

```python
import numpy as np
import pytest
from sklearn.metrics import log_loss


def test_get_metrics_adds_log_loss_for_binary_probability_matrix() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]])

    metrics = get_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        score_labels=[0, 1],
        task_type="classification",
    )

    assert metrics["log_loss"] == pytest.approx(log_loss(y_true, y_score, labels=[0, 1]))


def test_get_metrics_adds_log_loss_for_multiclass_probability_matrix() -> None:
    y_true = pd.Series(["bronze", "silver", "gold", "gold"])
    y_pred = pd.Series(["bronze", "silver", "silver", "gold"])
    y_score = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8],
        ]
    )

    metrics = get_metrics(
        y_true,
        y_pred,
        y_score=y_score,
        score_labels=["bronze", "silver", "gold"],
        task_type="classification",
    )

    assert metrics["log_loss"] == pytest.approx(
        log_loss(y_true, y_score, labels=["bronze", "silver", "gold"])
    )
    assert "roc_auc" in metrics
```

If `pytest` is already imported, do not duplicate it. If `numpy as np` is already imported, do not duplicate it.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/model/test_metrics.py -v
```

Expected: FAIL because `get_metrics()` does not accept `score_labels`.

- [ ] **Step 3: Update metrics implementation**

Modify `src/lumosai/model/metrics.py`:

1. Add `log_loss` to the sklearn imports:

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
```

2. Update `get_metrics()` signature and classification branch:

```python
def get_metrics(
    y_true: Sequence[Any] | pd.Series,
    y_pred: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series | None = None,
    score_labels: Sequence[Any] | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
) -> dict[str, float]:
    resolved_task = task_type or detect_task_type(y_true, y_pred)
    metrics: dict[str, float] = {}

    if resolved_task == "classification":
        average = "weighted"
        zero_division = 0
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=zero_division)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=zero_division)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=zero_division)
        )
        if y_score is not None:
            metrics["roc_auc"] = _roc_auc(y_true, y_score, score_labels)
            metrics["log_loss"] = _log_loss(y_true, y_score, score_labels)
    else:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    for name, metric_func in custom_metrics or []:
        metrics[name] = float(metric_func(y_true, y_pred))

    return metrics
```

3. Replace `_roc_auc()` and add `_log_loss()`:

```python
def _roc_auc(
    y_true: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series,
    score_labels: Sequence[Any] | None = None,
) -> float:
    labels = list(score_labels) if score_labels is not None else pd.Series(y_true).dropna().unique()
    score_array = np.asarray(y_score)
    if len(labels) <= 2:
        if score_array.ndim == 2:
            if score_array.shape[1] != 2:
                msg = "binary y_score must be one-dimensional or have two probability columns"
                raise ValueError(msg)
            score_array = score_array[:, 1]
        return float(roc_auc_score(y_true, score_array))
    return float(
        roc_auc_score(
            y_true,
            score_array,
            labels=list(labels),
            multi_class="ovr",
            average="weighted",
        )
    )


def _log_loss(
    y_true: Sequence[Any] | pd.Series,
    y_score: Sequence[Any] | pd.Series,
    score_labels: Sequence[Any] | None,
) -> float:
    score_array = np.asarray(y_score)
    labels = list(score_labels) if score_labels is not None else None
    if score_array.ndim == 1:
        return float(log_loss(y_true, score_array, labels=labels))
    return float(log_loss(y_true, score_array, labels=labels))
```

- [ ] **Step 4: Run focused metric tests**

Run:

```bash
uv run pytest tests/model/test_metrics.py -v
```

Expected: PASS.

- [ ] **Step 5: Format, lint, type-check, commit**

Run:

```bash
uv run ruff format src/lumosai/model/metrics.py tests/model/test_metrics.py
uv run ruff check src/lumosai/model/metrics.py tests/model/test_metrics.py
uv run mypy src/lumosai/model/metrics.py
git add src/lumosai/model/metrics.py tests/model/test_metrics.py
git commit -m "feat: add classification log loss"
```

Expected: ruff and mypy pass.

---

### Task 3: Add Decile Lift Computation

**Files:**

- Create: `src/lumosai/model/lift.py`
- Create: `tests/model/test_lift.py`

- [ ] **Step 1: Write failing lift tests**

Create `tests/model/test_lift.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumosai.model.lift import lift_metrics
from lumosai.model.scores import ClassificationScores


def test_binary_lift_deciles_returns_top_decile_metric_and_table() -> None:
    y_true = pd.Series([1] * 5 + [0] * 15)
    scores = ClassificationScores(
        values=np.column_stack(
            [
                1 - np.array(
                    [
                        0.99,
                        0.95,
                        0.93,
                        0.91,
                        0.89,
                        0.7,
                        0.65,
                        0.6,
                        0.55,
                        0.5,
                        0.45,
                        0.4,
                        0.35,
                        0.3,
                        0.25,
                        0.2,
                        0.15,
                        0.1,
                        0.05,
                        0.01,
                    ]
                ),
                np.array(
                    [
                        0.99,
                        0.95,
                        0.93,
                        0.91,
                        0.89,
                        0.7,
                        0.65,
                        0.6,
                        0.55,
                        0.5,
                        0.45,
                        0.4,
                        0.35,
                        0.3,
                        0.25,
                        0.2,
                        0.15,
                        0.1,
                        0.05,
                        0.01,
                    ]
                ),
            ]
        ),
        labels=[0, 1],
        labels_inferred=True,
        positive_label=1,
        source="column",
    )

    metrics, summary = lift_metrics(y_true, scores)

    assert metrics["lift/positive/top_decile"] == pytest.approx(4.0)
    assert metrics["lift/positive/decile_1"] == pytest.approx(4.0)
    assert len(summary["classes"]["positive"]) == 10
    assert summary["classes"]["positive"][0]["rows"] == 2


def test_multiclass_lift_runs_one_vs_rest_per_class() -> None:
    y_true = pd.Series(["bronze", "silver", "gold", "gold", "silver", "bronze"])
    scores = ClassificationScores(
        values=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.2, 0.2, 0.6],
                [0.1, 0.7, 0.2],
                [0.6, 0.3, 0.1],
            ]
        ),
        labels=["bronze", "silver", "gold"],
        labels_inferred=False,
        positive_label=None,
        source="array",
    )

    metrics, summary = lift_metrics(y_true, scores)

    assert "lift/bronze/top_decile" in metrics
    assert "lift/silver/top_decile" in metrics
    assert "lift/gold/top_decile" in metrics
    assert set(summary["classes"]) == {"bronze", "silver", "gold"}


def test_lift_skips_class_with_no_positive_examples() -> None:
    y_true = pd.Series(["bronze", "bronze", "silver", "silver"])
    scores = ClassificationScores(
        values=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
            ]
        ),
        labels=["bronze", "silver", "gold"],
        labels_inferred=False,
        positive_label=None,
        source="array",
    )

    metrics, summary = lift_metrics(y_true, scores)

    assert not any(key.startswith("lift/gold/") for key in metrics)
    assert summary["warnings"] == ["Skipping lift for class 'gold' because it has no positives."]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/model/test_lift.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lumosai.model.lift'`.

- [ ] **Step 3: Implement lift helper**

Create `src/lumosai/model/lift.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lumosai.model.scores import ClassificationScores, safe_label


def _decile_table(events: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_events = events[order]
    baseline = float(events.mean())
    rows: list[dict[str, Any]] = []
    for decile, indices in enumerate(np.array_split(np.arange(len(sorted_events)), 10), start=1):
        decile_events = sorted_events[indices]
        event_count = int(decile_events.sum())
        event_rate = float(decile_events.mean()) if len(decile_events) else 0.0
        lift = float(event_rate / baseline) if baseline > 0 else float("nan")
        rows.append(
            {
                "decile": decile,
                "rows": int(len(decile_events)),
                "event_count": event_count,
                "event_rate": event_rate,
                "baseline_event_rate": baseline,
                "lift": lift,
            }
        )
    return rows


def lift_metrics(
    y_true: pd.Series,
    scores: ClassificationScores,
) -> tuple[dict[str, float], dict[str, Any]]:
    metrics: dict[str, float] = {}
    summary: dict[str, Any] = {"classes": {}, "warnings": []}

    labels_to_score = [scores.positive_label] if scores.positive_label is not None else scores.labels
    for label in labels_to_score:
        assert label is not None
        class_index = scores.label_index(label)
        events = (y_true == label).to_numpy(dtype=int)
        if events.sum() == 0:
            summary["warnings"].append(
                f"Skipping lift for class {label!r} because it has no positives."
            )
            continue
        class_key = "positive" if scores.positive_label is not None else safe_label(label)
        table = _decile_table(events, scores.values[:, class_index])
        summary["classes"][class_key] = table
        for row in table:
            metrics[f"lift/{class_key}/decile_{row['decile']}"] = float(row["lift"])
        metrics[f"lift/{class_key}/top_decile"] = float(table[0]["lift"])

    return metrics, summary
```

- [ ] **Step 4: Run lift tests**

Run:

```bash
uv run pytest tests/model/test_lift.py -v
```

Expected: PASS.

- [ ] **Step 5: Format, lint, type-check, commit**

Run:

```bash
uv run ruff format src/lumosai/model/lift.py tests/model/test_lift.py
uv run ruff check src/lumosai/model/lift.py tests/model/test_lift.py
uv run mypy src/lumosai/model/lift.py
git add src/lumosai/model/lift.py tests/model/test_lift.py
git commit -m "feat: add classification lift deciles"
```

Expected: ruff and mypy pass.

---

### Task 4: Extend Performance Report With Score Labels and Lift

**Files:**

- Modify: `src/lumosai/model/performance.py`
- Modify: `tests/model/test_performance.py`

- [ ] **Step 1: Add failing performance tests**

Append to `tests/model/test_performance.py`:

```python
def test_performance_report_records_explicit_score_labels_and_log_loss() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold", "gold"],
            "prediction": ["bronze", "silver", "silver", "gold"],
            "score": [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.1, 0.8],
            ],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        score_labels=["bronze", "silver", "gold"],
        task_type="classification",
    )

    assert "performance/log_loss" in result.metrics
    assert result.metadata["score_labels"] == ["bronze", "silver", "gold"]
    assert result.metadata["score_labels_inferred"] is False


def test_performance_report_infers_score_labels_with_warning_metadata() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold"],
            "prediction": ["silver", "silver", "gold"],
            "score": [[0.2, 0.1, 0.7], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        task_type="classification",
    )

    assert result.metadata["score_labels"] == ["bronze", "gold", "silver"]
    assert result.metadata["score_labels_inferred"] is True
    assert "score_label_warning" in result.metadata


def test_performance_report_accepts_probability_mapping() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "gold"],
            "prediction": ["bronze", "gold"],
            "p_bronze": [0.8, 0.2],
            "p_gold": [0.2, 0.8],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
        task_type="classification",
    )

    assert "performance/log_loss" in result.metrics
    assert result.metadata["score_labels"] == ["bronze", "gold"]


def test_performance_report_adds_lift_when_enabled() -> None:
    df = pd.DataFrame(
        {
            "actual": [1] * 5 + [0] * 15,
            "prediction": [1] * 5 + [0] * 15,
            "score": [
                0.99,
                0.95,
                0.93,
                0.91,
                0.89,
                0.7,
                0.65,
                0.6,
                0.55,
                0.5,
                0.45,
                0.4,
                0.35,
                0.3,
                0.25,
                0.2,
                0.15,
                0.1,
                0.05,
                0.01,
            ],
        }
    )

    result = performance_report(
        df,
        target="actual",
        prediction="prediction",
        prediction_score="score",
        include_lift=True,
        task_type="classification",
    )

    assert result.metrics["performance/lift/positive/top_decile"] == pytest.approx(4.0)
    assert "lift" in result.summary
```

If `pytest` is not imported at the top of `tests/model/test_performance.py`, add `import pytest`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/model/test_performance.py -v
```

Expected: FAIL because `performance_report()` does not accept `score_labels` or `include_lift`.

- [ ] **Step 3: Update performance report implementation**

Modify `src/lumosai/model/performance.py`:

1. Update imports:

```python
from lumosai.model.lift import lift_metrics
from lumosai.model.metrics import TaskType, detect_task_type, get_metrics
from lumosai.model.scores import ScoreInput, normalize_classification_scores
```

2. Remove the `_score_values()` helper.

3. Update `validate_prediction_frame()` call so mappings are validated by the score normalizer:

```python
    validate_prediction_frame(
        current_pd,
        target=target,
        prediction=prediction,
        prediction_score=prediction_score if isinstance(prediction_score, str) else None,
    )
```

4. Update signature:

```python
def performance_report(
    current: Any,
    target: str,
    prediction: str,
    prediction_score: ScoreInput | None = None,
    score_labels: list[Any] | None = None,
    task_type: TaskType | None = None,
    custom_metrics: list[tuple[str, Callable[..., float]]] | None = None,
    include_lift: bool | None = None,
    report_name: str | None = None,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    experiment_name: str | None = None,
```

5. Replace raw metric computation with:

```python
    resolved_task = task_type or detect_task_type(current_pd[target], current_pd[prediction])
    scores = (
        normalize_classification_scores(
            current_pd,
            target=target,
            prediction=prediction,
            prediction_score=prediction_score,
            score_labels=score_labels,
        )
        if resolved_task == "classification" and prediction_score is not None
        else None
    )
    raw_metrics = get_metrics(
        current_pd[target],
        current_pd[prediction],
        y_score=scores.values if scores is not None else None,
        score_labels=scores.labels if scores is not None else None,
        task_type=resolved_task,
        custom_metrics=custom_metrics,
    )
    summary: dict[str, Any] = {"rows": len(current_pd), "metrics": raw_metrics}
    if include_lift:
        if resolved_task != "classification" or scores is None:
            msg = "include_lift=True requires classification prediction_score"
            raise LumosValidationError(msg)
        lift_raw_metrics, lift_summary = lift_metrics(current_pd[target], scores)
        raw_metrics.update(lift_raw_metrics)
        summary["lift"] = lift_summary
```

6. Add score metadata before constructing `LumosResult`:

```python
    if scores is not None:
        metadata.update(scores.metadata())
```

7. Construct metrics and result from the updated `raw_metrics` and `summary`:

```python
    metrics = {f"performance/{name}": value for name, value in raw_metrics.items()}
    result = LumosResult(metrics=metrics, summary=summary, metadata=metadata)
```

- [ ] **Step 4: Run performance tests**

Run:

```bash
uv run pytest tests/model/test_performance.py -v
```

Expected: PASS.

- [ ] **Step 5: Run related model tests**

Run:

```bash
uv run pytest tests/model/test_metrics.py tests/model/test_scores.py tests/model/test_lift.py tests/model/test_performance.py -v
```

Expected: PASS.

- [ ] **Step 6: Format, lint, type-check, commit**

Run:

```bash
uv run ruff format src/lumosai/model/performance.py tests/model/test_performance.py
uv run ruff check src/lumosai/model/performance.py tests/model/test_performance.py
uv run mypy src/lumosai/model/performance.py
git add src/lumosai/model/performance.py tests/model/test_performance.py
git commit -m "feat: extend performance with score labels and lift"
```

Expected: ruff and mypy pass.

---

### Task 5: Add Calibration Report Primitive

**Files:**

- Create: `src/lumosai/model/calibration.py`
- Create: `tests/model/test_calibration.py`

- [ ] **Step 1: Write failing calibration tests**

Create `tests/model/test_calibration.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.calibration import calibration_report


def test_calibration_report_binary_returns_brier_ece_and_bins() -> None:
    df = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "score": [0.1, 0.4, 0.8, 0.9],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        score_labels=[0, 1],
        n_bins=2,
    )

    assert result.metrics["calibration/positive/brier"] == pytest.approx(0.055)
    assert result.metrics["calibration/positive/ece"] == pytest.approx(0.25)
    assert result.metrics["calibration/macro_brier"] == pytest.approx(0.055)
    assert result.metrics["calibration/macro_ece"] == pytest.approx(0.25)
    assert result.metadata["report_type"] == "calibration"
    assert result.metadata["score_labels"] == [0, 1]
    assert len(result.summary["calibration"]["classes"]["positive"]) == 2


def test_calibration_report_multiclass_returns_macro_metrics() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold", "gold"],
            "score": [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.1, 0.8],
            ],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        score_labels=["bronze", "silver", "gold"],
        n_bins=2,
        report_name="Holdout Calibration",
    )

    assert "calibration/bronze/brier" in result.metrics
    assert "calibration/silver/ece" in result.metrics
    assert "calibration/gold/ece" in result.metrics
    assert "calibration/macro_brier" in result.metrics
    assert result.metadata["report_name"] == "Holdout Calibration"


def test_calibration_report_rejects_invalid_bin_count() -> None:
    df = pd.DataFrame({"actual": [0, 1], "score": [0.2, 0.8]})

    with pytest.raises(LumosValidationError, match="n_bins"):
        calibration_report(
            df,
            target="actual",
            prediction_score="score",
            n_bins=1,
        )


def test_calibration_report_rejects_unknown_strategy() -> None:
    df = pd.DataFrame({"actual": [0, 1], "score": [0.2, 0.8]})

    with pytest.raises(LumosValidationError, match="strategy"):
        calibration_report(
            df,
            target="actual",
            prediction_score="score",
            strategy="quantile",
        )


def test_calibration_report_logs_mlflow_result(monkeypatch) -> None:
    logged: dict[str, object] = {}

    def fake_log_result(result, *, experiment_name=None, loaded_settings=None):
        logged["result"] = result
        logged["experiment_name"] = experiment_name
        return result

    monkeypatch.setattr("lumosai.model.calibration.log_result", fake_log_result)
    df = pd.DataFrame({"actual": [0, 1], "score": [0.2, 0.8]})

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        experiment_name="training",
    )

    assert logged["result"] is result
    assert logged["experiment_name"] == "training"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/model/test_calibration.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lumosai.model.calibration'`.

- [ ] **Step 3: Implement calibration report**

Create `src/lumosai/model/calibration.py`:

```python
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError
from lumosai.mlflow import log_result
from lumosai.model.scores import ScoreInput, normalize_classification_scores, safe_label
from lumosai.results import LumosResult


def _calibration_bins(
    events: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    brier = float(np.mean((probabilities - events) ** 2))
    rows: list[dict[str, Any]] = []
    total = len(events)
    ece = 0.0
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(events[mask].mean())
        else:
            mean_probability = 0.0
            observed_rate = 0.0
        absolute_error = abs(mean_probability - observed_rate)
        ece += (count / total) * absolute_error if total else 0.0
        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "rows": count,
                "mean_predicted_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": float(absolute_error),
            }
        )
    return brier, float(ece), rows


def calibration_report(
    current: Any,
    target: str,
    prediction_score: ScoreInput,
    *,
    score_labels: list[Any] | None = None,
    n_bins: int = 10,
    strategy: Literal["uniform"] = "uniform",
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    if n_bins < 2:
        msg = "n_bins must be at least 2"
        raise LumosValidationError(msg)
    if strategy != "uniform":
        msg = "strategy must be 'uniform'"
        raise LumosValidationError(msg)

    frame = to_pandas(current)
    require_columns(frame, [target])
    proxy_prediction = "__lumosai_prediction__"
    working = frame.copy()
    working[proxy_prediction] = working[target]
    scores = normalize_classification_scores(
        working,
        target=target,
        prediction=proxy_prediction,
        prediction_score=prediction_score,
        score_labels=score_labels,
    )

    metrics: dict[str, float] = {}
    class_summaries: dict[str, list[dict[str, Any]]] = {}
    brier_values: list[float] = []
    ece_values: list[float] = []
    labels_to_score = [scores.positive_label] if scores.positive_label is not None else scores.labels
    for label in labels_to_score:
        assert label is not None
        class_index = scores.label_index(label)
        class_key = "positive" if scores.positive_label is not None else safe_label(label)
        events = (working[target] == label).to_numpy(dtype=float)
        probabilities = scores.values[:, class_index]
        brier, ece, rows = _calibration_bins(events, probabilities, n_bins=n_bins)
        metrics[f"calibration/{class_key}/brier"] = brier
        metrics[f"calibration/{class_key}/ece"] = ece
        class_summaries[class_key] = rows
        brier_values.append(brier)
        ece_values.append(ece)

    metrics["calibration/macro_brier"] = float(np.mean(brier_values))
    metrics["calibration/macro_ece"] = float(np.mean(ece_values))
    metadata: dict[str, Any] = {
        "report_type": "calibration",
        "strategy": strategy,
        "n_bins": n_bins,
        **scores.metadata(),
    }
    if report_name is not None:
        metadata["report_name"] = report_name
    result = LumosResult(
        metrics=metrics,
        summary={
            "calibration": {
                "strategy": strategy,
                "n_bins": n_bins,
                "classes": class_summaries,
            }
        },
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
```

- [ ] **Step 4: Run calibration tests**

Run:

```bash
uv run pytest tests/model/test_calibration.py -v
```

Expected: PASS.

- [ ] **Step 5: Run related model tests**

Run:

```bash
uv run pytest tests/model/test_scores.py tests/model/test_calibration.py -v
```

Expected: PASS.

- [ ] **Step 6: Format, lint, type-check, commit**

Run:

```bash
uv run ruff format src/lumosai/model/calibration.py tests/model/test_calibration.py
uv run ruff check src/lumosai/model/calibration.py tests/model/test_calibration.py
uv run mypy src/lumosai/model/calibration.py
git add src/lumosai/model/calibration.py tests/model/test_calibration.py
git commit -m "feat: add calibration report"
```

Expected: ruff and mypy pass.

---

### Task 6: Export Calibration Public API

**Files:**

- Modify: `src/lumosai/model/__init__.py`
- Modify: `src/lumosai/__init__.py`
- Modify: `tests/test_public_api.py`

- [ ] **Step 1: Add failing public API tests**

Append to `tests/test_public_api.py`:

```python
def test_calibration_report_public_api() -> None:
    import lumosai
    from lumosai.model import calibration_report

    assert lumosai.calibration_report is calibration_report
    assert callable(calibration_report)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_public_api.py -v
```

Expected: FAIL because `calibration_report` is not exported.

- [ ] **Step 3: Update model and top-level lazy exports**

Modify `src/lumosai/model/__init__.py`:

```python
__all__ = [
    "bias_report",
    "calibration_report",
    "feature_importance",
    "get_metrics",
    "performance_report",
]
```

Add branch:

```python
    if name == "calibration_report":
        from lumosai.model.calibration import calibration_report

        return calibration_report
```

Modify `src/lumosai/__init__.py`:

1. Add `"calibration_report"` to `__all__`.
2. Add `"calibration_report"` to the model lazy import set:

```python
    if name in {
        "bias_report",
        "calibration_report",
        "feature_importance",
        "get_metrics",
        "performance_report",
    }:
        from lumosai import model

        return getattr(model, name)
```

- [ ] **Step 4: Run public API tests**

Run:

```bash
uv run pytest tests/test_public_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Format, lint, commit**

Run:

```bash
uv run ruff format src/lumosai/model/__init__.py src/lumosai/__init__.py tests/test_public_api.py
uv run ruff check src/lumosai/model/__init__.py src/lumosai/__init__.py tests/test_public_api.py
git add src/lumosai/model/__init__.py src/lumosai/__init__.py tests/test_public_api.py
git commit -m "feat: export calibration report api"
```

Expected: ruff passes.

---

### Task 7: Update Documentation

**Files:**

- Modify: `docs/api.md`
- Modify: `docs/recipes/training-pipeline-reporting.md`
- Modify: `docs/recipes/pipeline-patterns.md`

- [ ] **Step 1: Update API reference**

In `docs/api.md`, update `performance_report(...)` signature to include:

```python
    prediction_score=None,
    score_labels=None,
    task_type=None,
    custom_metrics=None,
    include_lift=None,
```

Add bullets under `performance_report(...)`:

```markdown
- `prediction_score` may be a numeric score column, a column of probability arrays, or a mapping of labels to probability columns.
- `score_labels` defines probability order for binary or multiclass arrays. Pass `list(model.classes_)` for sklearn-style classifiers.
- When multiclass array scores omit `score_labels`, labels are inferred by sorting observed target/prediction labels and warning metadata is recorded.
- When classification probabilities are supplied, log loss is included.
- Pass `include_lift=True` to add decile lift metrics under `performance/lift/<class>/...`.
```

Add new API section after `performance_report(...)`:

```markdown
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
- Uses one-vs-rest calibration for multiclass problems.
- Returns Brier score and expected calibration error metrics under `calibration/<class>/...`.
- Returns macro calibration metrics under `calibration/macro_brier` and `calibration/macro_ece`.
- Stores bin tables in `result.summary["calibration"]`.
```
```

- [ ] **Step 2: Update training recipe**

In `docs/recipes/training-pipeline-reporting.md`, add this example after the performance report example:

```markdown
For classifiers, store model-native probability arrays and pass the model class order:

```python
from lumosai import calibration_report

validation_scored["prediction"] = model.predict(X_validation)
validation_scored["prediction_score"] = list(model.predict_proba(X_validation))

calibration_report(
    validation_scored,
    target="actual",
    prediction_score="prediction_score",
    score_labels=list(model.classes_),
    report_name="Holdout Calibration",
    experiment_name="model-training",
)
```

Passing `score_labels=list(model.classes_)` avoids relying on sorted-label inference.
```
```

- [ ] **Step 3: Update pipeline patterns**

In `docs/recipes/pipeline-patterns.md`, add `calibration_report` to the training import:

```python
from lumosai.model import bias_report, calibration_report, feature_importance, performance_report
```

Add this call after `performance_report(...)` in the training section:

```python
calibration_report(
    validation_scored,
    target="actual",
    prediction_score="prediction_score",
    score_labels=list(model.classes_),
    report_name="Holdout Calibration",
    experiment_name="model-training",
)
```

Add one paragraph:

```markdown
Use calibration in training or evaluation jobs where labels and probabilities are available. Monitoring can trend calibration later when late-arriving labels are joined back to scored windows.
```

- [ ] **Step 4: Build docs**

Run:

```bash
uv run mkdocs build --strict
```

Expected: PASS. Existing Material/MkDocs notices and nav info for `docs/superpowers` files are acceptable.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add docs/api.md docs/recipes/training-pipeline-reporting.md docs/recipes/pipeline-patterns.md
git commit -m "docs: add calibration and lift guide"
```

---

### Task 8: Full Verification

**Files:**

- No planned file edits.

- [ ] **Step 1: Run formatter**

Run:

```bash
uv run ruff format .
```

Expected: Either no files changed, or only files from this plan are formatted.

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check .
```

Expected: PASS with `All checks passed!`.

- [ ] **Step 3: Run mypy**

Run:

```bash
uv run mypy src/lumosai
```

Expected: PASS with `Success: no issues found`.

- [ ] **Step 4: Run full tests**

Run:

```bash
uv run pytest -v
```

Expected: PASS. Dependency deprecation warnings from Evidently/Litestar are acceptable if tests pass.

- [ ] **Step 5: Run strict docs build**

Run:

```bash
uv run mkdocs build --strict
```

Expected: PASS. Existing Material/MkDocs notices and nav info for `docs/superpowers` files are acceptable.

- [ ] **Step 6: Inspect git status**

Run:

```bash
git status --short
```

Expected: clean worktree.
