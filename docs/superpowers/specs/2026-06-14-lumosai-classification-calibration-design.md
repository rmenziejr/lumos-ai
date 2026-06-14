# Lumosai Classification Scores, Lift, and Calibration Design

## Purpose

The next iteration should improve classification evaluation without wrapping Evidently's classification quality report. `lumosai` should support the way model probabilities naturally appear after training, especially sklearn-style `predict_proba()` arrays, and produce stable first-party metrics for performance, lift, and calibration.

The product principle is:

> Accept model-native scores. Normalize once. Report stable Lumos metrics.

This keeps `performance_report()` useful for production tracking while adding training/evaluation depth through decile lift and a standalone calibration report.

## Decision: Do Not Add Evidently Classification Quality Yet

Evidently `0.7.21` includes `ClassificationQuality`, `ClassificationQualityByLabel`, and `ClassificationPreset`. These can compute useful metrics and visualizations, but multiclass probability inputs must be represented as one probability column per class.

That does not match the preferred Lumos user workflow:

```python
df["prediction_score"] = list(model.predict_proba(X))
```

For v0.3, Lumos should own classification score normalization, lift metrics, and calibration metrics. Evidently classification quality can remain a future optional artifact if users ask for its confusion matrix, PR curve, or PR table visuals.

## Goals

1. Support model-native probability arrays for binary and multiclass classification.
2. Keep separate probability columns as a secondary escape hatch.
3. Add deterministic score-label resolution with clear warning metadata when labels are inferred.
4. Add log loss when classification probabilities are available.
5. Add decile lift metrics for binary and multiclass one-vs-rest problems.
6. Add a standalone `calibration_report()` primitive for training/evaluation workflows.
7. Keep metric names stable and MLflow-friendly.

## Non-Goals

- Do not add Evidently classification quality wrapping in v0.3.
- Do not add top-k lift, arbitrary percentiles, gains charts, or cumulative lift charts yet.
- Do not add calibration plots in v0.3.
- Do not add monitoring-bundle calibration by default.
- Do not change regression behavior except where shared validation code is touched.

## Classification Score Inputs

Classification reports should support these score input shapes:

### Binary 1D Score

```python
performance_report(
    df,
    target="actual",
    prediction="prediction",
    prediction_score="score",
)
```

`score` is interpreted as the positive-class probability. If `score_labels` is provided, it must contain the negative and positive label in probability order. If omitted, infer labels with `sorted(unique(target union prediction))` and use the last sorted label as the positive class. Metadata must record that labels were inferred.

### Binary or Multiclass Array Score

```python
performance_report(
    df,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    score_labels=model.classes_,
)
```

Each `prediction_score` value may be a list, tuple, or numpy array. For multiclass, the array width must match the number of resolved score labels.

### Probability Column Mapping

```python
performance_report(
    df,
    target="actual",
    prediction="prediction",
    prediction_score={
        "bronze": "p_bronze",
        "silver": "p_silver",
        "gold": "p_gold",
    },
)
```

This form is useful when probabilities are already split into columns. It is secondary to the array-column path but should share the same normalized internal representation.

## Score Label Resolution

Add `score_labels` to score-aware APIs:

```python
score_labels: list[Any] | None = None
```

Resolution rules:

1. If `score_labels` is provided, use it as probability order.
2. If `prediction_score` is a dict, use dict keys as labels unless `score_labels` is also provided; if both are provided, they must match.
3. If `score_labels` is omitted for a multiclass array, infer labels with `sorted(unique(target union prediction))`.
4. If labels cannot be sorted because types are mixed or not orderable, raise `LumosValidationError` asking the user to pass `score_labels`.
5. If the probability array width does not match the resolved label count, raise `LumosValidationError`.

When labels are inferred, include metadata:

```python
{
    "score_labels": [...],
    "score_labels_inferred": True,
    "score_label_warning": "Multiclass score_labels were inferred by sorted labels; pass score_labels to match model.classes_.",
}
```

When labels are provided, include:

```python
{
    "score_labels": [...],
    "score_labels_inferred": False,
}
```

## Internal Score Normalization

Add a focused helper module, `src/lumosai/model/scores.py`, to normalize score inputs once.

Required dataclass:

```python
@dataclass(slots=True)
class ClassificationScores:
    values: np.ndarray
    labels: list[Any]
    labels_inferred: bool
    positive_label: Any | None
    source: Literal["column", "array", "mapping"]
```

Expected behavior:

- `values` is always two-dimensional.
- Binary 1D scores are converted to a two-column matrix using `1 - score` for the negative class and `score` for the positive class.
- Labels preserve original values for summaries.
- Metric path labels use a sanitized string form.

## Performance Report Changes

Extend `performance_report()`:

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

Behavior:

- Existing classification metrics remain.
- ROC AUC continues to use normalized probabilities when available.
- Add log loss when classification probabilities are available.
- Add lift deciles when `include_lift=True`.
- Default `include_lift` should be `False` for v0.3 to avoid expanding metric volume unexpectedly.
- Store `score_labels` metadata when probabilities are supplied.

Metric names:

```text
performance/log_loss
performance/lift/<class>/decile_1
performance/lift/<class>/decile_2
...
performance/lift/<class>/decile_10
performance/lift/<class>/top_decile
```

For binary 1D scores without explicit labels, use `positive` for the class path and record the inference in metadata.

## Lift Deciles

Add first-party decile lift logic.

Binary behavior:

- Sort by positive-class probability descending.
- Split rows into 10 deciles as evenly as possible.
- Compute baseline event rate.
- Compute decile event rate.
- Lift is `decile_event_rate / baseline_event_rate`.

Multiclass behavior:

- Repeat binary one-vs-rest lift for each class.
- Sort by that class probability.
- Event is `target == class_label`.

Summary shape:

```python
summary["lift"] = {
    "classes": {
        "gold": [
            {
                "decile": 1,
                "rows": 100,
                "event_count": 42,
                "event_rate": 0.42,
                "baseline_event_rate": 0.18,
                "lift": 2.3333,
            },
            ...
        ]
    }
}
```

If a class has zero positive examples, skip lift metrics for that class and record a warning in summary metadata rather than raising.

## Calibration Report

Add standalone primitive:

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
) -> LumosResult
```

Behavior:

- Classification only.
- Supports binary 1D scores, array score columns, and probability column mappings through the shared score normalizer.
- Binary calibration evaluates the positive class.
- Multiclass calibration runs one-vs-rest calibration per class.
- `n_bins` must be at least 2.
- `strategy="uniform"` is the only required v0.3 strategy.

Metrics:

```text
calibration/<class>/brier
calibration/<class>/ece
calibration/macro_brier
calibration/macro_ece
```

Summary:

```python
summary["calibration"] = {
    "strategy": "uniform",
    "n_bins": 10,
    "classes": {
        "gold": [
            {
                "bin": 1,
                "lower": 0.0,
                "upper": 0.1,
                "rows": 50,
                "mean_predicted_probability": 0.06,
                "observed_rate": 0.04,
                "absolute_error": 0.02,
            },
            ...
        ]
    }
}
```

ECE should be the weighted mean absolute calibration gap across bins.
Brier should be the mean squared error between one-vs-rest event indicators and predicted probabilities.

MLflow behavior should follow other primitives: log metrics and `lumosai_result.json` through `log_result()`.

## Public API

Export `calibration_report` from:

- `lumosai.model`
- top-level `lumosai` lazy API

Do not add a new bundle in this iteration. Training-bundle integration can come after `training_report()` exists.

## Documentation

Update:

- `docs/api.md`
- `docs/recipes/training-pipeline-reporting.md`
- `docs/recipes/pipeline-patterns.md`

Docs should show sklearn-style probability arrays:

```python
scored["prediction"] = model.predict(X)
scored["prediction_score"] = list(model.predict_proba(X))

calibration_report(
    scored,
    target="actual",
    prediction_score="prediction_score",
    score_labels=list(model.classes_),
)
```

Docs should also explain the sorted-label fallback and why passing `model.classes_` is recommended.

## Testing Strategy

Add tests for:

- binary 1D score normalization;
- multiclass array score normalization with explicit `score_labels`;
- multiclass array score normalization with inferred sorted labels and warning metadata;
- unsortable mixed labels fail fast without `score_labels`;
- array width mismatch fail fast;
- dict probability mapping;
- log loss for binary and multiclass;
- binary lift deciles;
- multiclass one-vs-rest lift deciles;
- calibration report binary metrics and summary bins;
- calibration report multiclass macro metrics;
- MLflow logging for `calibration_report()`;
- public API export.

Full verification should include:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/lumosai
uv run pytest -v
uv run mkdocs build --strict
```

## Open Follow-Up: Evidently Quality

Keep Evidently classification quality out of v0.3. Reconsider it later if users need visual artifacts such as confusion matrix, PR curve, PR table, or classification quality HTML reports. If added later, Lumos should adapt normalized scores into Evidently's required per-class probability columns internally.
