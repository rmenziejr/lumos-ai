# Performance Drift

Use `performance_drift_report()` to compare a stable baseline window with a current window. The report adapts to the signals available in the data.

## Prediction Scores Before Labels

When labels have not arrived yet, compare prediction score distributions with PSI:

```python
from lumosai.model import performance_drift_report

drift = performance_drift_report(
    holdout_scores,
    production_scores,
    prediction_score="prediction_score",
    report_name="Prediction Score Drift",
    experiment_name="model-monitoring",
)

print(drift.metrics["performance_drift/baseline/score_psi"])
print(drift.flagged)
```

This is useful in early monitoring windows where model outputs are available before outcomes.

## Labels Available

When labels arrive, pass `target` and `prediction` to compare performance metrics. If `prediction_score` is also present, the report adds residual drift.

```python
drift = performance_drift_report(
    holdout_labeled,
    production_labeled,
    target="actual",
    prediction="prediction",
    prediction_score="prediction_score",
    score_labels=[0, 1],
    report_name="Labeled Performance Drift",
    experiment_name="model-monitoring",
)

print(drift.metrics["performance_drift/baseline/current/f1"])
print(drift.metrics["performance_drift/baseline/residual_psi"])
```

Classification residuals use `actual_indicator - predicted_probability`. Regression residuals use `actual - prediction`.

## Thresholds

Performance metric flags use `settings.model.metric_thresholds`. PSI flags use `settings.model.performance_drift_psi_threshold`, which defaults to `0.2`.

Override PSI for a specific report:

```python
performance_drift_report(
    holdout_scores,
    production_scores,
    prediction_score="prediction_score",
    psi_threshold=0.1,
)
```
