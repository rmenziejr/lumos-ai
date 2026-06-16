# Performance Drift Design

## Goal

Add one adaptive `performance_drift_report()` that compares a baseline window to a current window. Without labels it reports prediction-score distribution shift. With labels it also reports metric degradation and residual drift.

## API

```python
performance_drift_report(
    baseline,
    current,
    *,
    target=None,
    prediction=None,
    prediction_score=None,
    score_labels=None,
    task_type=None,
    comparison="baseline",
    metric_thresholds=None,
    psi_threshold=None,
    report_name=None,
    include_plots=True,
    experiment_name=None,
)
```

At least one signal is required:

- `prediction_score`, for prediction-score PSI;
- or `target` and `prediction`, for labeled performance drift.

`mode` is `prediction_only` when labels are absent and `labeled` when `target` and `prediction` are supplied.

## Metrics

Metrics are namespaced under `performance_drift/<comparison>/...`.

When `prediction_score` is provided:

```text
performance_drift/<comparison>/score_psi
```

For labeled classification with class probabilities, per-class score PSI is also emitted:

```text
performance_drift/<comparison>/score_psi/<class>
```

When `target` and `prediction` are provided:

```text
performance_drift/<comparison>/baseline/<metric>
performance_drift/<comparison>/current/<metric>
performance_drift/<comparison>/delta/<metric>
performance_drift/<comparison>/ratio/<metric>
```

When residuals can be computed:

```text
performance_drift/<comparison>/residual_psi
```

## Residuals

Regression residuals are:

```text
actual - prediction
```

Binary classification probability residuals are:

```text
actual_indicator - predicted_probability_for_positive_class
```

Multiclass residuals use true-class probability:

```text
1 - predicted_probability_for_actual_class
```

The multiclass residual is nonnegative and measures missing confidence assigned to the true class.

## PSI

PSI uses baseline quantile bins and compares the current distribution against those bins. Empty-bin shares are stabilized with a small epsilon.

Default PSI threshold is `settings.model.performance_drift_psi_threshold`, set to `0.2`. A call-level `psi_threshold` overrides the setting.

Score PSI flags use:

```python
{
    "comparison": comparison,
    "metric": "score_psi",
    "value": value,
    "threshold": threshold,
}
```

Residual PSI flags use `metric: "residual_psi"`.

## Metric Drift

Metric drift uses `settings.model.metric_thresholds` by default. The caller can pass `metric_thresholds` to override or extend thresholds for this report.

For each metric present in both baseline and current:

- `delta` is current minus baseline;
- `ratio` is current divided by baseline when the baseline is positive;
- flags follow existing `MetricThreshold` semantics through `compare_metric()`.

Flags use:

```python
{
    "comparison": comparison,
    "metric": "metric_drift",
    "performance_metric": metric_name,
    "baseline": baseline_value,
    "current": current_value,
    "delta": delta,
    "ratio": ratio,
    "threshold": threshold,
    "comparison_mode": comparison_mode,
}
```

## Plots

With `include_plots=True`, the report writes an HTML artifact with:

- metric delta table when labels are available;
- baseline/current score distribution when scores are available;
- baseline/current residual distribution when residuals are available;
- classification probability residual scatter for the current window when binary probabilities are available;
- regression residual scatter for the current window when regression labels are available.

## Module Placement

Create `src/lumosai/model/performance_drift.py` and export `performance_drift_report` from `lumosai.model` and top-level `lumosai`.

Add plot helpers to `src/lumosai/model/plots.py`.

## Validation

- `prediction_score` without labels supports numeric score columns and score mappings. Array-valued score columns require labels and predictions so existing score normalization can resolve labels.
- `target` and `prediction` must be provided together.
- `score_labels` require `prediction_score`.
- Baseline and current frames must contain all requested columns.
- PSI requires at least one finite value in each compared distribution.

## Testing

Add tests for:

- prediction-only score PSI and flags;
- labeled classification metric drift, score PSI, residual PSI, and probability residual summary;
- regression metric drift and residual PSI;
- validation for missing signals and partial labeled inputs;
- public API exports;
- HTML artifact generation.
