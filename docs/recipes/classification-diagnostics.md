# Classification and Performance Diagnostics

`performance_report()` renders all plots applicable to the task by default. Use the typed `PerformancePlot` enum to request only the diagnostics needed for a report.

```python
from lumosai.model import PerformancePlot, performance_report

result = performance_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    prediction_score="risk_probability",
    task_type="classification",
    positive_label=1,
    plots=[
        PerformancePlot.CAPTURE,
        PerformancePlot.THRESHOLD_PERFORMANCE,
        PerformancePlot.DECISION_CURVE,
    ],
)
```

An explicit `plots=[...]` selection takes precedence over the legacy `include_plots` flag. Omitting `plots` preserves the existing behavior: `include_plots=True` renders all applicable plots and `include_plots=False` renders none.

## Classification plot options

- `CONFUSION_MATRIX`
- `ROC`
- `PRECISION_RECALL`
- `LIFT`
- `CAPTURE`
- `THRESHOLD_PERFORMANCE`
- `DECISION_CURVE`

### Observed Event Rate and Cumulative Capture

Rows are sorted from highest to lowest predicted probability and divided into score deciles. Bars show observed positive-event rate in each decile, while the line shows the cumulative fraction of all positive events captured through that decile. The structured lift summary also exposes `cumulative_rows`, `cumulative_event_count`, `population_fraction`, and `cumulative_capture_rate`.

### Threshold Performance

Shows precision, recall/sensitivity, specificity, and F1 across probability thresholds.

### Decision Curve Analysis

Compares model net benefit with **Treat All** and **Treat None** strategies across probability thresholds. Binary `positive_label` determines the event of interest.

## Regression plot options

- `PREDICTED_VS_ACTUAL`
- `RESIDUALS_VS_PREDICTION`
- `RESIDUAL_DISTRIBUTION`
- `RESIDUAL_QQ`

The residual Q-Q plot compares ordered residuals with theoretical normal quantiles. It is useful for identifying skew, heavy tails, and outliers that are less obvious in the residual histogram.

```python
result = performance_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    task_type="regression",
    plots=[
        PerformancePlot.PREDICTED_VS_ACTUAL,
        PerformancePlot.RESIDUALS_VS_PREDICTION,
        PerformancePlot.RESIDUAL_DISTRIBUTION,
        PerformancePlot.RESIDUAL_QQ,
    ],
)
```

Classification-only plot values are rejected for regression reports and regression-only values are rejected for classification reports.
