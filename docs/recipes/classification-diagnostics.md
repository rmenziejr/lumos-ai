# Classification Diagnostics

Scored binary classification reports include diagnostics that help translate ranking performance into operational decisions.

```python
from lumosai.model import performance_report

result = performance_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    prediction_score="risk_probability",
    task_type="classification",
    positive_label=1,
)
```

When probability scores and plots are available, the HTML report includes the standard confusion matrix, ROC curve, precision-recall curve, and lift chart, plus three utility-oriented diagnostics for binary classification.

## Observed Event Rate and Cumulative Capture

Rows are sorted from highest to lowest predicted probability and divided into score deciles.

- Bars show the observed positive-event rate in each decile.
- The cumulative-capture line shows the fraction of all observed positive events captured by targeting through that decile.
- Decile 1 is the highest-risk group.

The lift summary also exposes `cumulative_rows`, `cumulative_event_count`, `population_fraction`, and `cumulative_capture_rate` for each decile.

## Threshold Performance

The threshold-performance plot shows precision, recall/sensitivity, specificity, and F1 across probability thresholds. Use it to understand the operating trade-off created by selecting a particular intervention threshold rather than relying on a single default cutoff.

## Decision Curve Analysis

Decision curve analysis compares the model's net benefit across probability thresholds with two reference strategies:

- **Treat All**: intervene on every observation.
- **Treat None**: intervene on no observations.

For threshold probability `pt`, model net benefit is calculated as:

```text
TP / N - FP / N * pt / (1 - pt)
```

The decision curve is generated only for binary classification with probability scores. `positive_label` determines which outcome is treated as the event of interest.

## Multiclass Behavior

Multiclass reports continue to use the existing one-vs-rest ROC, precision-recall, and lift behavior. Threshold-performance and decision-curve diagnostics are intentionally binary-only in this version.
