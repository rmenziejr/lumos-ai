# Classification metrics

Lumos reports classification metrics differently for binary and multiclass targets so the averaging semantics are explicit.

## Binary classification

`performance_report()` and `get_metrics()` accept `positive_label`, which defaults to `1`.

```python
result = performance_report(
    validation_frame,
    target="actual",
    prediction="prediction",
    prediction_score="score",
    positive_label=1,
)
```

For binary classification, the headline metrics are calculated for the configured event class:

- `accuracy`
- `precision` — precision for `positive_label`
- `recall` — recall/sensitivity for `positive_label`
- `f1` — F1 for `positive_label`
- `roc_auc` — one-vs-rest ROC AUC for `positive_label` when scores are supplied
- `pr_auc` — precision-recall AUC for `positive_label` when scores are supplied
- `log_loss` — when valid probabilities are supplied

If the default label `1` is not one of the two observed labels, pass the event label explicitly. Lumos raises an error rather than silently choosing a different positive class.

```python
result = performance_report(
    validation_frame,
    target="actual",
    prediction="prediction",
    positive_label="readmitted",
)
```

When a binary probability matrix or score-label mapping is supplied, the configured positive label is also used for ROC/PR curves, lift, and report metadata.

## Multiclass classification

Multiclass classification has no single positive class. Lumos therefore reports both macro and weighted variants explicitly:

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `weighted_precision`
- `weighted_recall`
- `weighted_f1`
- `roc_auc` and `pr_auc` when class probabilities are supplied
- `log_loss` when valid probabilities are supplied

Macro averaging gives every class equal weight. Weighted averaging weights each class by its support in the evaluation data. Reporting both makes class-level weakness less likely to be hidden by class imbalance while preserving an overall support-weighted view.

## Confusion matrix

The confusion matrix remains independent of `positive_label`. Rows are actual classes, columns are predicted classes, and each cell is the raw count for that actual/predicted combination.
