# Feature Importance

Use `feature_importance()` after training or evaluation to record model explainability metrics with the same `LumosResult` shape as other reports. The default method is `both`, which combines permutation importance and SHAP mean absolute importance because they answer different questions: permutation importance measures global model reliance on each feature, while SHAP starts from local attributions and aggregates them over the sampled rows.

Install the optional importance dependencies before using the default `method="both"` or `method="shap"`:

```bash
uv sync --extra importance
```

The default report also writes an HTML artifact with importance plots at `result.artifacts["html"]`.

## Permutation Importance

```python
from lumosai.model import feature_importance

importance = feature_importance(
    model,
    validation_frame,
    target="actual",
    feature_columns=["tenure", "plan_code", "monthly_spend", "day_of_week"],
    method="permutation",
    n_repeats=10,
    sample_size=5000,
    report_name="Holdout Feature Importance",
    experiment_name="model-training",
)

print(importance.metrics)
print(importance.summary["methods"]["permutation"]["features"])
```

Permutation metrics are logged as `importance/permutation/<feature>` and sorted by mean importance in the summary.

## SHAP Importance

```python
from lumosai.model import feature_importance

importance = feature_importance(
    model,
    validation_frame,
    target="actual",
    feature_columns=["tenure", "plan_code", "monthly_spend", "day_of_week"],
    method="shap",
    sample_size=1000,
    report_name="Holdout SHAP Importance",
    experiment_name="model-training",
)
```

SHAP support requires the optional `lumosai[importance]` dependency when the package is installed from a built distribution.

## Both Methods

```python
importance = feature_importance(
    model,
    validation_frame,
    target="actual",
    feature_columns=["tenure", "plan_code", "monthly_spend", "day_of_week"],
    method="both",
    sample_size=1000,
    report_name="Holdout Feature Importance",
    experiment_name="model-training",
)

print(importance.metrics["importance/permutation/monthly_spend"])
print(importance.metrics["importance/shap/monthly_spend"])
```

## Use Importance For Drift Alerts

Pass an importance result into a later drift report when training and monitoring happen in the same workflow:

```python
from lumosai.data import drift_report

importance = feature_importance(
    model,
    holdout,
    target="target",
    feature_columns=feature_columns,
    method="permutation",
)

drift = drift_report(
    train_benchmark,
    production_window,
    temporal_features=["event_date"],
    feature_columns=feature_columns,
    importance_result=importance,
)
```

This keeps ordinary drift-share alerts and adds important-feature drift metrics such as `drift/benchmark/important_feature/glucose/drifted`.

Set shared defaults with environment variables:

```bash
export LUMOSAI_MODEL__FEATURE_IMPORTANCE_METHOD=permutation
export LUMOSAI_MODEL__INCLUDE_FEATURE_IMPORTANCE_PLOTS=false
```
