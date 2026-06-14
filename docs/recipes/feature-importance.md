# Feature Importance

Use `feature_importance()` after training or evaluation to record model explainability metrics with the same `LumosResult` shape as other reports. The default method is permutation importance, which works with fitted estimators that expose a scikit-learn compatible prediction interface.

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
print(importance.summary["features"])
```

Metrics are logged as `importance/<feature>` and sorted by mean importance in the summary.

## SHAP Importance

Install the optional importance dependencies before using `method="shap"`:

```bash
uv sync --extra importance
```

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
