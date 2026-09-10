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

Permutation metrics are logged as `importance/permutation/<feature>` and sorted by mean importance in the summary. Permutation-only runs never create or log a SHAP explainer, even if `log_shap_explainer=True` is passed.

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

### Save the SHAP explainer to MLflow

When SHAP importance is computed, Lumos can persist the same `shap.Explainer` instance used for the importance calculation as an MLflow model. No second explainer is created.

```python
importance = feature_importance(
    model,
    validation_frame,
    target="actual",
    feature_columns=feature_columns,
    method="shap",
    experiment_name="model-training",
    log_shap_explainer=True,
)

print(importance.artifacts["shap_explainer"]["model_uri"])
```

The explainer is logged with the stable MLflow model name `shap-explainer` in the same run as the importance metrics and report artifacts. The resulting model URI is stored in `result.artifacts["shap_explainer"]["model_uri"]`.

`log_shap_explainer=None` follows `settings.model.log_shap`, which defaults to `True`. Passing `False` disables explainer persistence for a specific call. An MLflow experiment or active Lumos MLflow run is still required; without MLflow logging configured, Lumos computes SHAP importance normally and does not attempt to persist the explainer.

By default, MLflow serializes the explainer's underlying model using its native MLflow flavor. MLflow currently supports this path for scikit-learn and PyTorch models. For other model types, use SHAP's internal model serialization instead:

```python
importance = feature_importance(
    model,
    validation_frame,
    target="actual",
    feature_columns=feature_columns,
    method="shap",
    experiment_name="model-training",
    log_shap_explainer=True,
    shap_serialize_model=False,
)
```

The saved explainer can later be restored with MLflow's SHAP flavor:

```python
import mlflow

explainer = mlflow.shap.load_explainer(
    importance.artifacts["shap_explainer"]["model_uri"]
)
```

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

When `method="both"`, the SHAP explainer logging controls work the same way as `method="shap"`.

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
export LUMOSAI_MODEL__LOG_SHAP=true
```
