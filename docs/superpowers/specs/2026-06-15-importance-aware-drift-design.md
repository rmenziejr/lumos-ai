# Importance-Aware Drift Design

## Goal

Add an importance-aware alert lane to drift reporting. Normal drift share still answers, "Did enough of the feature space move?" The new lane answers, "Did any model-critical feature move, even if total drift share is small?"

## User-Facing Behavior

`feature_importance()` already logs permutation metrics as `importance/permutation/<feature>` and stores sorted rows in `result.summary["methods"]["permutation"]["features"]`.

`drift_report()` accepts either:

```python
drift_report(
    reference,
    current,
    temporal_features=["event_date"],
    importance_result=importance,
)
```

or:

```python
drift_report(
    reference,
    current,
    temporal_features=["event_date"],
    important_features=["glucose", "bmi", "age"],
)
```

When `importance_result` is provided, `drift_report()` selects the top N permutation features using `settings.data.important_drift_top_n`. Explicit `important_features` take precedence over `importance_result`.

## Settings

Add these data settings:

```python
important_drift_top_n: int = 10
alert_on_important_feature_drift: bool = True
```

`important_drift_top_n` controls how many top permutation features are pulled from an importance result. `alert_on_important_feature_drift` controls whether drifted important features are added to `result.flagged`.

## Metrics

Drift keeps the existing aggregate metrics:

```text
drift/<comparison>/n_drifted_columns
drift/<comparison>/share_drifted_columns
```

When important features are available, add:

```text
drift/<comparison>/important_n_drifted_columns
drift/<comparison>/important_share_drifted_columns
drift/<comparison>/important_feature/<feature>/drifted
```

Per-feature drift metrics are `1.0` when the important feature drifted and `0.0` otherwise. These are intentionally metric-friendly for MLflow trend charts and alerts.

## Flagging

Existing dataset drift flagging remains based on `settings.data.drift_share_threshold`.

If `settings.data.alert_on_important_feature_drift` is true, each drifted important feature adds a flag:

```python
{
    "comparison": "benchmark",
    "metric": "important_feature_drift",
    "feature": "glucose",
    "importance_method": "permutation",
    "importance_rank": 1,
}
```

For explicit `important_features`, `importance_method` is omitted and `importance_rank` follows the caller-provided order.

## Drift Summary Source

The implementation extracts per-column drift decisions from Evidently output when available. If the installed Evidently payload shape does not expose per-column drift details, aggregate drift metrics still work and important-feature metrics default to no drift rather than guessing.

## Validation

`important_features` must be a subset of the drift analysis columns after temporal feature exclusion and `feature_columns` filtering.

`importance_result` must include permutation rows at `summary["methods"]["permutation"]["features"]`; otherwise raise `LumosValidationError` with a clear message.

If both `important_features` and `importance_result` are provided, use `important_features` and do not require permutation rows.

## Bundles

Do not automatically wire training importance into monitoring yet. Scheduled monitoring usually runs in a different process from training, so the first version keeps the low-level API explicit. A later bundle enhancement can accept `importance_result` or a saved importance feature list if users want one-call production monitoring.

## Testing

Add tests for:

- settings defaults and environment overrides;
- extracting top N permutation features from `importance_result`;
- explicit `important_features` precedence;
- validation when important features are not in drift analysis columns;
- metrics and flags when an important feature drifts;
- graceful behavior when only aggregate drift information is available.
