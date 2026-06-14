# Data Pipeline Monitoring

Run `profile()` after feature table creation to inspect the produced dataset.
Run `drift_report()` against a stable benchmark when a new feature table or production extract is available.

```python
from lumosai.data import drift_report, profile

profile(feature_table, time_column="event_date")

drift_report(
    reference=train_benchmark,
    current=current_feature_window,
    temporal_features=["event_date", "event_month"],
    comparison="benchmark",
    experiment_name="model-monitoring",
)
```
