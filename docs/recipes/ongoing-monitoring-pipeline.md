# Ongoing Monitoring Pipeline

`lumos-ai` evaluates the frames passed to it. It does not schedule jobs, build monitoring windows, join late labels, or own orchestration.

For each production window, run data drift. Run performance when labels are available. Run bias when labels and permitted protected attributes are available.
Install `lumosai[mlflow]` before passing `experiment_name` or setting `LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME`. Results stay local only when neither is configured.
To keep a scheduled execution in one MLflow run, start the run before calling `lumos-ai`.

```python
import mlflow

from lumosai.data import drift_report
from lumosai.model import bias_report, performance_report

mlflow.set_experiment("model-monitoring")

with mlflow.start_run(run_name="production-window"):
    drift_report(
        reference=train_benchmark,
        current=current_window,
        temporal_features=["event_date", "event_month"],
        comparison="benchmark",
        experiment_name="model-monitoring",
    )

    drift_report(
        reference=previous_window,
        current=current_window,
        temporal_features=["event_date", "event_month"],
        comparison="previous_window",
        experiment_name="model-monitoring",
    )

    performance_report(
        current_window_with_labels,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        experiment_name="model-monitoring",
    )

    bias_report(
        current_window_with_labels,
        target="actual",
        prediction="prediction",
        protected_attribute=["region", "segment"],
        experiment_name="model-monitoring",
    )
```
