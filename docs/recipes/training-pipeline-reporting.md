# Training Pipeline Reporting

After scoring a holdout or validation set, use `performance_report()` and `bias_report()` to record model behavior.
Install `lumosai[mlflow]` before passing `experiment_name`; omit `experiment_name` for local-only results.
To keep both reports in one MLflow run, start the run before calling `lumosai`.

```python
import mlflow

from lumosai.model import bias_report, performance_report

mlflow.set_experiment("model-training")

with mlflow.start_run(run_name="training-report"):
    performance_report(
        validation_scored,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        experiment_name="model-training",
    )

    bias_report(
        validation_scored,
        target="actual",
        prediction="prediction",
        protected_attribute=["region", "segment"],
        experiment_name="model-training",
    )
```
