# Training Pipeline Reporting

After scoring a holdout or validation set, use `performance_report()`, `calibration_report()`, and `bias_report()` to record model behavior.
Install `lumosai[mlflow]` before passing `experiment_name`; omit `experiment_name` for local-only results.
To keep the reports in one MLflow run, start the run before calling `lumosai`.

```python
import mlflow

from lumosai.model import bias_report, calibration_report, performance_report

mlflow.set_experiment("model-training")

validation_scored["prediction"] = model.predict(X_validation)
validation_scored["prediction_score"] = list(model.predict_proba(X_validation))

with mlflow.start_run(run_name="training-report"):
    performance_report(
        validation_scored,
        target="actual",
        prediction="prediction",
        prediction_score="prediction_score",
        score_labels=list(model.classes_),
        include_lift=True,
        experiment_name="model-training",
    )

    calibration_report(
        validation_scored,
        target="actual",
        prediction_score="prediction_score",
        score_labels=list(model.classes_),
        report_name="Holdout Calibration",
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
