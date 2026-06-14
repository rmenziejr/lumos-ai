# Training Pipeline Reporting

After scoring a holdout or validation set, use `performance_report()` and `bias_report()` to record model behavior.

```python
from lumosai.model import bias_report, performance_report

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
