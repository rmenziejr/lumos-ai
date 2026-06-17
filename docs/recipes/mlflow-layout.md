# MLflow Layout

Use one experiment per monitored model or model family.
Each scheduled monitoring execution can create one run.
Install `lumosai[mlflow]` to enable MLflow logging.
When no MLflow run is active, each `lumos-ai` report call with `experiment_name`, or with `settings.mlflow.default_experiment_name` configured, creates its own run.
Start an MLflow run around grouped calls when one scheduled execution should produce one run.

Set `LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME` when a whole environment should log to the same default experiment without passing `experiment_name` to every report call.

Metric namespaces:

- `performance/<metric>`
- `bias/...`
- `drift/<comparison>/...`

Use `drift/benchmark/...` for drift against training or stable baseline data.
Use `drift/previous_window/...` for rolling comparisons.

By default, representative datasets should be logged as metadata or external references rather than raw MLflow artifacts.
