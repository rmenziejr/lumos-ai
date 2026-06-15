# lumosai Documentation

`lumosai` is a function-first Python package for ML monitoring and reporting.
It provides helpers for dataset profiling, data drift, model performance, and bias checks while returning structured `LumosResult` objects.

## Start Here

- [Getting started](getting-started.md)
- [API reference](api.md)
- [Development guide](development.md)

## Recipes

- [Data pipeline monitoring](recipes/data-pipeline-monitoring.md)
- [Training pipeline reporting](recipes/training-pipeline-reporting.md)
- [Tuning and final training](recipes/tuning-and-final-training.md)
- [Full Pima walkthrough](recipes/full-pima-walkthrough.md)
- [Ongoing monitoring pipeline](recipes/ongoing-monitoring-pipeline.md)
- [Representative samples](recipes/representative-samples.md)
- [Feature importance](recipes/feature-importance.md)
- [Pipeline patterns](recipes/pipeline-patterns.md)
- [MLflow layout](recipes/mlflow-layout.md)

## Core Concepts

- Inputs can be pandas, Polars, or other Narwhals-compatible dataframe objects.
- Report functions normalize dataframes internally and return `LumosResult`.
- `lumosai.settings` acts as the shared control point for defaults, so teams can set standards once with `LUMOSAI_` environment variables instead of repeating the same arguments in every job.
- Passing `experiment_name`, or setting `settings.mlflow.default_experiment_name`, enables MLflow logging and requires the optional MLflow dependency.
- Results stay local only when neither an explicit experiment name nor a default experiment setting is configured.
