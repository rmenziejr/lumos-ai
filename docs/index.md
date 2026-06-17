# lumos-ai Documentation

`lumos-ai` is a function-first Python package for ML monitoring and reporting.
It provides helpers for dataset profiling, data drift, model performance, and bias checks while returning structured `LumosResult` objects.

The name is a nod to making ML systems easier to inspect. `lumos-ai` is built to illuminate the parts of the workflow that are easy to miss: dataset quality, drift against a benchmark, model performance, calibration, bias slices, and the reporting trail around each pipeline run.

The installed Python package and import path remain `lumosai`, so code examples use `from lumosai...`.

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
