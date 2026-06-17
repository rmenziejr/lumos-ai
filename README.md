# lumos-ai

`lumos-ai` is an opinionated Python package for ML monitoring and reporting.
It provides simple function-first APIs for profiling, drift, performance, and bias checks while returning structured results and optionally logging to MLflow.

The name comes from "lumos," the idea of lighting up what is usually hidden inside an ML workflow. The package is meant to make model behavior, data movement, drift, calibration, and monitoring signals easier to see before they become production surprises.

The Python distribution and import module are still named `lumosai`, so code examples use `import lumosai` and extras such as `lumosai[mlflow]`.

## Recipes

- [Documentation home](docs/index.md)
- [Getting started](docs/getting-started.md)
- [API reference](docs/api.md)
- [Development guide](docs/development.md)
- [Data pipeline monitoring](docs/recipes/data-pipeline-monitoring.md)
- [Training pipeline reporting](docs/recipes/training-pipeline-reporting.md)
- [Tuning and final training](docs/recipes/tuning-and-final-training.md)
- [Full Pima walkthrough](docs/recipes/full-pima-walkthrough.md)
- [Ongoing monitoring pipeline](docs/recipes/ongoing-monitoring-pipeline.md)
- [Representative samples](docs/recipes/representative-samples.md)
- [Feature importance](docs/recipes/feature-importance.md)
- [Pipeline patterns](docs/recipes/pipeline-patterns.md)
- [MLflow layout](docs/recipes/mlflow-layout.md)

Local docs can be served with:

```bash
uv run mkdocs serve
```
