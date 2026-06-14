# MLflow Layout

Use one experiment per monitored model or model family.
Each scheduled monitoring execution can create one run.

Metric namespaces:

- `performance/<metric>`
- `bias/...`
- `drift/<comparison>/...`

Use `drift/benchmark/...` for drift against training or stable baseline data.
Use `drift/previous_window/...` for rolling comparisons.

By default, representative datasets should be logged as metadata or external references rather than raw MLflow artifacts.
