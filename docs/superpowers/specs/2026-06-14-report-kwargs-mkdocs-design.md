# Report Kwargs And MkDocs Design

## Goal

Expand the report APIs with controlled engine kwargs, report naming, and schema context, then add local MkDocs documentation for SDK/API usage.

## API Changes

All report functions accept `report_name: str | None = None`. When provided, it is stored in `LumosResult.metadata["report_name"]`. `profile()` uses it as the ydata report title. `drift_report()` passes it to Evidently where the installed API supports report run names.

`profile()` accepts:

- `target: str | None = None`
- `feature_columns: list[str] | None = None`
- `categorical_columns: list[str] | None = None`
- `ydata_kwargs: dict[str, Any] | None = None`

`target` is the outcome column. When present, the profiled dataframe is ordered with `target` first. When `feature_columns` is present, only `target`, features, and any needed `time_column` are included.

`drift_report()` accepts:

- `feature_columns: list[str] | None = None`
- `categorical_columns: list[str] | None = None`
- `evidently_kwargs: dict[str, Any] | None = None`

Drift runs only over `feature_columns` when provided. Categorical columns are semantic overrides for numeric-coded categories such as `day_of_week`; for current Evidently they are passed through `DataDefinition`.

`performance_report()` and `bias_report()` accept:

- `report_name: str | None = None`
- `feature_columns: list[str] | None = None`
- `categorical_columns: list[str] | None = None`

For model reports, `target` remains the outcome column. Feature and categorical columns are validation-backed metadata for now.

## Controlled Kwargs

`ydata_kwargs` is allowlisted to:

- `explorative`
- `dark_mode`
- `config_file`
- `vars`
- `sort`
- `sensitive`

`title` and `minimal` are rejected because `report_name` and the first-class `minimal` argument own those concepts.

`evidently_kwargs` uses this shape:

```python
{
    "preset": {"drift_share": 0.4, "num_threshold": 0.01},
    "report": {"tags": ["production"]},
}
```

Allowed `preset` keys are `columns`, `drift_share`, `method`, `cat_method`, `num_method`, `text_method`, `threshold`, `cat_threshold`, `num_threshold`, `text_threshold`, and `per_column_threshold`.

Allowed `report` keys are `metadata`, `tags`, `include_tests`, `model_id`, `reference_id`, `batch_size`, and `dataset_id`.

Unsupported kwargs raise `LumosValidationError`.

## Documentation

Add local MkDocs support:

- `mkdocs.yml`
- `mkdocs` and `mkdocs-material` in the `dev` extra
- Docs nav for overview, getting started, API reference, recipes, and development
- Development docs include `uv run mkdocs serve` and `uv run mkdocs build --strict`

No GitHub Pages workflow is added yet.
