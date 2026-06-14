# Development Guide

## Environment

Use `uv` for local development:

```bash
uv sync
```

## Verification

Run formatting:

```bash
uv run ruff format .
```

Run lint:

```bash
uv run ruff check .
```

Run tests:

```bash
uv run pytest -v
```

Run type checks:

```bash
uv run mypy src/lumosai
```

## Package Conventions

- Public report functions return `LumosResult`.
- Dataframe inputs should enter through `lumosai.data.ingest.to_pandas`.
- User-facing validation errors should use package exception types.
- Heavy optional or unstable third-party imports should stay lazy when practical.
- MLflow logging should go through `lumosai.mlflow` helpers unless a report needs a single shared run for artifact and result logging.

## Documentation

Docs are plain Markdown for now. A future GitHub Pages setup can add MkDocs without moving the current files.
