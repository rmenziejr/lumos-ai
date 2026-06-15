# Local Documentation Design

## Goal

Create a local-first documentation structure for `lumosai` that helps users understand the package without introducing a docs build system yet.

## Scope

Add plain Markdown documentation that can later be migrated to MkDocs or GitHub Pages with minimal reshuffling.

## Files

- `docs/index.md`: documentation landing page and navigation.
- `docs/getting-started.md`: installation, first reports, and MLflow note.
- `docs/api.md`: hand-written API reference for the main public functions and settings.
- `docs/development.md`: local development and verification commands.
- Existing `docs/recipes/*.md`: remain as practical workflow examples and are linked from the docs index.

## Non-Goals

- No MkDocs config.
- No GitHub Pages workflow.
- No generated API documentation.

## Design Notes

The docs should describe current implemented behavior, including optional MLflow usage and the fact that results stay local only when neither `experiment_name` nor `settings.mlflow.default_experiment_name` is configured. The API reference should stay concise and avoid duplicating every test case.
