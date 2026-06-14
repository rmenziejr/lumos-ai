from __future__ import annotations

from lumosai.results import LumosResult


def log_result(result: LumosResult, *, experiment_name: str | None = None) -> LumosResult:
    result.metadata["logged_to_mlflow"] = False
    return result
