from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from sklearn.inspection import permutation_importance

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.mlflow import log_result
from lumosai.results import LumosResult


def _shap_feature_importance() -> LumosResult:
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        msg = "SHAP feature importance requires the 'importance' extra."
        raise LumosOptionalDependencyError(msg) from exc

    msg = "SHAP feature importance is not implemented yet; use method='permutation'."
    raise LumosOptionalDependencyError(msg)


def feature_importance(
    model: Any,
    data: Any,
    *,
    target: str,
    feature_columns: list[str],
    method: Literal["permutation", "shap"] = "permutation",
    scoring: str | Callable[..., float] | None = None,
    n_repeats: int = 5,
    sample_size: int | None = None,
    random_state: int = 42,
    report_name: str | None = None,
    experiment_name: str | None = None,
) -> LumosResult:
    frame = to_pandas(data)
    require_columns(frame, [target, *feature_columns])
    if n_repeats < 1:
        msg = "n_repeats must be at least 1"
        raise LumosValidationError(msg)
    if sample_size is not None and sample_size < 1:
        msg = "sample_size must be at least 1"
        raise LumosValidationError(msg)
    if method not in {"permutation", "shap"}:
        msg = "method must be one of: permutation, shap"
        raise LumosValidationError(msg)

    frame_used = frame
    if sample_size is not None and sample_size < len(frame):
        frame_used = frame.sample(n=sample_size, random_state=random_state)

    if method == "shap":
        return _shap_feature_importance()

    importance = permutation_importance(
        model,
        frame_used[feature_columns],
        frame_used[target],
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            feature_columns,
            importance.importances_mean,
            importance.importances_std,
            strict=True,
        )
    ]
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)

    metadata: dict[str, Any] = {
        "report_type": "feature_importance",
        "method": method,
        "feature_columns": list(feature_columns),
    }
    if report_name is not None:
        metadata["report_name"] = report_name

    result = LumosResult(
        metrics={f"importance/{row['feature']}": row["importance_mean"] for row in rows},
        summary={"rows": len(frame_used), "features": rows},
        metadata=metadata,
    )
    log_result(result, experiment_name=experiment_name)
    return result
