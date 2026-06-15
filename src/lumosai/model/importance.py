from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]

from lumosai.data.ingest import to_pandas
from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.mlflow import log_result
from lumosai.results import LumosResult


def _require_shap() -> Any:
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            "SHAP feature importance requires the optional dependency: "
            "pip install lumosai[importance]"
        )
        raise LumosOptionalDependencyError(msg) from exc

    return shap


def _shap_feature_importance(
    model: Any,
    frame_used: Any,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    shap = _require_shap()
    features = frame_used[feature_columns]
    explainer = shap.Explainer(model, features)
    values = explainer(features)
    raw_values = np.asarray(values.values)
    feature_count = len(feature_columns)

    if raw_values.ndim < 2:
        msg = "SHAP values must include a feature dimension"
        raise LumosValidationError(msg)
    if raw_values.shape[1] == feature_count:
        feature_axis = 1
    elif raw_values.shape[-1] == feature_count:
        feature_axis = raw_values.ndim - 1
    else:
        msg = "SHAP values do not match feature_columns"
        raise LumosValidationError(msg)

    abs_values = np.abs(raw_values)
    values_by_feature = np.moveaxis(abs_values, feature_axis, 0).reshape(feature_count, -1)
    mean_values = values_by_feature.mean(axis=1)
    rows: list[dict[str, Any]] = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": 0.0,
        }
        for feature, mean in zip(feature_columns, mean_values, strict=True)
    ]
    rows.sort(key=lambda row: float(row["importance_mean"]), reverse=True)
    return rows


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
    """Compute permutation or SHAP feature importance for a fitted model.

    MLflow logging is enabled when `experiment_name` is provided or
    `settings.mlflow.default_experiment_name` is set.
    """

    frame = to_pandas(data)
    if not feature_columns:
        msg = "feature_columns must contain at least one feature"
        raise LumosValidationError(msg)
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
        rows = _shap_feature_importance(model, frame_used, feature_columns)
    else:
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
