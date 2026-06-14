from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from lumosai.data.validation import require_columns
from lumosai.exceptions import LumosValidationError


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def select_analysis_frame(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    feature_columns: list[str] | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    required_columns = required_columns or []
    if target is not None:
        require_columns(df, [target])
    if feature_columns is not None:
        require_columns(df, feature_columns)
        if target is not None and target in feature_columns:
            msg = "target must not also appear in feature_columns"
            raise LumosValidationError(msg)
        columns = [column for column in [target, *feature_columns, *required_columns] if column]
        require_columns(df, required_columns)
        return df[_dedupe_preserve_order(columns)].copy()

    require_columns(df, required_columns)
    columns = list(df.columns)
    if target is not None:
        columns = [target, *[column for column in columns if column != target]]
    return df[_dedupe_preserve_order([*columns, *required_columns])].copy()


def validate_categorical_columns(
    df: pd.DataFrame,
    *,
    categorical_columns: list[str] | None,
    analysis_columns: Iterable[str] | None = None,
) -> list[str]:
    if categorical_columns is None:
        return []
    require_columns(df, categorical_columns)
    if analysis_columns is not None:
        allowed = set(analysis_columns)
        outside = [column for column in categorical_columns if column not in allowed]
        if outside:
            msg = "categorical_columns must be included in analyzed columns: "
            msg += ", ".join(outside)
            raise LumosValidationError(msg)
    return list(categorical_columns)


def filter_supported_kwargs(
    kwargs: Mapping[str, Any] | None,
    *,
    allowed: set[str],
    parameter_name: str,
    rejected: set[str] | None = None,
) -> dict[str, Any]:
    if kwargs is None:
        return {}
    rejected = rejected or set()
    keys = set(kwargs)
    rejected_keys = sorted(keys.intersection(rejected))
    if rejected_keys:
        msg = f"{parameter_name} cannot include: {', '.join(rejected_keys)}"
        raise LumosValidationError(msg)
    unsupported = sorted(keys - allowed)
    if unsupported:
        msg = f"{parameter_name} has unsupported keys: {', '.join(unsupported)}"
        raise LumosValidationError(msg)
    return dict(kwargs)
