from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from lumosai.exceptions import LumosValidationError


def require_no_duplicate_columns(df: pd.DataFrame) -> None:
    duplicates = df.columns[df.columns.duplicated()].unique()
    if len(duplicates) > 0:
        duplicate_names = sorted(map(str, duplicates))
        msg = f"dataframe has duplicate columns: {', '.join(duplicate_names)}"
        raise LumosValidationError(msg)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"dataframe is missing required columns: {', '.join(missing)}"
        raise LumosValidationError(msg)


def validate_temporal_features(df: pd.DataFrame, temporal_features: list[str]) -> None:
    require_no_duplicate_columns(df)
    missing = [column for column in temporal_features if column not in df.columns]
    if missing:
        msg = f"temporal_features not found: {', '.join(missing)}"
        raise LumosValidationError(msg)
