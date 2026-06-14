from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd
from narwhals.exceptions import DuplicateError

from lumosai.data.validation import require_no_duplicate_columns
from lumosai.exceptions import LumosValidationError


def to_pandas(df: Any) -> pd.DataFrame:
    """Convert a Narwhals-compatible dataframe to a pandas DataFrame copy."""
    if isinstance(df, pd.DataFrame):
        require_no_duplicate_columns(df)

    try:
        native = nw.from_native(df)
        pandas_df = native.to_pandas()
    except DuplicateError as exc:
        msg = f"dataframe has duplicate columns: {exc}"
        raise LumosValidationError(msg) from exc
    except Exception as exc:
        msg = "df must be a Narwhals-compatible dataframe object"
        raise LumosValidationError(msg) from exc

    if not isinstance(pandas_df, pd.DataFrame):
        msg = "converted dataframe must be a pandas DataFrame"
        raise LumosValidationError(msg)
    if pandas_df.empty:
        msg = "dataframe must contain at least one row"
        raise LumosValidationError(msg)
    return pandas_df.copy()
