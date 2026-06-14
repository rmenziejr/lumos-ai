from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from lumosai.data.ingest import to_pandas
from lumosai.exceptions import LumosValidationError


def test_to_pandas_returns_copy_for_pandas_dataframe() -> None:
    source = pd.DataFrame({"a": [1, 2]})

    result = to_pandas(source)
    result.loc[0, "a"] = 99

    assert isinstance(result, pd.DataFrame)
    assert source.loc[0, "a"] == 1


def test_to_pandas_rejects_empty_dataframe() -> None:
    with pytest.raises(LumosValidationError, match="must contain at least one row"):
        to_pandas(pd.DataFrame({"a": []}))


def test_to_pandas_converts_polars_dataframe() -> None:
    result = to_pandas(pl.DataFrame({"a": [1, 2]}))

    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1, 2]


def test_to_pandas_rejects_duplicate_pandas_columns() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(LumosValidationError, match="duplicate columns: a"):
        to_pandas(frame)


def test_to_pandas_rejects_duplicate_non_pandas_columns() -> None:
    table = pa.table([[1], [2]], names=["a", "a"])

    with pytest.raises(LumosValidationError, match="duplicate columns"):
        to_pandas(table)


def test_to_pandas_rejects_invalid_input() -> None:
    with pytest.raises(LumosValidationError, match="Narwhals-compatible dataframe"):
        to_pandas({"a": [1]})
