from __future__ import annotations

import pandas as pd
import pytest

from lumosai.data.validation import (
    require_columns,
    require_no_duplicate_columns,
    validate_temporal_features,
)
from lumosai.exceptions import LumosValidationError


def test_require_columns_reports_missing_columns() -> None:
    frame = pd.DataFrame({"a": [1]})

    with pytest.raises(LumosValidationError, match="missing required columns: b"):
        require_columns(frame, ["a", "b"])


def test_require_no_duplicate_columns_reports_duplicates() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(LumosValidationError, match="duplicate columns: a"):
        require_no_duplicate_columns(frame)


def test_validate_temporal_features_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"feature": [1]})

    with pytest.raises(LumosValidationError, match="temporal_features not found: event_date"):
        validate_temporal_features(frame, ["event_date"])
