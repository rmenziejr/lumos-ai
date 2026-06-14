from __future__ import annotations

import pandas as pd

from lumosai.data.validation import require_columns, require_no_duplicate_columns


def validate_prediction_frame(
    df: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    prediction_score: str | None = None,
) -> None:
    required = [target, prediction]
    if prediction_score is not None:
        required.append(prediction_score)
    require_no_duplicate_columns(df)
    require_columns(df, required)
