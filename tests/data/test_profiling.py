from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from lumosai.data.profiling import profile, temporal_sample
from lumosai.results import LumosResult


def test_temporal_sample_includes_each_period() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01", "2026-03-01"]
            ),
            "value": [1, 2, 3, 4],
        }
    )

    sampled = temporal_sample(frame, time_column="event_date", freq="M", sample_size=1)

    assert sampled["event_date"].dt.to_period("M").nunique() == 3


def test_temporal_sample_uses_min_per_period() -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]
            ),
            "value": [1, 2, 3, 4],
        }
    )

    sampled = temporal_sample(
        frame,
        time_column="event_date",
        freq="M",
        sample_size=1,
        min_per_period=2,
    )

    assert len(sampled) == 4


def test_profile_returns_lumos_result(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    class FakeProfileReport:
        def __init__(self, df: pd.DataFrame, minimal: bool) -> None:
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file: Path) -> None:
            output_file.write_text("<html>profile</html>")

    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)

    result = profile(frame)

    assert isinstance(result, LumosResult)
    assert result.metadata["report_type"] == "profile"
    assert result.summary["rows"] == 3
    assert cast(Any, result.report).minimal is True
    assert "html" in result.artifacts


def test_profile_uses_temporal_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"]
            ),
            "value": [1, 2, 3, 4],
        }
    )

    class FakeProfileReport:
        def __init__(self, df: pd.DataFrame, minimal: bool) -> None:
            self.df = df
            self.minimal = minimal

        def to_file(self, output_file: Path) -> None:
            output_file.write_text("<html>profile</html>")

    monkeypatch.setattr("lumosai.data.profiling.ProfileReport", FakeProfileReport)

    result = profile(frame, time_column="event_date", sample_size=1)

    assert result.summary["rows"] == 2
    assert result.summary["sampling"]["mode"] == "temporal"
    assert result.metadata["minimal"] is False
