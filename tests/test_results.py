from __future__ import annotations

from pathlib import Path

import pandas as pd

from lumosai.results import LumosResult, json_safe_artifacts


def test_lumos_result_to_dict_excludes_report_and_serializes_artifacts() -> None:
    result = LumosResult(
        metrics={"performance/f1": 0.8},
        summary={"rows": 10},
        flagged=[{"metric": "f1", "reason": "below_threshold"}],
        artifacts={"html": Path("report.html"), "frame": pd.DataFrame({"a": [1]})},
        report=object(),
        metadata={"report_type": "performance"},
    )

    payload = result.to_dict()

    assert payload == {
        "metrics": {"performance/f1": 0.8},
        "summary": {"rows": 10},
        "flagged": [{"metric": "f1", "reason": "below_threshold"}],
        "artifacts": {"html": "report.html", "frame": "<DataFrame shape=(1, 1)>"},
        "metadata": {"report_type": "performance"},
    }


def test_json_safe_artifacts_handles_nested_values() -> None:
    payload = json_safe_artifacts({"paths": [Path("a.html"), Path("b.json")]})

    assert payload == {"paths": ["a.html", "b.json"]}
