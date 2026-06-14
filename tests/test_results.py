from __future__ import annotations

from pathlib import Path

import pandas as pd

import lumosai
from lumosai.results import LumosResult, LumosRun, json_safe_artifacts


def test_public_package_imports_during_staged_implementation() -> None:
    assert lumosai.LumosResult is LumosResult


def test_wildcard_exports_import_during_staged_implementation() -> None:
    namespace: dict[str, object] = {}

    exec("from lumosai import *", namespace)  # noqa: S102

    assert namespace["LumosResult"] is LumosResult


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


def test_lumos_run_aggregates_metrics_and_flagged_items() -> None:
    run = LumosRun(
        run_type="monitoring",
        results={
            "drift_benchmark": LumosResult(
                metrics={"drift/benchmark/share": 0.5},
                flagged=[{"metric": "share", "value": 0.5}],
            ),
            "performance": LumosResult(metrics={"performance/f1": 0.8}),
        },
        metadata={"model": "churn"},
    )

    assert run.metrics == {
        "drift/benchmark/share": 0.5,
        "performance/f1": 0.8,
    }
    assert run.flagged == [{"metric": "share", "value": 0.5, "result_key": "drift_benchmark"}]


def test_lumos_run_to_dict_is_json_safe() -> None:
    run = LumosRun(
        run_type="monitoring",
        results={
            "sample": LumosResult(
                artifacts={"frame": pd.DataFrame({"x": [1, 2]})},
                metadata={"report_type": "sample"},
            )
        },
        metadata={"skipped_reports": {"bias": "protected_attribute not provided"}},
    )

    payload = run.to_dict()

    assert payload == {
        "run_type": "monitoring",
        "metrics": {},
        "flagged": [],
        "metadata": {"skipped_reports": {"bias": "protected_attribute not provided"}},
        "results": {
            "sample": {
                "metrics": {},
                "summary": {},
                "flagged": [],
                "artifacts": {"frame": "<DataFrame shape=(2, 1)>"},
                "metadata": {"report_type": "sample"},
            }
        },
    }


def test_json_safe_artifacts_handles_nested_values() -> None:
    payload = json_safe_artifacts({"paths": [Path("a.html"), Path("b.json")]})

    assert payload == {"paths": ["a.html", "b.json"]}
