from __future__ import annotations

import lumosai as la
from lumosai.data import drift_report, profile
from lumosai.model import bias_report, get_metrics, performance_report


def test_top_level_exports_match_domain_exports() -> None:
    assert la.profile is profile
    assert la.drift_report is drift_report
    assert la.performance_report is performance_report
    assert la.bias_report is bias_report
    assert la.get_metrics is get_metrics
