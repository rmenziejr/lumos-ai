from __future__ import annotations

import lumosai as la
from lumosai.data import drift_report, profile
from lumosai.model import bias_report, get_metrics, performance_report
from lumosai.results import LumosResult
from lumosai.settings import settings


def test_top_level_exports_match_domain_exports() -> None:
    assert la.profile is profile
    assert la.drift_report is drift_report
    assert la.performance_report is performance_report
    assert la.bias_report is bias_report
    assert la.get_metrics is get_metrics
    assert la.LumosResult is LumosResult
    assert la.settings is settings


def test_sample_builder_public_api() -> None:
    import lumosai
    from lumosai.data import build_sample

    assert lumosai.build_sample is build_sample
