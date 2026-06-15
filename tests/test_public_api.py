from __future__ import annotations

import lumosai as la
from lumosai import LumosRun
from lumosai.data import drift_report, profile
from lumosai.model import bias_report, calibration_report, get_metrics, performance_report
from lumosai.results import LumosResult
from lumosai.results import LumosRun as ResultsLumosRun
from lumosai.settings import settings


def test_top_level_exports_match_domain_exports() -> None:
    assert la.profile is profile
    assert la.drift_report is drift_report
    assert la.performance_report is performance_report
    assert la.bias_report is bias_report
    assert la.calibration_report is calibration_report
    assert la.get_metrics is get_metrics
    assert la.LumosResult is LumosResult
    assert la.LumosRun is LumosRun
    assert la.LumosRun is ResultsLumosRun
    assert la.settings is settings


def test_sample_builder_public_api() -> None:
    import lumosai
    from lumosai.data import build_sample

    assert lumosai.build_sample is build_sample


def test_feature_importance_public_api() -> None:
    import lumosai
    from lumosai.model import feature_importance

    assert lumosai.feature_importance is feature_importance


def test_calibration_report_public_api() -> None:
    import lumosai
    from lumosai.model import calibration_report

    assert lumosai.calibration_report is calibration_report
    assert callable(calibration_report)


def test_monitoring_report_public_api() -> None:
    from lumosai import monitoring_report

    assert callable(monitoring_report)


def test_training_report_public_api() -> None:
    from lumosai import training_report

    assert callable(training_report)
