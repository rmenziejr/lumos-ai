from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lumosai.exceptions import LumosValidationError
from lumosai.model.calibration import calibration_report
from lumosai.settings import settings


def test_calibration_report_binary_returns_brier_ece_and_bins() -> None:
    df = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "score": [0.1, 0.4, 0.8, 0.9],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        score_labels=[0, 1],
        n_bins=2,
    )

    assert result.metrics["calibration/positive/brier"] == pytest.approx(0.055)
    assert result.metrics["calibration/positive/ece"] == pytest.approx(0.20)
    assert result.metrics["calibration/macro_brier"] == pytest.approx(0.055)
    assert result.metrics["calibration/macro_ece"] == pytest.approx(0.20)
    assert result.metadata["report_type"] == "calibration"
    assert result.metadata["strategy"] == "uniform"
    assert result.metadata["n_bins"] == 2
    assert result.metadata["score_labels"] == [0, 1]
    assert result.metadata["positive_label"] == 1
    assert result.metadata["score_source"] == "column"
    assert len(result.summary["calibration"]["classes"]["positive"]) == 2


def test_calibration_report_creates_default_html_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    df = pd.DataFrame(
        {
            "actual": [0, 0, 1, 1],
            "score": [0.1, 0.4, 0.8, 0.9],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        score_labels=[0, 1],
        n_bins=2,
        report_name="Post Calibration",
    )

    html_path = Path(result.artifacts["html"])
    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert "Post Calibration" in html
    assert "Calibration Curve" in html
    assert "Observed Rate" in html


def test_calibration_report_binary_infers_positive_label() -> None:
    df = pd.DataFrame(
        {
            "actual": ["no", "yes", "yes", "no"],
            "score": [0.2, 0.8, 0.7, 0.1],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
    )

    assert set(result.metrics) == {
        "calibration/positive/brier",
        "calibration/positive/ece",
        "calibration/macro_brier",
        "calibration/macro_ece",
    }
    assert result.metadata["score_labels"] == ["no", "yes"]
    assert result.metadata["score_labels_inferred"] is True
    assert result.metadata["positive_label"] == "yes"


def test_calibration_report_multiclass_returns_macro_metrics() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "silver", "gold", "gold"],
            "score": [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.1, 0.8],
            ],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        score_labels=["bronze", "silver", "gold"],
        n_bins=2,
        report_name="Holdout Calibration",
    )

    assert "calibration/bronze/brier" in result.metrics
    assert "calibration/silver/ece" in result.metrics
    assert "calibration/gold/ece" in result.metrics
    assert "calibration/macro_brier" in result.metrics
    assert "calibration/macro_ece" in result.metrics
    assert result.metadata["report_name"] == "Holdout Calibration"
    assert result.metadata["score_source"] == "array"
    assert set(result.summary["calibration"]["classes"]) == {"bronze", "silver", "gold"}


def test_calibration_report_mapping_scores_use_mapping_labels() -> None:
    df = pd.DataFrame(
        {
            "actual": ["bronze", "gold", "bronze", "gold"],
            "p_bronze": [0.8, 0.2, 0.7, 0.1],
            "p_gold": [0.2, 0.8, 0.3, 0.9],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score={"bronze": "p_bronze", "gold": "p_gold"},
        n_bins=2,
    )

    assert result.metrics["calibration/positive/brier"] == pytest.approx(0.045)
    assert result.metadata["score_labels"] == ["bronze", "gold"]
    assert result.metadata["score_source"] == "mapping"
    assert result.metadata["positive_label"] == "gold"


def test_calibration_report_preserves_real_score_column_matching_proxy_name() -> None:
    df = pd.DataFrame(
        {
            "actual": [0, 1, 0, 1],
            "__lumosai_calibration_prediction__": [0.1, 0.9, 0.8, 0.2],
        }
    )

    result = calibration_report(
        df,
        target="actual",
        prediction_score="__lumosai_calibration_prediction__",
        score_labels=[0, 1],
        n_bins=2,
    )

    assert result.metrics["calibration/positive/brier"] == pytest.approx(0.325)
    assert result.metrics["calibration/positive/ece"] == pytest.approx(0.35)
    assert result.metrics["calibration/positive/brier"] != pytest.approx(0.0)
    assert result.metrics["calibration/positive/ece"] != pytest.approx(0.0)


def test_calibration_report_rejects_null_targets() -> None:
    df = pd.DataFrame({"actual": [0, None, 1], "score": [0.2, 0.5, 0.8]})

    with pytest.raises(LumosValidationError, match="target.*null"):
        calibration_report(
            df,
            target="actual",
            prediction_score="score",
            score_labels=[0, 1],
        )


def test_calibration_report_rejects_invalid_bin_count() -> None:
    df = pd.DataFrame({"actual": [0, 1], "score": [0.2, 0.8]})

    with pytest.raises(LumosValidationError, match="n_bins"):
        calibration_report(
            df,
            target="actual",
            prediction_score="score",
            n_bins=1,
        )


def test_calibration_report_rejects_unknown_strategy() -> None:
    df = pd.DataFrame({"actual": [0, 1], "score": [0.2, 0.8]})

    with pytest.raises(LumosValidationError, match="strategy"):
        calibration_report(
            df,
            target="actual",
            prediction_score="score",
            strategy="quantile",
        )


def test_calibration_report_rejects_safe_label_metric_key_collisions() -> None:
    df = pd.DataFrame(
        {
            "actual": ["A/B", "A B", "other"],
            "score": [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
        }
    )

    with pytest.raises(LumosValidationError, match="metric key collision"):
        calibration_report(
            df,
            target="actual",
            prediction_score="score",
            score_labels=["A/B", "A B", "other"],
        )


def test_calibration_report_logs_mlflow_result(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: dict[str, object] = {}

    def fake_log_result(result, *, experiment_name=None, loaded_settings=None):
        logged["result"] = result
        logged["experiment_name"] = experiment_name
        logged["loaded_settings"] = loaded_settings
        return result

    monkeypatch.setattr("lumosai.model.calibration.log_result", fake_log_result)
    df = pd.DataFrame({"actual": [0, 1], "score": [0.2, 0.8]})

    result = calibration_report(
        df,
        target="actual",
        prediction_score="score",
        include_plots=False,
        experiment_name="training",
    )

    assert logged["result"] is result
    assert logged["experiment_name"] == "training"
