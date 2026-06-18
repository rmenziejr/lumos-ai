from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from lumosai.exceptions import LumosOptionalDependencyError, LumosValidationError
from lumosai.model.importance import feature_importance
from lumosai.settings import settings


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [0, 0, 1, 1, 0, 1, 0, 1],
            "noise": [4, 3, 4, 3, 4, 3, 4, 3],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_feature_importance_returns_sorted_permutation_metrics() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    result = feature_importance(
        model,
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="permutation",
        n_repeats=3,
        random_state=42,
        report_name="importance-baseline",
    )

    features = [row["feature"] for row in result.summary["methods"]["permutation"]["features"]]

    assert features[0] == "signal"
    assert "importance/permutation/signal" in result.metrics
    assert "features" not in result.summary
    assert result.metadata["report_type"] == "feature_importance"
    assert result.metadata["method"] == "permutation"
    assert result.metadata["report_name"] == "importance-baseline"


def test_feature_importance_samples_rows_before_permutation() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    result = feature_importance(
        model,
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="permutation",
        sample_size=4,
        random_state=42,
    )

    assert result.summary["rows"] == 4


def test_feature_importance_defaults_to_both_methods_and_html_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeExplainer:
        def __init__(self, model, features):
            self.model = model
            self.features = features

        def __call__(self, features):
            return SimpleNamespace(
                values=np.array(
                    [
                        [0.6, 0.1],
                        [0.5, 0.1],
                        [0.4, 0.2],
                        [0.7, 0.1],
                        [0.3, 0.1],
                        [0.8, 0.2],
                        [0.2, 0.1],
                        [0.9, 0.2],
                    ]
                )
            )

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=FakeExplainer))
    monkeypatch.setattr(settings.artifacts, "local_dir", tmp_path)
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    result = feature_importance(
        model,
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        n_repeats=3,
        random_state=42,
        report_name="Both Importance",
    )

    assert result.metadata["method"] == "both"
    assert "importance/permutation/signal" in result.metrics
    assert "importance/shap/signal" in result.metrics
    assert set(result.summary["methods"]) == {"permutation", "shap"}
    assert "features" not in result.summary
    html_path = Path(result.artifacts["html"])
    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert "Both Importance" in html
    assert "Permutation Importance" in html
    assert "SHAP Importance" in html


def test_feature_importance_validates_columns() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosValidationError, match="missing required columns"):
        feature_importance(
            model,
            frame,
            target="missing",
            feature_columns=["signal", "absent"],
        )


def test_feature_importance_rejects_empty_feature_columns() -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosValidationError, match="feature_columns"):
        feature_importance(
            model,
            frame,
            target="target",
            feature_columns=[],
        )


def test_shap_importance_requires_optional_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("missing shap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosOptionalDependencyError, match="SHAP"):
        feature_importance(
            model,
            frame,
            target="target",
            feature_columns=["signal", "noise"],
            method="shap",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_repeats": 0}, "n_repeats"),
        ({"sample_size": 0}, "sample_size"),
    ],
)
def test_feature_importance_validates_positive_counts(kwargs, match) -> None:
    frame = make_frame()
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
        frame[["signal", "noise"]],
        frame["target"],
    )

    with pytest.raises(LumosValidationError, match=match):
        feature_importance(
            model,
            frame,
            target="target",
            feature_columns=["signal", "noise"],
            **kwargs,
        )
