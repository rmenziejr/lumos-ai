from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from lumosai.model.importance import feature_importance
from lumosai.settings import settings


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [0, 0, 1, 1],
            "noise": [3, 2, 3, 2],
            "target": [0, 0, 1, 1],
        }
    )


class _FakeExplainer:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def __call__(self, features):
        return SimpleNamespace(values=np.array([[0.1, 0.2]] * len(features)))


class _FakeMLflow:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.shap = SimpleNamespace(log_explainer=self._log_explainer)

    def _log_explainer(self, explainer, **kwargs):
        self.logged.append({"explainer": explainer, **kwargs})
        return SimpleNamespace(model_uri="runs:/run-123/shap-explainer")

    def set_experiment(self, _name):
        return None

    def active_run(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def log_metrics(self, _metrics):
        return None

    def log_dict(self, _payload, _path):
        return None

    def log_artifact(self, _path, artifact_path=None):
        return None


def _model(frame: pd.DataFrame) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=5, random_state=42).fit(
        frame[["signal", "noise"]], frame["target"]
    )


def test_feature_importance_logs_shap_explainer_to_same_mlflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=_FakeExplainer))
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: fake_mlflow)

    result = feature_importance(
        _model(frame),
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="shap",
        include_plots=False,
        experiment_name="importance-test",
        log_shap_explainer=True,
        shap_serialize_model=False,
    )

    assert len(fake_mlflow.logged) == 1
    call = fake_mlflow.logged[0]
    assert isinstance(call["explainer"], _FakeExplainer)
    assert call["name"] == "shap-explainer"
    assert call["serialize_model_using_mlflow"] is False
    assert result.artifacts["shap_explainer"]["model_uri"] == "runs:/run-123/shap-explainer"
    assert result.metadata["shap_explainer_logged"] is True
    assert result.metadata["shap_serialize_model"] is False


def test_feature_importance_logs_shap_explainer_with_html_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame()
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=_FakeExplainer))
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: fake_mlflow)
    monkeypatch.setattr(settings.artifacts, "display_cache_dir", tmp_path)

    result = feature_importance(
        _model(frame),
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="shap",
        include_plots=True,
        experiment_name="importance-test",
        log_shap_explainer=True,
    )

    assert len(fake_mlflow.logged) == 1
    assert result.metadata["mlflow_run_id"] == "run-123"
    assert result.metadata["shap_explainer_logged"] is True
    assert result.artifacts["shap_explainer"]["name"] == "shap-explainer"
    assert "html" in result.artifacts


def test_feature_importance_explicit_false_disables_shap_explainer_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=_FakeExplainer))
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: fake_mlflow)

    result = feature_importance(
        _model(frame),
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="shap",
        include_plots=False,
        experiment_name="importance-test",
        log_shap_explainer=False,
    )

    assert fake_mlflow.logged == []
    assert result.metadata["shap_explainer_logged"] is False
    assert "shap_explainer" not in result.artifacts


def test_feature_importance_defaults_shap_explainer_logging_to_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    fake_mlflow = _FakeMLflow()
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=_FakeExplainer))
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: fake_mlflow)
    monkeypatch.setattr(settings.model, "log_shap", True)

    feature_importance(
        _model(frame),
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="shap",
        include_plots=False,
        experiment_name="importance-test",
    )

    assert len(fake_mlflow.logged) == 1


def test_feature_importance_without_mlflow_does_not_try_to_log_explainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=_FakeExplainer))
    monkeypatch.setattr(settings.mlflow, "default_experiment_name", None)

    result = feature_importance(
        _model(frame),
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="shap",
        include_plots=False,
        log_shap_explainer=True,
    )

    assert result.metadata["logged_to_mlflow"] is False
    assert result.metadata["shap_explainer_logged"] is False
    assert "shap_explainer" not in result.artifacts


def test_permutation_only_never_logs_shap_explainer(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    fake_mlflow = _FakeMLflow()
    monkeypatch.setattr("lumosai.mlflow.require_mlflow", lambda: fake_mlflow)

    result = feature_importance(
        _model(frame),
        frame,
        target="target",
        feature_columns=["signal", "noise"],
        method="permutation",
        include_plots=False,
        experiment_name="importance-test",
        log_shap_explainer=True,
    )

    assert fake_mlflow.logged == []
    assert result.metadata["shap_explainer_logged"] is False
