# Importance-Aware Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add importance-aware drift metrics and flags so drift reports can alert when top permutation features drift even if aggregate drift share is low.

**Architecture:** Keep the feature in `lumosai.data.drift` because it is drift-report behavior, not model-report behavior. Add small helper functions for extracting top permutation features from a `LumosResult`, extracting per-column drift decisions from Evidently payloads, and building important-feature metrics/flags. Settings live under `settings.data`; bundles stay unchanged for v1.

**Tech Stack:** Python 3.11+, pandas, Evidently report payloads, Pydantic settings, pytest, uv.

---

## File Structure

- Modify `src/lumosai/settings.py`: add data settings `important_drift_top_n` and `alert_on_important_feature_drift`.
- Modify `src/lumosai/data/drift.py`: add `important_features` and `importance_result` parameters, helper extraction logic, validation, metrics, metadata, and flags.
- Modify `tests/test_settings.py`: verify settings defaults and env overrides.
- Modify `tests/data/test_drift.py`: add failing behavior tests for explicit important features, importance-result top N, precedence, validation, and aggregate-only graceful fallback.
- Modify `docs/api.md`: document the new `drift_report()` parameters, metrics, settings, and flags.
- Modify `docs/recipes/feature-importance.md`: show how to pass a feature importance result into drift.
- Modify `docs/recipes/monitoring-bundle.md`: clarify that bundles do not automatically connect training importance to monitoring yet.

### Task 1: Add Settings Defaults

**Files:**
- Modify: `src/lumosai/settings.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write failing settings tests**

Append these tests to `tests/test_settings.py`:

```python
def test_important_drift_settings_defaults() -> None:
    loaded = Settings()

    assert loaded.data.important_drift_top_n == 10
    assert loaded.data.alert_on_important_feature_drift is True


def test_important_drift_settings_env_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LUMOSAI_DATA__IMPORTANT_DRIFT_TOP_N", "3")
    monkeypatch.setenv("LUMOSAI_DATA__ALERT_ON_IMPORTANT_FEATURE_DRIFT", "false")

    loaded = Settings()

    assert loaded.data.important_drift_top_n == 3
    assert loaded.data.alert_on_important_feature_drift is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_settings.py::test_important_drift_settings_defaults tests/test_settings.py::test_important_drift_settings_env_override -v
```

Expected: both tests fail with missing `DataSettings` attributes.

- [ ] **Step 3: Add settings fields**

In `src/lumosai/settings.py`, update `DataSettings`:

```python
class DataSettings(BaseModel):
    """Data report defaults for drift, profiling, and representative samples."""

    drift_share_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    profile_minimal_default: bool = True
    log_analysis: bool = True
    default_sample_size: int = Field(default=10000, ge=1)
    sample_artifact_format: Literal["parquet", "csv"] = "parquet"
    log_sample_metadata: bool = True
    log_sample_artifacts: bool = False
    important_drift_top_n: int = Field(default=10, ge=1)
    alert_on_important_feature_drift: bool = True
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_settings.py::test_important_drift_settings_defaults tests/test_settings.py::test_important_drift_settings_env_override -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit settings work**

Run:

```bash
git add src/lumosai/settings.py tests/test_settings.py
git commit -m "Add important drift settings"
```

### Task 2: Add Importance Feature Extraction and Validation

**Files:**
- Modify: `src/lumosai/data/drift.py`
- Test: `tests/data/test_drift.py`

- [ ] **Step 1: Write failing tests for feature selection**

Append these tests to `tests/data/test_drift.py`:

```python
def _aggregate_only_report(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            return None

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "number_of_drifted_columns": 0,
                            "share_of_drifted_columns": 0.0,
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())


def test_drift_report_rejects_important_features_outside_analysis_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5]})

    with pytest.raises(LumosValidationError, match="important_features"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            feature_columns=["x"],
            important_features=["y"],
        )


def test_drift_report_rejects_importance_result_without_permutation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5]})
    importance = LumosResult(metrics={}, summary={"methods": {"shap": {"features": []}}})

    with pytest.raises(LumosValidationError, match="permutation"):
        drift_report(
            reference,
            current,
            temporal_features=["event_date"],
            importance_result=importance,
        )


def test_drift_report_extracts_top_n_permutation_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    monkeypatch.setattr(settings.data, "important_drift_top_n", 2)
    reference = pd.DataFrame(
        {"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0], "z": [3.0]}
    )
    current = pd.DataFrame(
        {"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5], "z": [3.5]}
    )
    importance = LumosResult(
        metrics={},
        summary={
            "methods": {
                "permutation": {
                    "features": [
                        {"feature": "y", "importance_mean": 0.9},
                        {"feature": "x", "importance_mean": 0.7},
                        {"feature": "z", "importance_mean": 0.2},
                    ]
                }
            }
        },
    )

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        importance_result=importance,
    )

    assert result.metadata["important_features"] == ["y", "x"]
    assert result.metadata["important_feature_source"] == "importance_result"
```

Also add `LumosResult` to the imports at the top of `tests/data/test_drift.py`:

```python
from lumosai.results import LumosResult
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/data/test_drift.py::test_drift_report_rejects_important_features_outside_analysis_columns tests/data/test_drift.py::test_drift_report_rejects_importance_result_without_permutation_rows tests/data/test_drift.py::test_drift_report_extracts_top_n_permutation_features -v
```

Expected: failures mention unexpected `important_features` / `importance_result` keyword arguments.

- [ ] **Step 3: Add drift function parameters and helpers**

In `src/lumosai/data/drift.py`, add these helpers after `_extract_drift_summary`:

```python
def _importance_feature_rows(importance_result: LumosResult) -> list[dict[str, Any]]:
    methods = importance_result.summary.get("methods")
    if not isinstance(methods, dict):
        msg = "importance_result must include permutation rows in summary['methods']"
        raise LumosValidationError(msg)
    permutation = methods.get("permutation")
    if not isinstance(permutation, dict):
        msg = "importance_result must include permutation rows in summary['methods']['permutation']"
        raise LumosValidationError(msg)
    rows = permutation.get("features")
    if not isinstance(rows, list):
        msg = "importance_result must include permutation rows in summary['methods']['permutation']['features']"
        raise LumosValidationError(msg)
    return rows


def _important_features_from_result(importance_result: LumosResult) -> list[str]:
    features: list[str] = []
    for row in _importance_feature_rows(importance_result):
        if not isinstance(row, dict) or not isinstance(row.get("feature"), str):
            msg = "importance_result permutation rows must include string feature names"
            raise LumosValidationError(msg)
        features.append(row["feature"])
    return features[: settings.data.important_drift_top_n]


def _resolve_important_features(
    *,
    important_features: list[str] | None,
    importance_result: LumosResult | None,
    analysis_columns: pd.Index,
) -> tuple[list[str], str | None]:
    if important_features is not None:
        resolved = list(important_features)
        source = "explicit"
    elif importance_result is not None:
        resolved = _important_features_from_result(importance_result)
        source = "importance_result"
    else:
        return [], None

    missing = [feature for feature in resolved if feature not in analysis_columns]
    if missing:
        msg = "important_features must be included in analyzed drift columns: "
        msg += ", ".join(missing)
        raise LumosValidationError(msg)
    return resolved, source
```

Update the `drift_report()` signature:

```python
def drift_report(
    reference: Any,
    current: Any,
    temporal_features: list[str],
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    column_mapping: Any = None,
    comparison: str = "benchmark",
    report_name: str | None = None,
    evidently_kwargs: dict[str, Any] | None = None,
    important_features: list[str] | None = None,
    importance_result: LumosResult | None = None,
    include_html: bool = True,
    experiment_name: str | None = None,
) -> LumosResult:
```

After `selected_categorical_columns = validate_categorical_columns(...)`, add:

```python
    resolved_important_features, important_feature_source = _resolve_important_features(
        important_features=important_features,
        importance_result=importance_result,
        analysis_columns=reference_for_drift.columns,
    )
```

In `metadata`, add:

```python
        **(
            {"important_features": resolved_important_features}
            if resolved_important_features
            else {}
        ),
        **(
            {"important_feature_source": important_feature_source}
            if important_feature_source is not None
            else {}
        ),
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
uv run pytest tests/data/test_drift.py::test_drift_report_rejects_important_features_outside_analysis_columns tests/data/test_drift.py::test_drift_report_rejects_importance_result_without_permutation_rows tests/data/test_drift.py::test_drift_report_extracts_top_n_permutation_features -v
```

Expected: all pass.

- [ ] **Step 5: Commit extraction work**

Run:

```bash
git add src/lumosai/data/drift.py tests/data/test_drift.py
git commit -m "Add important feature selection for drift"
```

### Task 3: Extract Per-Column Drift Decisions

**Files:**
- Modify: `src/lumosai/data/drift.py`
- Test: `tests/data/test_drift.py`

- [ ] **Step 1: Write failing tests for metrics and flags**

Append these tests to `tests/data/test_drift.py`:

```python
def test_drift_report_flags_drifted_important_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame(
        {"event_date": ["2026-01-01"], "glucose": [100.0], "age": [40.0]}
    )
    current = pd.DataFrame(
        {"event_date": ["2026-01-02"], "glucose": [180.0], "age": [41.0]}
    )

    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            return None

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "number_of_drifted_columns": 1,
                            "share_of_drifted_columns": 0.5,
                            "drift_by_columns": {
                                "glucose": {"drift_detected": True},
                                "age": {"drift_detected": False},
                            },
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["glucose", "age"],
    )

    assert result.metrics["drift/benchmark/important_n_drifted_columns"] == 1.0
    assert result.metrics["drift/benchmark/important_share_drifted_columns"] == 0.5
    assert result.metrics["drift/benchmark/important_feature/glucose/drifted"] == 1.0
    assert result.metrics["drift/benchmark/important_feature/age/drifted"] == 0.0
    assert result.flagged == [
        {
            "comparison": "benchmark",
            "metric": "important_feature_drift",
            "feature": "glucose",
            "importance_rank": 1,
        }
    ]
    assert result.summary["important_features"] == {
        "features": ["glucose", "age"],
        "drifted_features": ["glucose"],
        "n_drifted_columns": 1,
        "share_drifted_columns": 0.5,
    }


def test_drift_report_important_feature_flags_include_importance_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "glucose": [100.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "glucose": [180.0]})

    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            return None

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "number_of_drifted_columns": 1,
                            "share_of_drifted_columns": 1.0,
                            "drift_by_columns": {"glucose": {"drift_detected": True}},
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)
    importance = LumosResult(
        metrics={},
        summary={
            "methods": {
                "permutation": {
                    "features": [{"feature": "glucose", "importance_mean": 0.9}]
                }
            }
        },
    )

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        importance_result=importance,
    )

    assert result.flagged == [
        {
            "comparison": "benchmark",
            "metric": "important_feature_drift",
            "feature": "glucose",
            "importance_method": "permutation",
            "importance_rank": 1,
        }
    ]


def test_drift_report_can_disable_important_feature_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "glucose": [100.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "glucose": [180.0]})

    class FakeReport:
        def __init__(self, metrics: list[Any]) -> None:
            self.metrics = metrics

        def run(
            self,
            reference_data: pd.DataFrame,
            current_data: pd.DataFrame,
            column_mapping: Any = None,
        ) -> None:
            return None

        def as_dict(self) -> dict[str, Any]:
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "number_of_drifted_columns": 1,
                            "share_of_drifted_columns": 1.0,
                            "drift_by_columns": {"glucose": {"drift_detected": True}},
                        }
                    }
                ]
            }

    monkeypatch.setattr("lumosai.data.drift.Report", FakeReport)
    monkeypatch.setattr("lumosai.data.drift.DataDriftPreset", lambda: object())
    monkeypatch.setattr(settings.data, "drift_share_threshold", 1.0)
    monkeypatch.setattr(settings.data, "alert_on_important_feature_drift", False)

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["glucose"],
    )

    assert result.metrics["drift/benchmark/important_feature/glucose/drifted"] == 1.0
    assert result.flagged == []
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/data/test_drift.py::test_drift_report_flags_drifted_important_features tests/data/test_drift.py::test_drift_report_important_feature_flags_include_importance_metadata tests/data/test_drift.py::test_drift_report_can_disable_important_feature_flags -v
```

Expected: tests fail because important-feature metrics and flags are not implemented.

- [ ] **Step 3: Add drift-decision extraction helpers**

In `src/lumosai/data/drift.py`, add these helpers after `_extract_drift_summary`:

```python
def _coerce_drift_detected(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _extract_column_drift_decisions(report_payload: dict[str, Any]) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for metric in report_payload.get("metrics", []):
        for container_name in ("result", "value"):
            container = metric.get(container_name, {})
            if not isinstance(container, dict):
                continue
            column_payload = container.get("drift_by_columns") or container.get("columns")
            if not isinstance(column_payload, dict):
                continue
            for feature, details in column_payload.items():
                if not isinstance(feature, str) or not isinstance(details, dict):
                    continue
                detected = (
                    _coerce_drift_detected(details.get("drift_detected"))
                    if "drift_detected" in details
                    else _coerce_drift_detected(details.get("drifted"))
                )
                if detected is not None:
                    decisions[feature] = detected
    return decisions


def _important_feature_summary(
    *,
    important_features: list[str],
    column_drift: dict[str, bool],
) -> dict[str, Any]:
    drifted_features = [feature for feature in important_features if column_drift.get(feature)]
    share = len(drifted_features) / len(important_features) if important_features else 0.0
    return {
        "features": list(important_features),
        "drifted_features": drifted_features,
        "n_drifted_columns": len(drifted_features),
        "share_drifted_columns": share,
    }
```

- [ ] **Step 4: Add metrics and flags in `drift_report()`**

Replace:

```python
    summary = _extract_drift_summary(_report_payload(report, run_result))
```

with:

```python
    report_payload = _report_payload(report, run_result)
    summary = _extract_drift_summary(report_payload)
    column_drift = _extract_column_drift_decisions(report_payload)
```

After the aggregate `metrics` dict, add:

```python
    important_summary: dict[str, Any] | None = None
    if resolved_important_features:
        important_summary = _important_feature_summary(
            important_features=resolved_important_features,
            column_drift=column_drift,
        )
        summary["important_features"] = important_summary
        metrics.update(
            {
                f"drift/{safe_comparison}/important_n_drifted_columns": float(
                    important_summary["n_drifted_columns"]
                ),
                f"drift/{safe_comparison}/important_share_drifted_columns": float(
                    important_summary["share_drifted_columns"]
                ),
            }
        )
        for feature in resolved_important_features:
            drifted = feature in important_summary["drifted_features"]
            metrics[f"drift/{safe_comparison}/important_feature/{feature}/drifted"] = (
                1.0 if drifted else 0.0
            )
```

After the existing aggregate share flag block, add:

```python
    if (
        important_summary is not None
        and settings.data.alert_on_important_feature_drift
    ):
        for rank, feature in enumerate(resolved_important_features, start=1):
            if feature not in important_summary["drifted_features"]:
                continue
            flag = {
                "comparison": safe_comparison,
                "metric": "important_feature_drift",
                "feature": feature,
                "importance_rank": rank,
            }
            if important_feature_source == "importance_result":
                flag["importance_method"] = "permutation"
            flagged.append(flag)
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
uv run pytest tests/data/test_drift.py::test_drift_report_flags_drifted_important_features tests/data/test_drift.py::test_drift_report_important_feature_flags_include_importance_metadata tests/data/test_drift.py::test_drift_report_can_disable_important_feature_flags -v
```

Expected: all pass.

- [ ] **Step 6: Commit important drift metrics**

Run:

```bash
git add src/lumosai/data/drift.py tests/data/test_drift.py
git commit -m "Add importance-aware drift metrics"
```

### Task 4: Add Precedence and Graceful Fallback Coverage

**Files:**
- Modify: `tests/data/test_drift.py`
- Modify: `src/lumosai/data/drift.py` only if tests expose a gap

- [ ] **Step 1: Write failing or confirmatory tests**

Append these tests to `tests/data/test_drift.py`:

```python
def test_drift_report_explicit_important_features_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5]})
    importance = LumosResult(
        metrics={},
        summary={
            "methods": {
                "shap": {"features": [{"feature": "x", "importance_mean": 0.9}]}
            }
        },
    )

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["y"],
        importance_result=importance,
    )

    assert result.metadata["important_features"] == ["y"]
    assert result.metadata["important_feature_source"] == "explicit"


def test_drift_report_defaults_important_feature_metrics_to_no_drift_without_column_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _aggregate_only_report(monkeypatch)
    reference = pd.DataFrame({"event_date": ["2026-01-01"], "x": [1.0], "y": [2.0]})
    current = pd.DataFrame({"event_date": ["2026-01-02"], "x": [1.5], "y": [2.5]})

    result = drift_report(
        reference,
        current,
        temporal_features=["event_date"],
        important_features=["x", "y"],
    )

    assert result.metrics["drift/benchmark/important_n_drifted_columns"] == 0.0
    assert result.metrics["drift/benchmark/important_share_drifted_columns"] == 0.0
    assert result.metrics["drift/benchmark/important_feature/x/drifted"] == 0.0
    assert result.metrics["drift/benchmark/important_feature/y/drifted"] == 0.0
    assert result.flagged == []
```

- [ ] **Step 2: Run tests**

Run:

```bash
uv run pytest tests/data/test_drift.py::test_drift_report_explicit_important_features_take_precedence tests/data/test_drift.py::test_drift_report_defaults_important_feature_metrics_to_no_drift_without_column_details -v
```

Expected: pass if Tasks 2-3 were implemented correctly. If either fails, adjust `_resolve_important_features()` or `_important_feature_summary()` to match the contract in the tests.

- [ ] **Step 3: Run all drift tests**

Run:

```bash
uv run pytest tests/data/test_drift.py -v
```

Expected: all drift tests pass.

- [ ] **Step 4: Commit coverage work**

If Task 4 only added tests, run:

```bash
git add tests/data/test_drift.py
git commit -m "Cover important drift edge cases"
```

If Task 4 required code fixes, run:

```bash
git add src/lumosai/data/drift.py tests/data/test_drift.py
git commit -m "Cover important drift edge cases"
```

### Task 5: Update API and Recipes

**Files:**
- Modify: `docs/api.md`
- Modify: `docs/recipes/feature-importance.md`
- Modify: `docs/recipes/monitoring-bundle.md`

- [ ] **Step 1: Update `docs/api.md` drift signature and bullets**

In the `drift_report(...)` API section, update the signature to include:

```python
    important_features=None,
    importance_result=None,
```

Add bullets:

```markdown
- `important_features` names model-critical features to track as a second alert lane.
- `importance_result` can be a `feature_importance()` result; `drift_report()` uses the top N permutation features from `settings.data.important_drift_top_n`.
- Explicit `important_features` take precedence over `importance_result`.
- Adds `drift/<comparison>/important_n_drifted_columns`, `drift/<comparison>/important_share_drifted_columns`, and `drift/<comparison>/important_feature/<feature>/drifted` when important features are supplied.
- Flags `important_feature_drift` when an important feature drifts and `settings.data.alert_on_important_feature_drift` is true.
```

In the settings section, add:

```markdown
- `important_drift_top_n`: number of top permutation features to pull from `importance_result`; defaults to `10`.
- `alert_on_important_feature_drift`: whether drifted important features are added to `result.flagged`; defaults to `True`.
```

- [ ] **Step 2: Update feature importance recipe**

In `docs/recipes/feature-importance.md`, add this Markdown section:

````markdown
## Use Importance for Drift Alerts

Pass an importance result into a later drift report when training and monitoring happen in the same workflow:

```python
importance = feature_importance(
    model,
    holdout,
    target="target",
    feature_columns=feature_columns,
    method="permutation",
)

drift = drift_report(
    train_benchmark,
    production_window,
    temporal_features=["event_date"],
    feature_columns=feature_columns,
    importance_result=importance,
)
```

This keeps ordinary drift-share alerts and adds important-feature drift metrics such as `drift/benchmark/important_feature/glucose/drifted`.
````

- [ ] **Step 3: Update monitoring bundle recipe**

In `docs/recipes/monitoring-bundle.md`, add:

```markdown
`monitoring_report()` does not automatically attach feature-importance results yet. For importance-aware drift alerts, call `drift_report()` directly with `important_features` or `importance_result` in the monitoring step where that information is available.
```

- [ ] **Step 4: Run docs build**

Run:

```bash
uv run mkdocs build --strict
```

Expected: docs build exits 0. Existing informational warnings about unnav'd superpowers docs are acceptable only if the command succeeds.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add docs/api.md docs/recipes/feature-importance.md docs/recipes/monitoring-bundle.md
git commit -m "Document importance-aware drift alerts"
```

### Task 6: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run formatter/linter**

Run:

```bash
uv run ruff check src tests docs
```

Expected: `All checks passed!`

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest -v
```

Expected: all tests pass. Existing dependency deprecation warnings are acceptable if the command exits 0.

- [ ] **Step 3: Run docs build**

Run:

```bash
uv run mkdocs build --strict
```

Expected: docs build exits 0.

- [ ] **Step 4: Inspect git history and status**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: branch is ahead of origin with the new implementation commits and no unstaged changes.

---

## Self-Review

- Spec coverage: settings, API parameters, metrics, flags, validation, aggregate fallback, bundle boundary, and docs are covered in Tasks 1-5.
- Placeholder scan: no placeholder markers remain; each code-changing step includes exact code or exact expected edits.
- Type consistency: `importance_result` is typed as `LumosResult | None`; `important_features` is `list[str] | None`; settings live under `settings.data`; metric names match the design spec.
