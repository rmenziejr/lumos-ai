from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ArtifactSettings(BaseModel):
    """Artifact retention defaults for reports that create local files."""

    keep_local: bool = False
    local_dir: Path | None = None


class MLflowSettings(BaseModel):
    """MLflow connection and logging defaults shared by report functions."""

    tracking_uri: str | None = None
    username: str | None = None
    password: str | None = None
    default_experiment_name: str | None = None
    run_mode: Literal["auto", "require_active"] = "auto"
    log_artifacts: bool = True
    log_dicts: bool = True


class DataSettings(BaseModel):
    """Data report defaults for drift, profiling, and representative samples."""

    drift_share_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    profile_minimal_default: bool = True
    log_analysis: bool = True
    default_sample_size: int = Field(default=10000, ge=1)
    sample_artifact_format: Literal["parquet", "csv"] = "parquet"
    log_sample_metadata: bool = True
    log_sample_artifacts: bool = False


class MetricThreshold(BaseModel):
    """Threshold definition used by model and bias comparisons."""

    mode: Literal["relative", "absolute"] = "relative"
    value: float
    greater_is_better: bool = True


def default_metric_thresholds() -> dict[str, MetricThreshold]:
    """Return default metric thresholds for model quality comparisons."""

    return {
        "precision": MetricThreshold(mode="relative", value=0.8, greater_is_better=True),
        "recall": MetricThreshold(mode="relative", value=0.8, greater_is_better=True),
        "f1": MetricThreshold(mode="relative", value=0.8, greater_is_better=True),
        "positive_prediction_rate": MetricThreshold(
            mode="relative",
            value=0.8,
            greater_is_better=True,
        ),
        "mae": MetricThreshold(mode="relative", value=1.25, greater_is_better=False),
        "rmse": MetricThreshold(mode="relative", value=1.25, greater_is_better=False),
    }


class ModelSettings(BaseModel):
    """Model report defaults for metrics, thresholds, and importance."""

    classification_metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )
    classification_probability_metrics: list[str] = Field(default_factory=lambda: ["roc_auc"])
    regression_metrics: list[str] = Field(default_factory=lambda: ["mae", "rmse", "r2"])
    metric_thresholds: dict[str, MetricThreshold] = Field(default_factory=default_metric_thresholds)
    include_perm_importance: bool = True
    log_shap: bool = True
    shap_sample_size: int = 1000

    @model_validator(mode="before")
    @classmethod
    def merge_metric_threshold_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "metric_thresholds" not in data:
            return data

        defaults = {
            name: threshold.model_dump() for name, threshold in default_metric_thresholds().items()
        }
        merged = defaults.copy()
        provided = data["metric_thresholds"] or {}

        for metric, threshold in provided.items():
            metric_key = str(metric)
            threshold_data = (
                threshold.model_dump()
                if isinstance(threshold, MetricThreshold)
                else dict(threshold)
            )
            merged[metric_key] = {**defaults.get(metric_key, {}), **threshold_data}

        return {**data, "metric_thresholds": merged}


class BundleSettings(BaseModel):
    """Bundle defaults that control grouped training and monitoring reports."""

    include_profile_in_training: bool = False
    include_feature_importance_in_training: bool = True
    include_previous_window_drift: bool = True
    fail_fast: bool = True


class Settings(BaseSettings):
    """Global lumosai settings loaded from `LUMOSAI_` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="LUMOSAI_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    bundles: BundleSettings = Field(default_factory=BundleSettings)


settings = Settings()
