# Tuning and Final Training

Use `performance_report()` inside each Optuna trial to make trial quality visible in MLflow. Then run a richer final training report after the best parameters are selected. This keeps tuning runs lightweight while preserving detailed model diagnostics for the selected model.

This recipe assumes the training pipeline already owns feature engineering, train/validation splits, cross-validation folds, model registration, and promotion decisions.
Install `lumosai[mlflow]` and Optuna before running the examples.

## Tune With Nested Runs

Create one parent MLflow run for the study. Each Optuna trial gets a nested run with its parameters and a validation `performance_report()`.

```python
import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier

from lumosai.model import performance_report

EXPERIMENT_NAME = "churn-training"

mlflow.set_experiment(EXPERIMENT_NAME)


def build_model(params):
    return RandomForestClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
    )


def score_validation(model, validation_frame, feature_columns):
    scored = validation_frame.copy()
    scored["prediction"] = model.predict(scored[feature_columns])
    scored["prediction_score"] = list(model.predict_proba(scored[feature_columns]))
    return scored


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
    }

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)

        model = build_model(params)
        model.fit(train_frame[feature_columns], train_frame[target])

        validation_scored = score_validation(model, validation_frame, feature_columns)
        result = performance_report(
            validation_scored,
            target=target,
            prediction="prediction",
            prediction_score="prediction_score",
            score_labels=list(model.classes_),
            include_lift=True,
            feature_columns=feature_columns,
            report_name=f"Trial {trial.number} Validation Performance",
            experiment_name=EXPERIMENT_NAME,
        )

        f1 = result.metrics["performance/f1"]
        trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        return f1


with mlflow.start_run(run_name="optuna-study"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
    mlflow.log_metric("best_validation_f1", study.best_value)
```

When `experiment_name` is passed and an MLflow run is already active, `lumosai` logs into that active run. The nested trial run therefore receives the `performance/...` metrics and result artifact for that trial.

## Add Fold-Level Reports

For cross-validation, keep the trial run as the parent and create a nested run for each fold. This is useful when a tuning job needs visibility into fold variance, not just the trial mean.

```python
import numpy as np

from lumosai.model import performance_report


def objective_with_folds(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
    }
    fold_scores = []

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(params)

        for fold_index, (train_index, valid_index) in enumerate(cv.split(train_frame, train_frame[target])):
            with mlflow.start_run(run_name=f"fold-{fold_index}", nested=True):
                fold_train = train_frame.iloc[train_index]
                fold_valid = train_frame.iloc[valid_index]

                model = build_model(params)
                model.fit(fold_train[feature_columns], fold_train[target])

                fold_scored = score_validation(model, fold_valid, feature_columns)
                result = performance_report(
                    fold_scored,
                    target=target,
                    prediction="prediction",
                    prediction_score="prediction_score",
                    score_labels=list(model.classes_),
                    include_lift=True,
                    feature_columns=feature_columns,
                    report_name=f"Trial {trial.number} Fold {fold_index} Performance",
                    experiment_name=EXPERIMENT_NAME,
                )
                fold_scores.append(result.metrics["performance/f1"])

        mean_f1 = float(np.mean(fold_scores))
        mlflow.log_metric("mean_fold_f1", mean_f1)
        mlflow.log_metric("std_fold_f1", float(np.std(fold_scores)))
        return mean_f1
```

Fold-level reports add MLflow runs and artifacts, so they are best for smaller studies, late-stage searches, or debugging unstable trials. For broad sweeps, log only trial-level performance.

## Final Training Report

After selecting parameters, fit the final model and run the standard training bundle in one MLflow run. This final run is the durable record for the selected model.

```python
import mlflow
import mlflow.sklearn

from lumosai import training_report
from lumosai.model import calibration_report

best_model = build_model(study.best_params)
best_model.fit(final_train_frame[feature_columns], final_train_frame[target])

holdout_scored = score_validation(best_model, holdout_frame, feature_columns)

with mlflow.start_run(run_name="final-training-report"):
    mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})

    run = training_report(
        final_train_frame,
        holdout_scored,
        target=target,
        prediction="prediction",
        prediction_score="prediction_score",
        model=best_model,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        time_column=time_column,
        protected_attribute=["region", "segment"],
        include_profile=True,
        report_name="Final Training",
        experiment_name=EXPERIMENT_NAME,
    )

    # Add calibration when probability quality needs a dedicated view.
    calibration_report(
        holdout_scored,
        target=target,
        prediction_score="prediction_score",
        score_labels=list(best_model.classes_),
        report_name="Final Holdout Calibration",
        experiment_name=EXPERIMENT_NAME,
    )

    mlflow.sklearn.log_model(best_model, name="model")
```

Run bias only when protected attributes are available and permitted for the training purpose. The training bundle includes feature importance when `model` and `feature_columns` are provided. Use the lower-level `feature_importance()` function directly when you need a non-default importance method such as SHAP.

## Settings For Repeated Runs

Use settings to make the repeated tuning and training calls less noisy. For example:

```bash
export LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME=churn-training
export LUMOSAI_DATA__DEFAULT_SAMPLE_SIZE=25000
export LUMOSAI_ARTIFACTS__KEEP_LOCAL=false
export LUMOSAI_MODEL__METRIC_THRESHOLDS__F1__VALUE=0.9
```

With those defaults in the environment, tuning code can focus on trial parameters, data splits, and report boundaries instead of repeating package-level standards in every call.
