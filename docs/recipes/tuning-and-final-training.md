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

When `experiment_name` is passed and an MLflow run is already active, `lumos-ai` logs into that active run. The nested trial run therefore receives the `performance/...` metrics and result artifact for that trial.

## Add Fold-Level Reports

For cross-validation, keep the trial run as the parent and log each fold as an MLflow metric step. This preserves fold-level visibility without creating a nested MLflow run and a full artifact set for every fold.

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

        for fold_index, (train_index, valid_index) in enumerate(
            cv.split(train_frame, train_frame[target])
        ):
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
                metrics=["f1", "roc_auc", "pr_auc"],
                profile="metrics_only",
                mlflow_step=fold_index,
                feature_columns=feature_columns,
                report_name="Fold Validation",
                experiment_name=EXPERIMENT_NAME,
            )
            fold_scores.append(result.metrics["performance/f1"])

        mean_f1 = float(np.mean(fold_scores))
        mlflow.log_metric("mean_fold_f1", mean_f1)
        mlflow.log_metric("std_fold_f1", float(np.std(fold_scores)))
        return mean_f1
```

`profile="metrics_only"` defaults to scalar metrics only: no plots, no lift metrics, and no per-result JSON artifact. `mlflow_step=fold_index` keeps metric names stable while MLflow records the fold on the metric history. Explicit options still override the profile—for example, pass `plots=[...]`, `include_lift=True`, or `log_dict=True` when a particular fold needs richer diagnostics. Create nested fold runs only when separate child runs or fold-specific artifacts are intentionally required. Final training should use the standard profile so the selected model retains the richer diagnostic report.

## Record Training and Validation Loss by Iteration

Lumos reports evaluate predictions after fitting; enabling Lumos MLflow logging
does not instrument `model.fit()`. Keep training-curve logging in the training
pipeline and use the same active MLflow run for the later Lumos reports.

### CatBoost: Log the Recorded Evaluation History

CatBoost's MLflow flavor supports model logging, but does not provide autologging.
Use `get_evals_result()` after fitting to send the recorded train and validation
metrics to MLflow. This works for CPU and GPU training because it does not depend
on a training callback.

Install the additional training dependency:

```bash
pip install "lumosai[mlflow]" catboost
```

This example assumes `train_frame`, `validation_frame`, `feature_columns`, and
`target` are prepared as in the tuning examples above. Validation is used for
early stopping; keep the final holdout separate.

```python
import mlflow
from catboost import CatBoostClassifier

from lumosai.model import performance_report

EXPERIMENT_NAME = "churn-training"
# For a remote server, configure MLFLOW_TRACKING_URI and authentication before
# starting the run. Lumos settings alone do not configure these direct calls.
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="catboost-training"):
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        metric_period=1,  # Required here: each history entry is one iteration.
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    mlflow.log_params(model.get_params())
    model.fit(
        train_frame[feature_columns],
        train_frame[target],
        eval_set=(
            validation_frame[feature_columns],
            validation_frame[target],
        ),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    history = model.get_evals_result()
    for dataset_name, metrics in history.items():
        split = {"learn": "train", "validation": "val"}.get(
            dataset_name, dataset_name
        )
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values):
                mlflow.log_metric(
                    f"training/{split}/{metric_name}",
                    float(value),
                    step=iteration,
                )

    mlflow.log_dict(history, "training/evaluation_history.json")
    mlflow.log_metric("training/tree_count", model.tree_count_)
    best_iteration = model.get_best_iteration()
    if best_iteration is not None and best_iteration >= 0:
        mlflow.log_metric("training/best_iteration", best_iteration)

    validation_scored = score_validation(model, validation_frame, feature_columns)
    performance_report(
        validation_scored,
        target=target,
        prediction="prediction",
        prediction_score="prediction_score",
        score_labels=list(model.classes_),
        profile="metrics_only",
        experiment_name=EXPERIMENT_NAME,
    )
```

In MLflow, plot `training/train/Logloss` and `training/val/Logloss` against
**step** to compare the curves. Steps and `best_iteration` are zero-based.
With early stopping, the history covers the iterations actually evaluated,
including iterations after the best one; `tree_count_` can be smaller because
`use_best_model=True` retains the best model.

This uploads the curves **after fit returns**, not live during training. A failed
or interrupted fit will not reach this logging block. `metric_period=1` keeps the
step mapping exact, at the cost of computing metrics every iteration. Do not
enumerate a history collected with a larger metric period as though each entry
were a consecutive training iteration. Only metrics CatBoost actually computes
appear in the history; for example, training AUC is skipped by default.

For a fitted sklearn pipeline, read history from the final CatBoost estimator
(for example, `pipeline.named_steps["model"].get_evals_result()`). When fitting,
the validation data passed to CatBoost must receive the same fitted preprocessing
as the training data; simply forwarding a raw `model__eval_set` through a
standard pipeline does not transform it automatically.

For Optuna, put the fitting and history logging block inside the trial's active
nested run. For cross-validation, give each fold its own nested run, or include
the fold in the metric name, such as `training/fold_0/val/Logloss`. Do not log
multiple fold curves under the same key and iteration in one run.
`performance_report(mlflow_step=fold_index)` records fold-level evaluation,
not per-iteration training loss.

### XGBoost and LightGBM: Enable Framework Autologging

For supported framework versions, enable the specific integration **before**
fitting and pass validation data to the framework's training API:

```python
# Choose the integration matching your estimator:
import mlflow.xgboost
mlflow.xgboost.autolog(log_models=False)

# Or, for LightGBM instead:
# import mlflow.lightgbm
# mlflow.lightgbm.autolog(log_models=False)

with mlflow.start_run(run_name="boosting-training"):
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
    # Call Lumos reports here with experiment_name=EXPERIMENT_NAME.
```

Configure the framework's evaluation metric explicitly and inspect the
framework-generated metric keys in MLflow. The precise fit arguments, early
stopping options, and compatibility constraints depend on the framework version.
`log_models=False` leaves model serialization to the existing final-model step.
Generic sklearn autologging is not a guarantee of per-iteration loss curves for
every sklearn-compatible estimator. The RandomForest examples above do not have
a boosting-style train/validation loss history.

References:
[CatBoost evaluation history](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_evals_result),
[MLflow CatBoost flavor](https://mlflow.org/docs/latest/api_reference/_modules/mlflow/catboost.html),
[MLflow framework autologging](https://mlflow.org/docs/latest/ml/tracking/autolog/).

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

Run bias only when protected attributes are available and permitted for the training purpose. The training bundle includes feature importance when `model` and `feature_columns` are provided. The default importance method is controlled by `settings.model.feature_importance_method`; set it to `permutation` for base-install training jobs, or install `lumosai[importance]` when using the default `both` method with SHAP.

## Settings For Repeated Runs

Use settings to make the repeated tuning and training calls less noisy. For example:

```bash
export LUMOSAI_MLFLOW__DEFAULT_EXPERIMENT_NAME=churn-training
export LUMOSAI_DATA__DEFAULT_SAMPLE_SIZE=25000
export LUMOSAI_ARTIFACTS__KEEP_LOCAL=false
export LUMOSAI_MODEL__METRIC_THRESHOLDS__F1__VALUE=0.9
```

With those defaults in the environment, tuning code can focus on trial parameters, data splits, and report boundaries instead of repeating package-level standards in every call.
