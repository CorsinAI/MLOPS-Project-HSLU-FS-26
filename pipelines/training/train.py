"""
Trains a LightGBM regression model and logs everything to MLflow.
"""
import os

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipelines.training.prepare import CATEGORICALS, FEATURES, TARGET

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def _eval_metrics(booster: lgb.Booster, X: pd.DataFrame, y: pd.Series, prefix: str) -> dict[str, float]:
    """Compute RMSE, MAE and R² for a given split and return them as prefixed keys."""
    preds = booster.predict(X)
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, preds))),
        f"{prefix}_mae": float(mean_absolute_error(y, preds)),
        f"{prefix}_r2": float(r2_score(y, preds)),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
) -> tuple[lgb.Booster, dict]:
    """
    Train LightGBM using val for early stopping, evaluate on test once at the end.
    Logs everything to MLflow and registers the model.
    Returns the booster and a dict of all metrics.
    """
    if params is None:
        params = DEFAULT_PARAMS

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICALS, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICALS, reference=dtrain, free_raw_data=False)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("job_posting_forecast")

    with mlflow.start_run() as run:
        mlflow.log_params({**params, "num_boost_round": num_boost_round})
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("val_start", str(val_df["window_start"].min().date()))
        mlflow.log_param("test_start", str(test_df["window_start"].min().date()))

        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
        ]

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        metrics = {
            "best_iteration": booster.best_iteration,
            **_eval_metrics(booster, X_val, y_val, "val"),
            **_eval_metrics(booster, X_test, y_test, "test"),
        }

        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "best_iteration"})
        mlflow.log_metric("best_iteration", metrics["best_iteration"])
        mlflow.lightgbm.log_model(booster, artifact_path="model", registered_model_name="job_posting_forecast")

        print(f"  {'':20s}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}")
        print(f"  {'Val':20s}  {metrics['val_rmse']:7.3f}  {metrics['val_mae']:7.3f}  {metrics['val_r2']:7.3f}")
        print(f"  {'Test':20s}  {metrics['test_rmse']:7.3f}  {metrics['test_mae']:7.3f}  {metrics['test_r2']:7.3f}")
        print(f"  Best iteration: {metrics['best_iteration']}")
        run_id = run.info.run_id

    # Promote to champion if this model beats the current champion's val RMSE
    client = MlflowClient(mlflow_uri)
    versions = client.search_model_versions("name='job_posting_forecast'")
    new_version = next(v.version for v in versions if v.run_id == run_id)

    try:
        champion_mv = client.get_model_version_by_alias("job_posting_forecast", "champion")
        champion_rmse = client.get_run(champion_mv.run_id).data.metrics.get("val_rmse", float("inf"))
    except Exception:
        champion_rmse = float("inf")

    if metrics["val_rmse"] < champion_rmse:
        client.set_registered_model_alias("job_posting_forecast", "champion", new_version)
        print(f"  Promoted version {new_version} to champion (val_rmse {metrics['val_rmse']:.3f})")
    else:
        print(f"  Version {new_version} not promoted — champion val_rmse {champion_rmse:.3f} is still better")

    return booster, metrics
