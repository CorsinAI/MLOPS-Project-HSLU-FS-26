"""
Loads the latest registered LightGBM model from the MLflow model registry.
"""
import os

import lightgbm as lgb
import mlflow
import mlflow.lightgbm


def load_model(model_name: str = "job_posting_forecast") -> tuple[lgb.Booster, str]:
    """
    Return (booster, version_string) for the latest registered model version.

    Raises RuntimeError if no versions are registered.
    """
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(mlflow_uri)

    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions found for model '{model_name}'. "
            "Run the training pipeline first."
        )
    mv = max(versions, key=lambda v: int(v.version))
    version = mv.version

    # MLflow 3.x stores artifacts under a per-model-ID path exposed via
    # `storage_location`.  The legacy `models:/<name>/<version>` URI resolves
    # through the run's artifact directory, which is empty in this layout.
    # Prefer `storage_location` when available; fall back to the model-ID URI
    # (`mv.source`) which also points to the new location, and finally to the
    # classic versioned URI.
    if getattr(mv, "storage_location", None):
        model_uri = mv.storage_location
    elif getattr(mv, "source", None) and mv.source.startswith("models:/m-"):
        model_uri = mv.source
    else:
        model_uri = f"models:/{model_name}/{version}"

    booster: lgb.Booster = mlflow.lightgbm.load_model(model_uri)

    print(f"Loaded model '{model_name}' version {version} from {model_uri}")
    return booster, version
