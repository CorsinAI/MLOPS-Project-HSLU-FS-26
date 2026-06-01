"""
Drift-gated retraining entry point.

Retrains the model if either:
  - Feature drift exceeds the threshold, or
  - No training run has occurred in the last 3 weeks.

Usage:
    python -m pipelines.training.retrain_if_drift
"""
import os
from datetime import timedelta

import mlflow
import pandas as pd

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.feature.hopsworks_reader import read_feature_group
from pipelines.inference.drift import DRIFT_THRESHOLD, compute_reference_stats, detect_drift
from pipelines.inference.prepare import build_inference_features

MAX_DAYS_WITHOUT_RETRAINING = 21


def _weeks_since_last_run() -> float | None:
    """Return the number of days since the last MLflow training run, or None if no runs exist."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    runs = mlflow.search_runs(
        experiment_names=["job_posting_forecast"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    last_run_time = pd.Timestamp(runs.iloc[0]["start_time"])
    return (pd.Timestamp.now(tz="UTC") - last_run_time).days


def main() -> None:
    features = read_feature_group()
    today = pd.Timestamp.now().normalize()

    complete_features = features[
        features["window_start"] + pd.Timedelta(days=WINDOW_DAYS) <= today
    ]

    reference_stats = compute_reference_stats(complete_features)
    current = build_inference_features(complete_features)
    report = detect_drift(current, reference_stats)

    print(f"Drift detected: {report['drift_detected']} (threshold: {DRIFT_THRESHOLD})")
    for feature, result in report["features"].items():
        print(f"  {feature}: z={result['z_score']:.3f} {'DRIFTED' if result['drifted'] else 'ok'}")

    days_since = _weeks_since_last_run()
    if days_since is None:
        print("No previous training runs found.")
    else:
        print(f"Days since last training run: {days_since}")

    stale = days_since is None or days_since >= MAX_DAYS_WITHOUT_RETRAINING

    if report["drift_detected"]:
        print("Drift threshold exceeded — retraining model.")
    elif stale:
        print(f"No retraining in {days_since if days_since is not None else 'ever'} days — retraining model.")
    else:
        print("No significant drift and model is recent — skipping retraining.")

    if report["drift_detected"] or stale:
        from pipelines.training.run import run
        run(from_hopsworks=True)


if __name__ == "__main__":
    main()
