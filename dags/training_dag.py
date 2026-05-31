"""
Training pipeline DAG.

Runs Monday, Wednesday, and Friday at 07:00 (after the feature pipeline).
Retrains the model when feature drift exceeds the configured threshold —
i.e. when the current feature distribution has shifted far enough from the
training distribution that a fresh model is warranted.
"""
import os
from datetime import datetime

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator


def check_drift_and_retrain() -> None:
    from pipelines.feature.aggregate import WINDOW_DAYS
    from pipelines.feature.hopsworks_reader import read_feature_group
    from pipelines.inference.drift import compute_reference_stats, detect_drift, DRIFT_THRESHOLD
    from pipelines.inference.prepare import build_inference_features

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

    if report["drift_detected"]:
        print("Drift threshold exceeded — retraining model.")
        from pipelines.training.run import run
        run(from_hopsworks=True)
    else:
        print("No significant drift — skipping retraining.")


with DAG(
    dag_id="training_pipeline",
    description="Retrain model when feature drift exceeds the drift threshold",
    schedule="0 7 * * 5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["training"],
) as dag:
    PythonOperator(
        task_id="check_drift_and_retrain",
        python_callable=check_drift_and_retrain,
    )
