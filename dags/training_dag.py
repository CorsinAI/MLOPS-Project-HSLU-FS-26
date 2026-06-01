"""
Training pipeline DAG.

Runs Friday at 07:00 (after the feature pipeline).
Delegates drift detection and retraining to pipelines.training.retrain_if_drift.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def check_drift_and_retrain() -> None:
    from pipelines.training.retrain_if_drift import main
    main()


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
