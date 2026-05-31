"""
Feature pipeline DAG.

Runs Monday, Wednesday, and Friday at 06:00 to ingest the latest job postings
and push updated features to the Hopsworks feature store.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_feature_pipeline() -> None:
    from pipelines.feature.run import run
    run(push_to_hopsworks=True)


with DAG(
    dag_id="feature_pipeline",
    description="Ingest new job postings and update Hopsworks feature store",
    schedule="0 6 * * 5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["feature"],
) as dag:
    PythonOperator(
        task_id="run_feature_pipeline",
        python_callable=run_feature_pipeline,
    )
