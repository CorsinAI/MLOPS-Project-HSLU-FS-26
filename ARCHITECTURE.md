# Job Posting Demand Forecasting

## Architecture

```
Raw Data (JSONL)
      |
      v
[Feature Pipeline] --> [Hopsworks Feature Store]
                                |
                                v
                      [Training Pipeline] --> [MLflow Registry]
                                                     |
                                                     v
                                           [Inference Service]
                                           FastAPI / Uvicorn
                                           /forecasts  /drift

[Airflow Scheduler]
  - Feature Pipeline: Friday 06:00
  - Training Pipeline: Friday 07:00 (drift-triggered)
```

## Description

A demand forecasting system for the Swiss job market. Raw job postings are
aggregated into 7-day windows and stored in a Hopsworks feature store. A
LightGBM model is trained on lag and rolling features and tracked via MLflow.
The best model is automatically promoted to production. Retraining triggers
when feature drift exceeds 1.5 standard deviations. Forecasts are served via
a FastAPI REST API. The full stack runs in Docker.
