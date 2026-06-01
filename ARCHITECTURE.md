# Job Posting Demand Forecasting

## Architecture

```
Azure Blob Storage (JSONL)
      |
      v
[Feature Pipeline] --> [Hopsworks Feature Store]
                                |
                                v
                      [Training Pipeline] --> [DagsHub MLflow Registry]
                                                     |
                                                     v
                                           [Inference Service]
                                           FastAPI / HuggingFace Spaces
                                           /forecasts  /drift  /dashboard

[GitHub Actions — every Friday]
  - Feature Pipeline: fetch latest data, update feature store
  - Training Pipeline: retrain if drift > 1.5 std devs
```

## Description

A demand forecasting system for the Swiss job market. Raw job postings are
fetched from Azure Blob Storage, aggregated into 7-day windows, and stored
in a Hopsworks feature store. A LightGBM model is trained on lag and rolling
features and tracked via MLflow on DagsHub. The best model is automatically
promoted to the champion alias. Retraining triggers when feature drift exceeds
1.5 standard deviations. Forecasts are served via a FastAPI REST API hosted
on HuggingFace Spaces. Pipelines are orchestrated weekly by GitHub Actions.
