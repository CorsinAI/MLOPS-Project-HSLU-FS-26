---
title: MLOPS Job Forecasting
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Job Posting Demand Forecasting

See the result @: https://huggingface.co/spaces/CorsinAI/MLOPS-Job-Forecasting 

Job posting demand forecasting for the Swiss job market. An end-to-end ML pipeline that predicts weekly hiring trends by job title and location.


**GitHub Actions automation:**
- Every Friday: Feature Pipeline → conditional retrain
- Every push to `main` or after weekly pipeline success: auto-deploy to HuggingFace Spaces

- **Feature Pipeline**: reads raw job postings from Azure Blob Storage, assigns 7-day windows anchored at 2026-01-04, computes lag and rolling features (`previous_count`, `rolling_avg_3`, `rolling_avg_5`, `growth_rate`), and writes to Hopsworks.
- **Training Pipeline**: reads features from Hopsworks, holds out the 2 most recent complete windows, performs a 60/20/20 time-based split, trains a LightGBM regressor, logs to MLflow, and registers the model as a new version.
- **Inference Pipeline**: reads features from Hopsworks, loads the latest registered model version from MLflow, forecasts the next 7-day window for every (job title, location) pair, and runs z-score drift detection comparing the held-out inference batch against the training distribution.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/CorsinAI/MLOPS-Project-HSLU-FS-26
cd MLOPS-Project-HSLU-FS-26
```

**2. Configure environment variables**
```bash
cp .env.example .env
```

`.env.example` contains working credentials for Azure Blob Storage, DagsHub MLflow, and Hopsworks.

**3. Start Docker Desktop**, then bring up the inference service:
```bash
docker compose up --build
```
Note that commands documented in docstrings might only work in Linux terminals because of dependency issues.
---

## Services

| Service | URL |
|---|---|
| Inference API | http://localhost:8000 |
| Drift + Dashboard UI | http://localhost:8000 |
| MLflow / Model Registry | https://dagshub.com/CorsinAI/MLOPS-Project-HSLU-FS-26.mlflow |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/drift` | Feature drift report (JSON) |
| GET | `/dashboard` | Interactive HTML dashboard |

---

## Orchestration

Two GitHub Actions workflows run automatically:

**Weekly ML Pipeline** (`weekly_pipeline.yml`) — every Friday at 15:00 UTC:
1. Runs the feature pipeline — fetches latest data from Azure, updates Hopsworks.
2. Checks for feature drift and time since last training run; retrains if drift exceeds 1.0 σ or the model is more than 21 days old.

**HuggingFace Deploy** (`deploy_hf.yml`) — on every push to `main` and after every successful Weekly ML Pipeline run:
- Force-pushes the repository to HuggingFace Spaces, triggering a Docker rebuild of the inference service.

To trigger either workflow manually: GitHub → Actions → select workflow → Run workflow.

---

## Running Pipelines Manually

```bash
# Update the feature store from Azure Blob Storage
docker compose --profile pipeline run feature

# Retrain the model (unconditional)
docker compose --profile pipeline run training
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

59 tests covering all three pipelines (feature engineering, training preparation, inference + drift detection).

---

## Project Structure

```
.
├── pipelines/
│   ├── feature/
│   │   ├── aggregate.py          # JSONL parsing, window assignment, count aggregation
│   │   ├── features.py           # lag and rolling feature computation
│   │   ├── hopsworks_writer.py   # writes feature group to Hopsworks
│   │   ├── hopsworks_reader.py   # reads feature group from Hopsworks
│   │   └── run.py                # entry point
│   ├── training/
│   │   ├── prepare.py            # holdout trim, dataset build, time split, get_X_y
│   │   ├── train.py              # LightGBM training, MLflow logging, champion promotion
│   │   ├── retrain_if_drift.py   # drift-gated + staleness-gated retraining entry point
│   │   └── run.py                # unconditional training entry point
│   └── inference/
│       ├── app.py                # FastAPI endpoints
│       ├── landing.py            # landing page (Drift + Dashboard tabs)
│       ├── dashboard.py          # interactive HTML dashboard
│       ├── drift.py              # z-score drift detection
│       ├── prepare.py            # selects latest complete window per pair
│       ├── load_model.py         # loads champion model from MLflow
│       └── run.py                # uvicorn entry point
├── tests/
│   ├── conftest.py               # shared fixtures and make_features_df helper
│   ├── test_feature_pipeline.py  # 16 tests: parsing, windowing, aggregation, features
│   ├── test_training_pipeline.py # 23 tests: holdout, dataset build, split, encoding
│   └── test_inference_pipeline.py # 20 tests: inference prep, reference stats, drift
├── .github/workflows/
│   ├── weekly_pipeline.yml       # Friday: feature pipeline + conditional retrain
│   └── deploy_hf.yml             # push to main or weekly pipeline success → HuggingFace Spaces deploy
├── Dockerfile                    # single image for all services (Python 3.11-slim)
├── docker-compose.yml            # inference service + pipeline job profiles
├── requirements.txt
└── .env.example                  # template with working Azure, DagsHub MLflow, and Hopsworks credentials
```

---

## Model Tracking

Every training run logs to the `job_posting_forecast` experiment on DagsHub MLflow:

- **Metrics**: RMSE, MAE, R² on validation and test sets
- **Parameters**: all LightGBM hyperparameters, split dates, row counts, window config
- **Artifacts**: serialized booster

After training, the new model is registered as a new version in the MLflow model registry. The inference service always loads the latest registered version.

---

## Monitoring

Drift is computed on every startup and on `/refresh`. The inference batch (latest complete window per pair) is compared against the training distribution (all windows except the 2 most recent complete ones) using z-scores. Features with |z| > 1.0 are flagged. The report is served at `/drift` and visualised in the landing page.

The 2-window holdout ensures the inference batch contains data the model has never seen, making the drift signal genuine.
