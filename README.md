---
title: MLOPS Job Forecasting
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Job Posting Demand Forecasting — MLOps Pipeline

A production-style MLOps project that forecasts job posting demand across Swiss job titles and cantons. Built on the FTI (Feature, Training, Inference) pipeline architecture with a Hopsworks feature store, DagsHub-hosted MLflow experiment tracking, and a FastAPI serving layer hosted on HuggingFace Spaces.

---

## Architecture

```
Azure Blob Storage (JSONL)
      |
      v
Feature Pipeline  -->  Hopsworks Feature Store
                              |
                              v
                    Training Pipeline  -->  DagsHub MLflow Registry
                                                  |
                                                  v
                                        Inference Service (FastAPI)
                                        HuggingFace Spaces
                                                  |
                                          /forecasts  /drift  /dashboard

GitHub Actions (every Friday)
  - Feature Pipeline: update feature store
  - Training Pipeline: retrain if drift detected
```

The stack is split into three independent pipelines:

- **Feature Pipeline** — loads raw job postings from Azure Blob Storage, assigns 7-day windows, computes lag and rolling features, and writes to the Hopsworks feature store.
- **Training Pipeline** — reads features from Hopsworks, performs a time-based train/val/test split, trains a LightGBM model, and logs all metrics and artifacts to MLflow on DagsHub.
- **Inference Pipeline** — reads the latest complete feature window from Hopsworks, loads the registered model from MLflow, generates forecasts, and serves them via a REST API with drift monitoring.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A [Hopsworks](https://www.hopsworks.ai/) account with a project and API key

---

## Setup

**1. Clone the repository**
```bash
git clone <repo-url>
cd MLOPS-Project-HSLU-FS-26
```

**2. Configure environment variables**
```bash
cp .env.example .env
```

The `.env.example` contains working read-only credentials for Azure Blob Storage and DagsHub MLflow. Fill in your own Hopsworks credentials:
```
HOPSWORKS_HOST=your-hopsworks-host.cloud.hopsworks.ai
HOPSWORKS_API_KEY=your-api-key-here
HOPSWORKS_PROJECT=your-project-name
```

**3. Start Docker Desktop**, then bring up the inference service:
```bash
docker compose up --build
```

**4. Run the training pipeline** (first time only — registers the model with MLflow):
```bash
docker compose --profile pipeline run training
```

**5. Restart the stack** so the inference service picks up the registered model:
```bash
docker compose up
```

---

## Services

| Service | URL | Description |
|---|---|---|
| Inference API | http://localhost:8000 | FastAPI forecast service |
| API Docs | http://localhost:8000/docs | Auto-generated Swagger UI |
| Dashboard | http://localhost:8000/dashboard | Interactive forecast + drift dashboard |
| MLflow UI | https://dagshub.com/CorsinAI/MLOPS-Project-HSLU-FS-26.mlflow | Experiment tracking and model registry (DagsHub) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/forecasts` | All forecasts (filterable by `role` and `location`) |
| GET | `/forecasts/{job_title}` | Forecasts for a specific job title |
| GET | `/drift` | Feature drift report |
| POST | `/refresh` | Reload data and regenerate forecasts |
| GET | `/dashboard` | Interactive HTML dashboard |

---

## Orchestration

Pipelines run automatically every Friday via GitHub Actions:

1. **Feature pipeline** — fetches latest data from Azure Blob Storage and updates the Hopsworks feature store.
2. **Training pipeline** — checks for feature drift; retrains and promotes a new champion model if drift exceeds 1.5 standard deviations.

To trigger manually: GitHub → Actions → Weekly ML Pipeline → Run workflow.

---

## Running Pipelines Manually

**Feature pipeline** — ingest new data and update the feature store:
```bash
docker compose --profile pipeline run feature
```

**Training pipeline** — retrain the model on the latest features:
```bash
docker compose --profile pipeline run training
```

---

## Running Tests

```bash
pytest tests/
```

---

## Project Structure

```
.
├── pipelines/
│   ├── feature/          # Feature engineering and Hopsworks integration
│   ├── training/         # LightGBM training and MLflow logging
│   └── inference/        # FastAPI serving, drift detection, dashboard
├── data_preprocessing/   # Raw data cleanup scripts (one-time use)
├── tests/                # Unit tests for all three pipelines
├── .github/workflows/    # GitHub Actions weekly pipeline
├── Dockerfile            # Single image for all services
├── docker-compose.yml    # Local stack: inference + pipeline jobs
├── requirements.txt      # Pinned dependencies
└── .env.example          # Environment variable template with public credentials
```

---

## Model Tracking

All training runs are logged to MLflow on DagsHub including:
- Metrics: RMSE, MAE, R² on validation and test sets
- Parameters: all LightGBM hyperparameters, dataset split dates, window config
- Artifacts: serialized model, environment specs

The inference service loads the model tagged as `champion` from the MLflow registry, falling back to the highest registered version if no alias is set.

---

## Monitoring

The inference service computes feature drift on every startup using z-scores against the training distribution. Features with a z-score above 1.5 are flagged. The drift report is available at `/drift` and is also shown in the dashboard.

Drift in count-based features at the end of a 7-day window is expected due to partially completed windows. The pipeline automatically excludes incomplete windows from inference.
