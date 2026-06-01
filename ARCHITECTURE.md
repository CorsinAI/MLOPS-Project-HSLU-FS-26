# Architecture — Job Posting Demand Forecasting

## Overview

The system follows the FTI (Feature / Training / Inference) pipeline pattern. Each pipeline is independently runnable, communicates through the Hopsworks feature store and the MLflow model registry, and has no direct runtime dependency on the others.

---

## Data Flow

```
Azure Blob Storage
  structured_jobs_normalized_cleaned.jsonl
        │
        │  load_postings()
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  Feature Pipeline  (pipelines/feature/)             │
  │                                                     │
  │  1. Parse JSONL → (job_title, location, date)       │
  │  2. Drop postings before 2026-01-04                 │
  │  3. Bin into 7-day windows                          │
  │  4. Count postings per (title, location, window)    │
  │  5. Compute lag features per time-series:           │
  │       previous_count, rolling_avg_3,                │
  │       rolling_avg_5, growth_rate                    │
  └─────────────┬───────────────────────────────────────┘
                │  write_feature_group()
                ▼
        Hopsworks Feature Store
          feature group: job_postings_features
                │
        ┌───────┴───────────────────────────────────────┐
        │                                               │
        │  read_feature_group()                         │  read_feature_group()
        ▼                                               ▼
  ┌───────────────────────────────┐      ┌───────────────────────────────────┐
  │  Training Pipeline            │      │  Inference Pipeline               │
  │  (pipelines/training/)        │      │  (pipelines/inference/)           │
  │                               │      │                                   │
  │  1. trim_for_training()       │      │  1. build_inference_features()    │
  │     hold out last 2 complete  │      │     select latest complete window │
  │     windows                   │      │     per (title, location) pair    │
  │  2. build_dataset()           │      │                                   │
  │     add target (next count),  │      │  2. booster.predict()             │
  │     drop NaN lag rows         │      │     → forecast next window        │
  │  3. time_split() 60/20/20     │      │                                   │
  │  4. LightGBM train            │      │  3. detect_drift()                │
  │  5. Log to MLflow             │      │     compare inference batch vs    │
  │  6. Promote champion          │      │     training distribution         │
  └───────────┬───────────────────┘      └────────────────┬──────────────────┘
              │  register model                           │  serve via FastAPI
              ▼                                           ▼
     DagsHub MLflow Registry               HuggingFace Spaces
       experiment: job_posting_forecast       GET /drift
       alias: champion                        GET /dashboard
                                              GET /forecasts
                                              GET /health
```

---

## Components

### Feature Pipeline (`pipelines/feature/`)

| File | Responsibility |
|---|---|
| `aggregate.py` | JSONL parsing, window assignment, posting count aggregation |
| `features.py` | Lag and rolling feature computation per time-series |
| `hopsworks_writer.py` | Writes the feature group to Hopsworks |
| `hopsworks_reader.py` | Reads the feature group from Hopsworks |
| `run.py` | Entry point: Azure → features → Hopsworks |

**Window strategy**: all dates are binned into 7-day buckets anchored at `2026-01-04`. Postings before that date are excluded due to sparse historical data.

### Training Pipeline (`pipelines/training/`)

| File | Responsibility |
|---|---|
| `prepare.py` | `trim_for_training`, `build_dataset`, `encode_categoricals`, `time_split`, `get_X_y` |
| `train.py` | LightGBM training, MLflow logging, champion promotion |
| `run.py` | Unconditional training entry point |
| `retrain_if_drift.py` | Conditional entry point: retrains if drift > 1.5 σ or model > 21 days old |

**Holdout strategy**: the 2 most recent complete windows (where `window_start + 7 days ≤ today`) are withheld from training. This means:
- The model trains on older data only.
- The inference batch (latest complete window) is genuinely unseen by the model.
- Drift detection produces a meaningful signal.

**Split**: 60% train / 20% val / 20% test, always by ascending `window_start` — no shuffling across time.

**Champion promotion**: after each run, if the new model's validation RMSE is lower than the current champion's, the `champion` alias is moved to the new version.

### Inference Pipeline (`pipelines/inference/`)

| File | Responsibility |
|---|---|
| `prepare.py` | Selects the latest complete window per (title, location); drops NaN-lag rows; encodes categoricals |
| `load_model.py` | Loads the `champion` model from MLflow (falls back to latest version) |
| `drift.py` | Computes z-score drift: `|current_mean − ref_mean| / ref_std`; flags features above threshold 1.5 |
| `app.py` | FastAPI endpoints |
| `landing.py` | Landing page HTML (Drift + Dashboard tabs) |
| `dashboard.py` | Interactive HTML dashboard with charts and forecast table |
| `run.py` | Uvicorn entry point |

**Incomplete window filtering**: a window is excluded from inference if `window_start + 7 days > today`, because a window still in progress has an artificially low count.

**Drift reference**: `compute_reference_stats` is called on `trim_for_training(features)`, not on all features, so the reference distribution matches what the model was trained on.

---

## Automation

### GitHub Actions

**`weekly_pipeline.yml`** — runs every Friday at 15:00 UTC (and on manual dispatch):
1. Runs the feature pipeline against Azure Blob Storage and Hopsworks.
2. Runs `retrain_if_drift.py`: retrains if any feature's z-score exceeds 1.5 or the last training run was more than 21 days ago.

**`deploy_hf.yml`** — runs on every push to `main`:
- Force-pushes the repository to HuggingFace Spaces, triggering a Docker rebuild of the inference service.

---

## External Services

| Service | Role |
|---|---|
| Azure Blob Storage | Stores the raw JSONL job postings file |
| Hopsworks (eu-west) | Feature store — persists and versions the feature group |
| DagsHub MLflow | Experiment tracking, model registry, champion alias |
| HuggingFace Spaces | Docker-based hosting for the inference service |
| GitHub Actions | Orchestration of weekly pipelines and continuous deployment |

---

## Feature Schema

| Column | Type | Description |
|---|---|---|
| `job_title` | string | Normalised job title |
| `location` | string | Swiss canton or city |
| `window_start` | datetime | Start of the 7-day posting window |
| `count` | int | Number of postings in this window |
| `previous_count` | float | Count from the prior window (lag 1) |
| `rolling_avg_3` | float | Rolling mean of the previous 3 windows |
| `rolling_avg_5` | float | Rolling mean of the previous 5 windows |
| `growth_rate` | float | `(count − previous_count) / (previous_count + 1)` |

`previous_count`, `rolling_avg_3`, `rolling_avg_5`, and `growth_rate` are NaN for the first observed window of each series (no prior data).
