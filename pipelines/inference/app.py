"""
FastAPI inference service for job-posting demand forecasts.

On startup the app:
  1. Loads features from Hopsworks (or falls back to local JSONL).
  2. Loads the latest registered LightGBM model from MLflow.
  3. Generates 3-day-ahead forecasts for every (job_title, location) pair.
  4. Computes a drift report against the full historical feature distribution.

Environment variables:
  USE_HOPSWORKS         – set to "true" to read features from Hopsworks (default: false)
  INFERENCE_DATA_PATH   – path to the JSONL postings file (fallback when USE_HOPSWORKS is false)
  MLFLOW_TRACKING_URI   – MLflow backend (default: mlruns)

Endpoints:
  GET /health                     – liveness / readiness probe
  GET /forecasts                  – all forecasts; optional ?role= and ?location= filters
  GET /forecasts/{job_title}      – forecasts for a specific role (URL-encoded)
  GET /drift                      – feature drift report
  POST /refresh                   – reload data + model and regenerate forecasts
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.inference.drift import compute_reference_stats, detect_drift
from pipelines.inference.load_model import load_model
from pipelines.inference.prepare import build_inference_features
from pipelines.training.prepare import FEATURES

_USE_HOPSWORKS = os.environ.get("USE_HOPSWORKS", "false").lower() == "true"
_DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "INFERENCE_DATA_PATH",
        "data/structured_jobs_20.05_normalized_cleaned.jsonl",
    )
)


def _run_pipeline() -> dict:
    """
    Full inference pipeline: features → model → forecasts + drift.

    Returns a state dict stored in app.state.
    """
    # --- Feature data ---
    if _USE_HOPSWORKS:
        from pipelines.feature.hopsworks_reader import read_feature_group
        features = read_feature_group()
    else:
        from pipelines.feature.aggregate import load_postings, assign_windows, aggregate_counts
        from pipelines.feature.features import compute_features
        df = load_postings(_DEFAULT_DATA_PATH)
        df = assign_windows(df)
        counts = aggregate_counts(df)
        features = compute_features(counts)

    # --- Load model ---
    booster, model_version = load_model()

    # --- Inference features: latest window per (job_title, location) ---
    inference_df = build_inference_features(features)

    raw_preds = booster.predict(inference_df[FEATURES])
    preds = np.clip(raw_preds, 0.0, None)  # counts cannot be negative

    # Forecast window: last observed window + WINDOW_DAYS
    last_window = inference_df["window_start"].max()
    forecast_start = last_window + timedelta(days=WINDOW_DAYS)
    forecast_end = forecast_start + timedelta(days=WINDOW_DAYS - 1)

    forecasts = [
        {
            "job_title": str(row["job_title"]),
            "location": str(row["location"]),
            "last_observed_window": str(row["window_start"].date()),
            "predicted_count": round(float(preds[idx]), 2),
        }
        for idx, (_, row) in enumerate(inference_df.iterrows())
    ]
    forecasts.sort(key=lambda x: x["predicted_count"], reverse=True)

    # --- Drift: current batch vs full historical distribution ---
    reference_stats = compute_reference_stats(features)
    drift_report = detect_drift(inference_df, reference_stats)

    return {
        "model_version": model_version,
        "data_source": "hopsworks" if _USE_HOPSWORKS else str(_DEFAULT_DATA_PATH),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "forecast_window_start": str(forecast_start.date()),
        "forecast_window_end": str(forecast_end.date()),
        "num_pairs": len(forecasts),
        "forecasts": forecasts,
        "drift_report": drift_report,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _run_pipeline()
    yield


app = FastAPI(
    title="JobAnalysis Inference API",
    description="3-day-ahead job-posting demand forecasts for the Swiss job market.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    """Landing page with links to all endpoints."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Job Posting Forecast API</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 60px auto; padding: 0 24px; color: #1a1a1a; }
    h1 { font-size: 1.8rem; margin-bottom: 4px; }
    .subtitle { color: #555; margin-bottom: 40px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px 24px; text-decoration: none; color: inherit; display: block; transition: box-shadow 0.15s; }
    .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
    .card .method { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; padding: 2px 8px; border-radius: 4px; display: inline-block; margin-bottom: 8px; }
    .get { background: #e8f5e9; color: #2e7d32; }
    .post { background: #fff3e0; color: #e65100; }
    .card h2 { font-size: 1rem; margin: 0 0 6px; }
    .card p { font-size: 0.875rem; color: #555; margin: 0; }
    .docs-link { margin-top: 32px; font-size: 0.875rem; color: #555; }
    .docs-link a { color: #1976d2; }
  </style>
</head>
<body>
  <h1>Job Posting Forecast API</h1>
  <p class="subtitle">3-day-ahead demand forecasts for the Swiss job market — powered by LightGBM, Hopsworks, and MLflow.</p>

  <div class="grid">
    <a class="card" href="/dashboard">
      <span class="method get">GET</span>
      <h2>/dashboard</h2>
      <p>Interactive dashboard with charts, forecast table, and drift report.</p>
    </a>
    <a class="card" href="/forecasts">
      <span class="method get">GET</span>
      <h2>/forecasts</h2>
      <p>All forecasts sorted by predicted count. Filter by <code>?role=</code> and <code>?location=</code>.</p>
    </a>
    <a class="card" href="/drift">
      <span class="method get">GET</span>
      <h2>/drift</h2>
      <p>Feature drift report comparing the current batch against the training distribution.</p>
    </a>
    <a class="card" href="/health">
      <span class="method get">GET</span>
      <h2>/health</h2>
      <p>Liveness probe. Returns model version and generation timestamp.</p>
    </a>
    <a class="card" href="/docs">
      <span class="method get">GET</span>
      <h2>/docs</h2>
      <p>Auto-generated Swagger UI with full API documentation.</p>
    </a>
    <a class="card" href="#" onclick="fetch('/refresh',{method:'POST'}).then(r=>r.json()).then(d=>alert('Refreshed at '+d.generated_at));return false;">
      <span class="method post">POST</span>
      <h2>/refresh</h2>
      <p>Reload data and regenerate all forecasts. Click to trigger.</p>
    </a>
  </div>

  <p class="docs-link">Full API docs at <a href="/docs">/docs</a> &middot; Redoc at <a href="/redoc">/redoc</a></p>
</body>
</html>
"""


@app.get("/health", summary="Liveness / readiness probe")
def health():
    state = app.state.pipeline
    return {
        "status": "ok",
        "model_version": state["model_version"],
        "generated_at": state["generated_at"],
        "num_pairs": state["num_pairs"],
    }


@app.get("/forecasts", summary="All 3-day-ahead forecasts")
def get_forecasts(
    role: Optional[str] = Query(None, description="Filter by job_title (case-insensitive substring)"),
    location: Optional[str] = Query(None, description="Filter by location (case-insensitive substring)"),
):
    """
    Returns all forecast rows, sorted by predicted_count descending.
    Optionally filtered by role and/or location substring.
    """
    state = app.state.pipeline
    results = state["forecasts"]

    if role:
        role_lower = role.lower()
        results = [f for f in results if role_lower in f["job_title"].lower()]
    if location:
        loc_lower = location.lower()
        results = [f for f in results if loc_lower in f["location"].lower()]

    return {
        "generated_at": state["generated_at"],
        "forecast_window_start": state["forecast_window_start"],
        "forecast_window_end": state["forecast_window_end"],
        "model_version": state["model_version"],
        "count": len(results),
        "forecasts": results,
    }


@app.get("/forecasts/{job_title}", summary="Forecasts for a specific role")
def get_forecasts_by_role(job_title: str):
    """
    Returns all location forecasts for the given job_title (exact match, case-insensitive).
    Raises 404 if the role is not recognised.
    """
    state = app.state.pipeline
    results = [
        f for f in state["forecasts"]
        if f["job_title"].lower() == job_title.lower()
    ]
    if not results:
        raise HTTPException(status_code=404, detail=f"Role '{job_title}' not found in forecasts.")

    return {
        "generated_at": state["generated_at"],
        "forecast_window_start": state["forecast_window_start"],
        "forecast_window_end": state["forecast_window_end"],
        "model_version": state["model_version"],
        "job_title": job_title,
        "count": len(results),
        "forecasts": results,
    }


@app.get("/drift", summary="Feature drift report")
def get_drift():
    """
    Returns the drift report comparing the most recent inference batch against
    the full historical feature distribution.
    """
    state = app.state.pipeline
    return {
        "generated_at": state["generated_at"],
        "model_version": state["model_version"],
        **state["drift_report"],
    }


@app.get("/dashboard", response_class=HTMLResponse, summary="Browser dashboard")
def dashboard():
    """Interactive HTML dashboard with charts and a filterable forecast table."""
    from pipelines.inference.dashboard import render
    return render(app.state.pipeline)


@app.post("/refresh", summary="Reload data and regenerate forecasts")
def refresh():
    """
    Reruns the full pipeline (feature engineering → model load → forecasts).
    Use this after a new data batch has been written to the data directory.
    """
    app.state.pipeline = _run_pipeline()
    state = app.state.pipeline
    return {
        "status": "refreshed",
        "generated_at": state["generated_at"],
        "model_version": state["model_version"],
        "num_pairs": state["num_pairs"],
    }
