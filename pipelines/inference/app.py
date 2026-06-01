"""
FastAPI inference service for job-posting demand forecasts.

On startup the app:
  1. Loads features from the Hopsworks feature store.
  2. Loads the latest registered LightGBM model from MLflow.
  3. Generates 3-day-ahead forecasts for every (job_title, location) pair.
  4. Computes a drift report against the full historical feature distribution.

Environment variables:
  HOPSWORKS_HOST        – Hopsworks instance host
  HOPSWORKS_API_KEY     – Hopsworks API key
  HOPSWORKS_PROJECT     – Hopsworks project name
  MLFLOW_TRACKING_URI   – MLflow backend (DagsHub)

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
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.inference.drift import compute_reference_stats, detect_drift
from pipelines.inference.load_model import load_model
from pipelines.inference.prepare import build_inference_features
from pipelines.training.prepare import FEATURES


def _run_pipeline() -> dict:
    """
    Full inference pipeline: features → model → forecasts + drift.

    Returns a state dict stored in app.state.
    """
    # --- Feature data ---
    from pipelines.feature.hopsworks_reader import read_feature_group
    features = read_feature_group()

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
        "data_source": "hopsworks",
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
    """Landing page with interactive UI for all endpoints."""
    from pipelines.inference.landing import render
    return render()


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
