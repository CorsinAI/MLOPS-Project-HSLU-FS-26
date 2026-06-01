# Architecture — Job Posting Demand Forecasting

## Overview

Raw job postings sit in Azure Blob Storage. The feature pipeline pulls them, bins them into weekly windows, and computes lag features, then writes everything to Hopsworks.
From there, two pipelines branch off. The training pipeline builds a LightGBM model on historical windows and registers the best one to MLflow. The inference pipeline takes the latest complete window, runs a forecast, checks for drift, and serves results via FastAPI on HuggingFace Spaces.
A GitHub Action runs the whole thing every Friday automatically.

<img width="1760" height="1727" alt="image" src="https://github.com/user-attachments/assets/58d7838c-f336-4afe-aadf-3470fae2108f" />
