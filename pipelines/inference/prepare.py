"""
Prepares features for inference.

From the full historical feature DataFrame (all windows), selects the most
recent observed window per (job_title, location).  Those rows are fed to the
model to produce a forecast for the NEXT 7-day window.
"""
import pandas as pd

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.training.prepare import CATEGORICALS


def build_inference_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Select the latest *complete* window per (job_title, location) and encode categoricals.

    A window is complete when its end date (window_start + WINDOW_DAYS) has passed.
    Incomplete windows are excluded because partial scrape data artificially deflates
    counts and distorts lag features.

    Rows where lag features are NaN (first window of a series) are dropped
    because the model cannot produce a meaningful prediction without them.

    Returns a DataFrame with the same columns as the training feature set,
    reset to a 0-based integer index.
    """
    today = pd.Timestamp.now().normalize()
    complete = features[
        features["window_start"] + pd.Timedelta(days=WINDOW_DAYS) <= today
    ].copy()

    latest = (
        complete
        .sort_values("window_start")
        .groupby(["job_title", "location"], as_index=False)
        .last()
    )

    latest = latest.dropna(subset=["previous_count", "growth_rate"]).copy()

    for col in CATEGORICALS:
        latest[col] = latest[col].astype("category")

    return latest.reset_index(drop=True)
