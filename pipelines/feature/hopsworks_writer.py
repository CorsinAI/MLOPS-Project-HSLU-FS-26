"""
Writes the feature DataFrame to a Hopsworks feature group.
Credentials are read from environment variables (never hardcoded).
"""
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_feature_store():
    import hopsworks  # imported lazily so the rest of the pipeline works without it
    project = hopsworks.login(
        host=os.environ["HOPSWORKS_HOST"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ.get("HOPSWORKS_PROJECT", "JobAnalysis"),
    )
    return project.get_feature_store()


def write_feature_group(
    df: pd.DataFrame,
    feature_group_name: str = "job_postings_features",
    version: int = 1,
    primary_key: list[str] | None = None,
    event_time: str = "window_start",
) -> None:
    if primary_key is None:
        primary_key = ["job_title", "location", "window_start"]

    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=False,
        description="Aggregated 3-day job posting counts with lag/rolling features",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} rows into feature group '{feature_group_name}' v{version}")
