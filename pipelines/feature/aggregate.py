"""
Reads normalized postings and aggregates them into a time-series DataFrame
with one row per (job_title, location, window_start).
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

WINDOW_DAYS = 7
DATA_START = datetime(2026, 1, 4)  # earlier data has gaps; exclude to avoid noisy time series


def _read_jsonl_lines(lines: list[str]) -> pd.DataFrame:
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records.append({
            "job_title": rec["job_title"],
            "location": rec["location"],
            "publication_date": datetime.strptime(rec["publication_date"], "%d.%m.%Y"),
        })
    return pd.DataFrame(records)


def load_postings(path: Path | None = None) -> pd.DataFrame:
    """
    Load job postings from a local JSONL file or Azure Blob Storage.

    If `path` is given, reads from that local file (no credentials needed).
    Otherwise reads from Azure Blob Storage using AZURE_SAS_URL env var.

    Returns a DataFrame with columns: job_title, location, publication_date.
    """
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return _read_jsonl_lines(f.read().splitlines())

    sas_url = os.environ["AZURE_SAS_URL"]
    from azure.storage.blob import ContainerClient
    blob_name = os.environ.get("AZURE_BLOB_NAME", "structured_jobs_normalized_cleaned.jsonl")
    container = ContainerClient.from_container_url(sas_url)
    content = container.get_blob_client(blob_name).download_blob().readall().decode("utf-8")
    return _read_jsonl_lines(content.splitlines())


def assign_windows(
    df: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    start: datetime = DATA_START,
) -> pd.DataFrame:
    """
    Drop postings before `start`, then bin publication_date into fixed
    window_days-wide buckets anchored at `start`.
    """
    df = df[df["publication_date"] >= start].copy()
    epoch = pd.Timestamp(start).normalize()
    df["window_start"] = df["publication_date"].apply(
        lambda d: epoch + timedelta(days=((d.normalize() - epoch).days // window_days) * window_days)
    )
    return df


def aggregate_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count postings per (job_title, location, window_start)."""
    counts = (
        df.groupby(["job_title", "location", "window_start"])
        .size()
        .reset_index(name="count")
        .sort_values(["job_title", "location", "window_start"])
        .reset_index(drop=True)
    )
    return counts
