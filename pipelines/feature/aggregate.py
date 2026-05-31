"""
Reads normalized postings and aggregates them into a time-series DataFrame
with one row per (job_title, location, window_start).
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

WINDOW_DAYS = 7
DATA_START = datetime(2026, 1, 4)  # earlier data has gaps; exclude to avoid noisy time series


def load_postings(path: str | Path) -> pd.DataFrame:
    """
    Read a JSONL file of normalized job postings and return a DataFrame
    with columns: job_title, location, publication_date.
    """
    records = []
    with open(path) as f:
        for line in f:
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
