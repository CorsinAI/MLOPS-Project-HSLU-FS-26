"""
Shared fixtures and helpers for all pipeline tests.

make_features_df() is a module-level helper (importable by test modules)
that builds a synthetic feature DataFrame with consecutive 7-day windows.
Fixtures raw_df, counts_df, and features_df are auto-discovered by pytest.
"""
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from pipelines.feature.aggregate import _read_jsonl_lines, assign_windows, aggregate_counts

# ---------------------------------------------------------------------------
# Minimal raw records — only the three fields the pipeline actually reads
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {"job_title": "Data Engineer", "location": "Zurich", "publication_date": "04.01.2026"},
    {"job_title": "Data Engineer", "location": "Zurich", "publication_date": "05.01.2026"},
    {"job_title": "Data Engineer", "location": "Zurich", "publication_date": "11.01.2026"},
    {"job_title": "Cloud Engineer", "location": "Bern",   "publication_date": "04.01.2026"},
    {"job_title": "Data Engineer", "location": "Bern",   "publication_date": "03.12.2025"},  # before DATA_START
]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def make_features_df(windows_per_pair: int = 6) -> pd.DataFrame:
    """
    Build a synthetic feature DataFrame with `windows_per_pair` consecutive
    7-day windows for two (job_title, location) pairs.

    Window 0 intentionally has NaN lag features (first observation).
    Windows 1+ have fully populated lag, rolling, and growth_rate columns.
    """
    base = datetime(2026, 1, 4)
    rows = []
    for title, location in [("Data Engineer", "Zurich"), ("Cloud Engineer", "Bern")]:
        prev = None
        for i in range(windows_per_pair):
            count = 3 + i
            rows.append({
                "job_title": title,
                "location": location,
                "window_start": pd.Timestamp(base + timedelta(weeks=i)),
                "count": count,
                "previous_count": float(prev) if prev is not None else float("nan"),
                "rolling_avg_3": float(count) if prev is not None else float("nan"),
                "rolling_avg_5": float(count) if prev is not None else float("nan"),
                "growth_rate": (count - prev) / (prev + 1) if prev is not None else float("nan"),
            })
            prev = count
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Parsed DataFrame from SAMPLE_RECORDS (no windowing applied)."""
    return _read_jsonl_lines([json.dumps(r) for r in SAMPLE_RECORDS])


@pytest.fixture
def counts_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated counts after window assignment."""
    return aggregate_counts(assign_windows(raw_df))


@pytest.fixture
def features_df() -> pd.DataFrame:
    """Full feature DataFrame with lag/rolling columns, 6 windows per pair."""
    return make_features_df(windows_per_pair=6)
