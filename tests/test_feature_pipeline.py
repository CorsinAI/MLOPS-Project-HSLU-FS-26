"""Smoke tests for the feature pipeline."""
import json

import pandas as pd
import pytest

from pipelines.feature.aggregate import _read_jsonl_lines, assign_windows, aggregate_counts
from pipelines.feature.features import compute_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "vacancy_link": "link-1",
        "job_title": "Data Engineer",
        "company_name": "Acme AG",
        "location": "Zurich",
        "publication_date": "04.01.2026",  # window 0 (epoch)
        "company_website": "",
        "employment_type": "Permanent",
        "workload_percentage": "100%",
        "language": ["German"],
        "industry_sector": "Tech",
        "technical_skills_keywords": ["Python", "Spark"],
        "soft_skills_keywords": [],
        "seniority": "Mid",
        "salary_info_summary": "",
        "benefits_keywords": [],
    },
    {
        "vacancy_link": "link-2",
        "job_title": "Data Engineer",
        "company_name": "Beta GmbH",
        "location": "Zurich",
        "publication_date": "05.01.2026",  # window 0
        "company_website": "",
        "employment_type": "Permanent",
        "workload_percentage": "100%",
        "language": ["German"],
        "industry_sector": "Tech",
        "technical_skills_keywords": ["Python", "Kafka"],
        "soft_skills_keywords": [],
        "seniority": "Senior",
        "salary_info_summary": "",
        "benefits_keywords": [],
    },
    {
        "vacancy_link": "link-3",
        "job_title": "Data Engineer",
        "company_name": "Gamma SA",
        "location": "Zurich",
        "publication_date": "11.01.2026",  # window 1 (Jan 11–17)
        "company_website": "",
        "employment_type": "Permanent",
        "workload_percentage": "80%",
        "language": ["French"],
        "industry_sector": "Finance",
        "technical_skills_keywords": ["Python"],
        "soft_skills_keywords": [],
        "seniority": "Junior",
        "salary_info_summary": "",
        "benefits_keywords": [],
    },
    {
        "vacancy_link": "link-4",
        "job_title": "Cloud Engineer",
        "company_name": "Delta AG",
        "location": "Bern",
        "publication_date": "04.01.2026",  # window 0
        "company_website": "",
        "employment_type": "Temporary",
        "workload_percentage": "100%",
        "language": ["German"],
        "industry_sector": "Banking",
        "technical_skills_keywords": ["Kubernetes", "Terraform"],
        "soft_skills_keywords": [],
        "seniority": "Mid",
        "salary_info_summary": "",
        "benefits_keywords": [],
    },
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    lines = [json.dumps(r) for r in SAMPLE_RECORDS]
    return _read_jsonl_lines(lines)


@pytest.fixture
def windowed_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return assign_windows(raw_df)


@pytest.fixture
def counts_df(windowed_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_counts(windowed_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_postings_shape(raw_df):
    assert len(raw_df) == len(SAMPLE_RECORDS)
    assert set(raw_df.columns) >= {"job_title", "location", "publication_date"}


def test_load_postings_date_dtype(raw_df):
    assert pd.api.types.is_datetime64_any_dtype(raw_df["publication_date"])


def test_assign_windows_drops_pre_cutoff(raw_df):
    from pipelines.feature.aggregate import DATA_START
    windowed = assign_windows(raw_df)
    assert (windowed["publication_date"] >= pd.Timestamp(DATA_START)).all()


def test_assign_windows_creates_column(windowed_df):
    assert "window_start" in windowed_df.columns


def test_assign_windows_7day_buckets(windowed_df):
    from pipelines.feature.aggregate import WINDOW_DAYS, DATA_START
    epoch = pd.Timestamp(DATA_START).normalize()
    for ws in windowed_df["window_start"]:
        delta = (ws - epoch).days
        assert delta % WINDOW_DAYS == 0, f"window_start {ws} is not on a 7-day boundary"


def test_aggregate_counts_columns(counts_df):
    assert set(counts_df.columns) == {"job_title", "location", "window_start", "count"}


def test_aggregate_counts_values(counts_df):
    # "Data Engineer" in Zurich: 2 postings in window 0, 1 in window 1
    de_zurich = counts_df[(counts_df["job_title"] == "Data Engineer") & (counts_df["location"] == "Zurich")]
    assert len(de_zurich) == 2
    assert sorted(de_zurich["count"].tolist()) == [1, 2]


def test_aggregate_counts_separate_title_location(counts_df):
    # "Cloud Engineer" in Bern must appear as its own row
    ce_bern = counts_df[(counts_df["job_title"] == "Cloud Engineer") & (counts_df["location"] == "Bern")]
    assert len(ce_bern) == 1
    assert ce_bern.iloc[0]["count"] == 1


def test_compute_features_lag(counts_df):
    features = compute_features(counts_df)
    de_zurich = features[
        (features["job_title"] == "Data Engineer") & (features["location"] == "Zurich")
    ].sort_values("window_start")
    # First window has no previous_count
    assert pd.isna(de_zurich.iloc[0]["previous_count"])
    # Second window's previous_count == first window's count
    assert de_zurich.iloc[1]["previous_count"] == de_zurich.iloc[0]["count"]


def test_compute_features_rolling(counts_df):
    features = compute_features(counts_df)
    assert "rolling_avg_3" in features.columns
    assert "rolling_avg_5" in features.columns


def test_compute_features_no_temporal_columns(counts_df):
    features = compute_features(counts_df)
    assert "day_of_week" not in features.columns
    assert "month" not in features.columns
