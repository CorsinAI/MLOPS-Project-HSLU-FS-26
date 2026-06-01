"""
Unit tests — Feature Pipeline
=================================
Covers three modules in order of data flow:
  1. aggregate.assign_windows     — date binning
  2. aggregate.aggregate_counts   — groupby counting
  3. features.compute_features    — lag / rolling feature computation
"""
import pandas as pd
import pytest

from pipelines.feature.aggregate import (
    DATA_START,
    WINDOW_DAYS,
    aggregate_counts,
    assign_windows,
)
from pipelines.feature.features import compute_features
from tests.conftest import SAMPLE_RECORDS


# ===========================================================================
# 1. Window Assignment
# ===========================================================================

class TestAssignWindows:

    def test_drops_postings_before_data_start(self, raw_df):
        windowed = assign_windows(raw_df)
        assert (windowed["publication_date"] >= pd.Timestamp(DATA_START)).all()

    def test_adds_window_start_column(self, raw_df):
        assert "window_start" in assign_windows(raw_df).columns

    def test_window_starts_on_7day_boundaries(self, raw_df):
        epoch = pd.Timestamp(DATA_START).normalize()
        for ws in assign_windows(raw_df)["window_start"]:
            assert (ws - epoch).days % WINDOW_DAYS == 0

    def test_postings_in_same_week_share_window(self, raw_df):
        windowed = assign_windows(raw_df)
        de_zurich = windowed[
            (windowed["job_title"] == "Data Engineer") & (windowed["location"] == "Zurich")
        ]
        # Jan 4 and Jan 5 are in the same 7-day window
        w_jan4 = de_zurich[de_zurich["publication_date"].dt.day == 4]["window_start"].iloc[0]
        w_jan5 = de_zurich[de_zurich["publication_date"].dt.day == 5]["window_start"].iloc[0]
        assert w_jan4 == w_jan5

    def test_postings_in_different_weeks_have_different_windows(self, raw_df):
        windowed = assign_windows(raw_df)
        de_zurich = windowed[
            (windowed["job_title"] == "Data Engineer") & (windowed["location"] == "Zurich")
        ]
        windows = de_zurich["window_start"].unique()
        assert len(windows) == 2  # Jan 4–10 and Jan 11–17


# ===========================================================================
# 2. Count Aggregation
# ===========================================================================

class TestAggregateCounts:

    def test_output_has_exactly_four_columns(self, counts_df):
        assert set(counts_df.columns) == {"job_title", "location", "window_start", "count"}

    def test_same_window_postings_are_summed(self, counts_df):
        # Jan 4 + Jan 5 → same window for Data Engineer / Zurich → count == 2
        row = counts_df[
            (counts_df["job_title"] == "Data Engineer") & (counts_df["location"] == "Zurich")
        ]
        assert 2 in row["count"].values

    def test_different_job_title_is_separate_row(self, counts_df):
        ce = counts_df[(counts_df["job_title"] == "Cloud Engineer") & (counts_df["location"] == "Bern")]
        assert len(ce) == 1
        assert ce.iloc[0]["count"] == 1

    def test_different_location_same_title_is_separate_row(self, counts_df):
        # Data Engineer appears in both Zurich and Bern
        titles = counts_df[counts_df["job_title"] == "Data Engineer"]["location"].unique()
        assert "Zurich" in titles

    def test_result_is_sorted_by_title_location_window(self, counts_df):
        expected = (
            counts_df
            .sort_values(["job_title", "location", "window_start"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(counts_df.reset_index(drop=True), expected)


# ===========================================================================
# 3. Feature Computation
# ===========================================================================

class TestComputeFeatures:

    def test_all_lag_columns_added(self, counts_df):
        features = compute_features(counts_df)
        for col in ["previous_count", "rolling_avg_3", "rolling_avg_5", "growth_rate"]:
            assert col in features.columns

    def test_first_window_lag_features_are_nan(self, counts_df):
        features = compute_features(counts_df)
        # Use idxmin to get the earliest window per group, avoiding groupby.first()
        # which skips NaN values by default in pandas 2.x.
        for (_, _), grp in features.groupby(["job_title", "location"]):
            earliest = grp.loc[grp["window_start"].idxmin()]
            assert pd.isna(earliest["previous_count"])
            assert pd.isna(earliest["growth_rate"])

    def test_previous_count_equals_prior_window_count(self, counts_df):
        features = compute_features(counts_df)
        de_z = (
            features[(features["job_title"] == "Data Engineer") & (features["location"] == "Zurich")]
            .sort_values("window_start")
            .reset_index(drop=True)
        )
        assert de_z.loc[1, "previous_count"] == de_z.loc[0, "count"]

    def test_growth_rate_formula(self, counts_df):
        features = compute_features(counts_df)
        de_z = (
            features[(features["job_title"] == "Data Engineer") & (features["location"] == "Zurich")]
            .sort_values("window_start")
            .reset_index(drop=True)
        )
        row = de_z.iloc[1]
        expected = (row["count"] - row["previous_count"]) / (row["previous_count"] + 1)
        assert abs(row["growth_rate"] - expected) < 1e-9

    def test_no_temporal_leakage_columns(self, counts_df):
        features = compute_features(counts_df)
        assert "day_of_week" not in features.columns
        assert "month" not in features.columns

    def test_single_window_pair_does_not_crash(self):
        df = pd.DataFrame([{
            "job_title": "X", "location": "Y",
            "window_start": pd.Timestamp("2026-01-04"),
            "count": 5,
        }])
        features = compute_features(df)
        assert len(features) == 1
        assert pd.isna(features.iloc[0]["previous_count"])
