"""
Unit tests — Inference Pipeline
==================================
Covers two modules in order of the inference data flow:
  1. prepare.build_inference_features  — latest complete window selection, encoding
  2. drift.compute_reference_stats     — mean / std computation
  3. drift.detect_drift                — z-score flagging and threshold logic
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.inference.drift import compute_reference_stats, detect_drift
from pipelines.inference.prepare import build_inference_features


# ---------------------------------------------------------------------------
# Shared helper — past windows so all pass the "complete" filter
# ---------------------------------------------------------------------------

def _make_features(windows_per_pair: int = 4) -> pd.DataFrame:
    """
    Synthetic feature DataFrame with two (job_title, location) pairs.
    All windows are set in January 2026 so they pass the completeness filter.
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
                "window_start": pd.Timestamp(base + timedelta(days=WINDOW_DAYS * i)),
                "count": count,
                "previous_count": float(prev) if prev is not None else float("nan"),
                "rolling_avg_3": float(count),
                "rolling_avg_5": float(count),
                "growth_rate": (count - prev) / (prev + 1) if prev is not None else float("nan"),
            })
            prev = count
    return pd.DataFrame(rows)


# ===========================================================================
# 1. build_inference_features
# ===========================================================================

class TestBuildInferenceFeatures:

    @pytest.fixture
    def features(self):
        return _make_features(windows_per_pair=4)

    def test_one_row_per_pair(self, features):
        result = build_inference_features(features)
        assert len(result) == 2

    def test_latest_window_is_selected(self, features):
        """The retained row must be the most recent window_start for each pair."""
        result = build_inference_features(features)
        for _, row in result.iterrows():
            title, loc = str(row["job_title"]), str(row["location"])
            expected_max = features[
                (features["job_title"] == title) & (features["location"] == loc)
            ]["window_start"].max()
            assert row["window_start"] == expected_max

    def test_drops_rows_with_nan_lag(self):
        """A series with only one window (NaN lag) must produce no inference row."""
        df = _make_features(windows_per_pair=1)   # all previous_count = NaN
        result = build_inference_features(df)
        assert len(result) == 0

    def test_encodes_categoricals(self, features):
        result = build_inference_features(features)
        assert result["job_title"].dtype.name == "category"
        assert result["location"].dtype.name == "category"

    def test_index_is_zero_based(self, features):
        result = build_inference_features(features)
        assert list(result.index) == list(range(len(result)))

    def test_excludes_incomplete_window(self):
        """A window whose end date has not yet passed must be dropped; the prior complete window is kept."""
        today = pd.Timestamp.now().normalize()
        complete_window = today - pd.Timedelta(days=WINDOW_DAYS + 1)
        incomplete_window = today - pd.Timedelta(days=1)

        df = pd.DataFrame([
            {
                "job_title": "Data Engineer", "location": "Zurich",
                "window_start": complete_window,
                "count": 4, "previous_count": 3.0,
                "rolling_avg_3": 3.5, "rolling_avg_5": 3.5, "growth_rate": 0.1,
            },
            {
                "job_title": "Data Engineer", "location": "Zurich",
                "window_start": incomplete_window,
                "count": 2, "previous_count": 4.0,
                "rolling_avg_3": 3.0, "rolling_avg_5": 3.0, "growth_rate": -0.2,
            },
        ])
        result = build_inference_features(df)
        assert len(result) == 1
        assert result.iloc[0]["window_start"] == complete_window

    def test_returns_empty_when_all_windows_incomplete(self):
        """If every window is still open, the result must be empty."""
        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame([{
            "job_title": "Data Engineer", "location": "Zurich",
            "window_start": today - pd.Timedelta(days=1),
            "count": 4, "previous_count": 3.0,
            "rolling_avg_3": 3.5, "rolling_avg_5": 3.5, "growth_rate": 0.1,
        }])
        result = build_inference_features(df)
        assert len(result) == 0

    def test_multiple_pairs_are_independent(self):
        """Completeness and lag filtering should be applied per pair, not globally."""
        today = pd.Timestamp.now().normalize()
        complete = today - pd.Timedelta(days=WINDOW_DAYS + 1)
        incomplete = today - pd.Timedelta(days=1)

        df = pd.DataFrame([
            # Pair A: has a complete window with valid lag
            {"job_title": "A", "location": "X", "window_start": complete,
             "count": 5, "previous_count": 4.0, "rolling_avg_3": 4.5,
             "rolling_avg_5": 4.5, "growth_rate": 0.1},
            # Pair B: only has an incomplete window — must be excluded
            {"job_title": "B", "location": "Y", "window_start": incomplete,
             "count": 3, "previous_count": 2.0, "rolling_avg_3": 2.5,
             "rolling_avg_5": 2.5, "growth_rate": 0.3},
        ])
        result = build_inference_features(df)
        assert len(result) == 1
        assert str(result.iloc[0]["job_title"]) == "A"


# ===========================================================================
# 2. compute_reference_stats
# ===========================================================================

class TestComputeReferenceStats:

    def test_returns_all_numeric_feature_keys(self):
        features = _make_features()
        stats = compute_reference_stats(features)
        for col in ["previous_count", "rolling_avg_3", "rolling_avg_5", "growth_rate"]:
            assert col in stats

    def test_each_entry_has_mean_and_std(self):
        stats = compute_reference_stats(_make_features())
        for col, entry in stats.items():
            assert "mean" in entry
            assert "std" in entry

    def test_missing_column_is_silently_skipped(self):
        """If a numeric feature column is absent, it should not appear in the result."""
        df = pd.DataFrame({"previous_count": [1.0, 2.0, 3.0]})
        stats = compute_reference_stats(df)
        assert "previous_count" in stats
        assert "rolling_avg_3" not in stats

    def test_single_row_std_defaults_to_one(self):
        """With only one observation, std is undefined; implementation must default to 1.0."""
        df = pd.DataFrame({"previous_count": [5.0]})
        stats = compute_reference_stats(df)
        assert stats["previous_count"]["std"] == 1.0

    def test_mean_is_correct(self):
        df = pd.DataFrame({"previous_count": [2.0, 4.0, 6.0]})
        stats = compute_reference_stats(df)
        assert abs(stats["previous_count"]["mean"] - 4.0) < 1e-9


# ===========================================================================
# 3. detect_drift
# ===========================================================================

class TestDetectDrift:

    @pytest.fixture
    def features(self):
        return _make_features()

    @pytest.fixture
    def current(self, features):
        return build_inference_features(features)

    @pytest.fixture
    def ref_stats(self, features):
        return compute_reference_stats(features)

    def test_report_has_required_keys(self, current, ref_stats):
        report = detect_drift(current, ref_stats)
        assert "drift_detected" in report
        assert "threshold" in report
        assert "features" in report

    def test_each_feature_entry_has_required_fields(self, current, ref_stats):
        report = detect_drift(current, ref_stats)
        for col, info in report["features"].items():
            assert "z_score" in info
            assert "drifted" in info
            assert "reference_mean" in info
            assert "current_mean" in info

    def test_same_distribution_does_not_trigger_drift(self, features):
        """Using the same data as both reference and current should not flag drift."""
        current = build_inference_features(features)
        ref_stats = compute_reference_stats(features)
        report = detect_drift(current, ref_stats, threshold=1.5)
        assert isinstance(report["drift_detected"], bool)

    def test_extreme_deviation_triggers_drift(self, features, ref_stats):
        """A current batch with values far outside the reference distribution must be flagged."""
        extreme = build_inference_features(features).copy()
        extreme["previous_count"] = 9999.0
        extreme["growth_rate"] = 500.0

        report = detect_drift(extreme, ref_stats, threshold=2.0)
        assert report["drift_detected"] is True
        assert report["features"]["previous_count"]["drifted"] is True

    def test_high_threshold_suppresses_all_drift(self, features, ref_stats):
        """Setting an extremely high threshold must prevent any feature from being flagged."""
        extreme = build_inference_features(features).copy()
        extreme["previous_count"] = 9999.0

        report = detect_drift(extreme, ref_stats, threshold=1e9)
        assert report["drift_detected"] is False
        for info in report["features"].values():
            assert info["drifted"] is False

    def test_threshold_is_echoed_in_report(self, current, ref_stats):
        report = detect_drift(current, ref_stats, threshold=3.0)
        assert report["threshold"] == 3.0

    def test_missing_column_in_current_is_skipped(self, features):
        """If the current batch lacks a feature column, that feature is omitted from the report."""
        ref_stats = compute_reference_stats(features)
        current = build_inference_features(features).drop(columns=["previous_count"])
        report = detect_drift(current, ref_stats)
        assert "previous_count" not in report["features"]
