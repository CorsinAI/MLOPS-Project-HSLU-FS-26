"""Smoke tests for the inference pipeline (prepare and drift modules)."""
import pandas as pd
import pytest

from pipelines.inference.drift import compute_reference_stats, detect_drift
from pipelines.inference.prepare import build_inference_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_features(windows_per_pair: int = 4) -> pd.DataFrame:
    """
    Minimal feature DataFrame with two (job_title, location) pairs,
    each observed over `windows_per_pair` consecutive 3-day windows.
    """
    from datetime import datetime, timedelta

    base = datetime(2026, 1, 4)
    rows = []
    for title, location in [("Data Engineer", "Zurich"), ("Cloud Engineer", "Bern")]:
        prev = None
        for i in range(windows_per_pair):
            count = 3 + i
            rows.append(
                {
                    "job_title": title,
                    "location": location,
                    "window_start": base + timedelta(days=3 * i),
                    "count": count,
                    "previous_count": prev,
                    "rolling_avg_3": float(count),
                    "rolling_avg_5": float(count),
                    "growth_rate": None if prev is None else (count - prev) / (prev + 1),
                    "day_of_week": (base + timedelta(days=3 * i)).weekday(),
                    "month": (base + timedelta(days=3 * i)).month,
                }
            )
            prev = count
    return pd.DataFrame(rows)


@pytest.fixture
def features_df() -> pd.DataFrame:
    return _make_features(windows_per_pair=4)


# ---------------------------------------------------------------------------
# prepare.build_inference_features
# ---------------------------------------------------------------------------


def test_build_inference_features_selects_latest_window(features_df):
    """Each (job_title, location) pair should appear exactly once — the latest window."""
    result = build_inference_features(features_df)
    assert len(result) == 2  # one row per pair
    pairs = set(zip(result["job_title"].astype(str), result["location"].astype(str)))
    assert pairs == {("Data Engineer", "Zurich"), ("Cloud Engineer", "Bern")}


def test_build_inference_features_latest_is_last_window(features_df):
    """The retained row must be the most recent window_start for each pair."""
    result = build_inference_features(features_df)
    for _, row in result.iterrows():
        title, loc = str(row["job_title"]), str(row["location"])
        group_max = features_df[
            (features_df["job_title"] == title) & (features_df["location"] == loc)
        ]["window_start"].max()
        assert row["window_start"] == group_max


def test_build_inference_features_drops_nan_lag_rows():
    """Rows whose lag features are NaN (first window of a series) must be dropped."""
    df = _make_features(windows_per_pair=1)  # only one window per pair → previous_count=None
    result = build_inference_features(df)
    assert len(result) == 0


def test_build_inference_features_encodes_categoricals(features_df):
    """job_title and location must be encoded as pandas Categorical dtype."""
    result = build_inference_features(features_df)
    assert result["job_title"].dtype.name == "category"
    assert result["location"].dtype.name == "category"


def test_build_inference_features_reset_index(features_df):
    """Returned index must be 0-based integer range."""
    result = build_inference_features(features_df)
    assert list(result.index) == list(range(len(result)))


def test_build_inference_features_excludes_incomplete_window():
    """A window that started less than WINDOW_DAYS ago must be excluded; the previous complete window is used instead."""
    from pipelines.feature.aggregate import WINDOW_DAYS

    today = pd.Timestamp.now().normalize()
    complete_window = today - pd.Timedelta(days=WINDOW_DAYS + 1)
    incomplete_window = today - pd.Timedelta(days=1)

    df = pd.DataFrame([
        {
            "job_title": "Data Engineer",
            "location": "Zurich",
            "window_start": complete_window,
            "count": 4,
            "previous_count": 3.0,
            "rolling_avg_3": 3.5,
            "rolling_avg_5": 3.5,
            "growth_rate": 0.1,
        },
        {
            "job_title": "Data Engineer",
            "location": "Zurich",
            "window_start": incomplete_window,
            "count": 2,
            "previous_count": 4.0,
            "rolling_avg_3": 3.0,
            "rolling_avg_5": 3.0,
            "growth_rate": -0.2,
        },
    ])

    result = build_inference_features(df)
    assert len(result) == 1
    assert result.iloc[0]["window_start"] == complete_window


def test_build_inference_features_returns_empty_if_all_windows_incomplete():
    """If every window is still open, the result must be empty."""
    from pipelines.feature.aggregate import WINDOW_DAYS

    today = pd.Timestamp.now().normalize()
    incomplete_window = today - pd.Timedelta(days=1)

    df = pd.DataFrame([{
        "job_title": "Data Engineer",
        "location": "Zurich",
        "window_start": incomplete_window,
        "count": 4,
        "previous_count": 3.0,
        "rolling_avg_3": 3.5,
        "rolling_avg_5": 3.5,
        "growth_rate": 0.1,
    }])

    result = build_inference_features(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# drift.compute_reference_stats
# ---------------------------------------------------------------------------


def test_compute_reference_stats_returns_expected_keys(features_df):
    stats = compute_reference_stats(features_df)
    for col in ["previous_count", "rolling_avg_3", "rolling_avg_5", "growth_rate"]:
        assert col in stats
        assert "mean" in stats[col]
        assert "std" in stats[col]


def test_compute_reference_stats_ignores_missing_columns():
    """Should not raise if a numeric feature column is absent."""
    df = pd.DataFrame({"previous_count": [1.0, 2.0, 3.0]})
    stats = compute_reference_stats(df)
    assert "previous_count" in stats
    assert "rolling_avg_3" not in stats


# ---------------------------------------------------------------------------
# drift.detect_drift
# ---------------------------------------------------------------------------


def test_detect_drift_no_drift_within_threshold(features_df):
    """Using the same data as reference and current should produce near-zero z-scores."""
    current = build_inference_features(features_df)
    ref_stats = compute_reference_stats(features_df)
    report = detect_drift(current, ref_stats, threshold=2.0)
    # Current batch is drawn from the same distribution — no drift expected
    assert isinstance(report["drift_detected"], bool)
    for col, info in report["features"].items():
        assert "z_score" in info
        assert "drifted" in info


def test_detect_drift_flags_large_deviation(features_df):
    """A batch with drastically different means must be flagged as drifted."""
    ref_stats = compute_reference_stats(features_df)

    # Build a current batch with values 100× higher than the reference
    extreme = build_inference_features(features_df).copy()
    extreme["previous_count"] = 9999.0
    extreme["growth_rate"] = 500.0

    report = detect_drift(extreme, ref_stats, threshold=2.0)
    assert report["drift_detected"] is True
    assert report["features"]["previous_count"]["drifted"] is True


def test_detect_drift_threshold_respected(features_df):
    """Setting an extremely high threshold should suppress all drift flags."""
    ref_stats = compute_reference_stats(features_df)
    extreme = build_inference_features(features_df).copy()
    extreme["previous_count"] = 9999.0

    report = detect_drift(extreme, ref_stats, threshold=1e9)
    assert report["drift_detected"] is False
