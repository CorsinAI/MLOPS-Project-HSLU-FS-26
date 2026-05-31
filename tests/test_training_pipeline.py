"""Smoke tests for the training pipeline."""
import pandas as pd
import pytest

from pipelines.training.prepare import (
    build_dataset,
    encode_categoricals,
    time_split,
    get_X_y,
    FEATURES,
    TARGET,
    CATEGORICALS,
)


# ---------------------------------------------------------------------------
# Minimal feature DataFrame that mirrors real pipeline output
# ---------------------------------------------------------------------------

def make_features():
    """Three consecutive windows for two (title, location) pairs."""
    rows = []
    from datetime import datetime, timedelta
    base = datetime(2026, 1, 4)
    for title, loc in [("System Engineer", "Zurich"), ("IT Support / Helpdesk", "Bern")]:
        for i in range(6):
            rows.append({
                "job_title": title,
                "location": loc,
                "window_start": pd.Timestamp(base + timedelta(days=3 * i)),
                "count": 5 + i,
                "previous_count": float(4 + i) if i > 0 else float("nan"),
                "rolling_avg_3": float(4 + i) if i > 0 else float("nan"),
                "rolling_avg_5": float(4 + i) if i > 0 else float("nan"),
                "growth_rate": 0.1 * i if i > 0 else float("nan"),
                "day_of_week": (base + timedelta(days=3 * i)).weekday(),
                "month": (base + timedelta(days=3 * i)).month,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def features_df():
    return make_features()


@pytest.fixture
def dataset(features_df):
    return build_dataset(features_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_dataset_has_target(dataset):
    assert TARGET in dataset.columns


def test_build_dataset_drops_nan_lag_rows(dataset, features_df):
    # Rows with NaN previous_count (first window) must be gone
    assert dataset["previous_count"].isna().sum() == 0


def test_build_dataset_target_is_next_count(dataset):
    # For each row, target should equal count of the next consecutive window
    se = dataset[(dataset["job_title"] == "System Engineer") & (dataset["location"] == "Zurich")]
    se = se.sort_values("window_start").reset_index(drop=True)
    for i in range(len(se) - 1):
        assert se.loc[i, TARGET] == se.loc[i + 1, "count"]


def test_build_dataset_no_gap_rows(dataset):
    # No row should have a target derived from a non-consecutive window
    from pipelines.feature.aggregate import WINDOW_DAYS
    for (title, loc), grp in dataset.groupby(["job_title", "location"]):
        grp = grp.sort_values("window_start")
        diffs = grp["window_start"].diff().dropna().dt.days
        assert (diffs == WINDOW_DAYS).all()


def test_encode_categoricals(dataset):
    encoded = encode_categoricals(dataset)
    for col in CATEGORICALS:
        assert encoded[col].dtype.name == "category"


def test_time_split_no_leakage(dataset):
    train, val, test = time_split(dataset)
    # Strictly ordered: train < val < test
    assert train["window_start"].max() < val["window_start"].min()
    assert val["window_start"].max() < test["window_start"].min()
    # No rows lost
    assert len(train) + len(val) + len(test) == len(dataset)


def test_time_split_proportions():
    # Use 10 windows so 70/20/10 divides cleanly
    from datetime import datetime, timedelta
    base = datetime(2026, 1, 4)
    rows = []
    for title, loc in [("System Engineer", "Zurich")]:
        for i in range(1, 11):  # windows 1-10 (skip window 0 — no lag)
            rows.append({
                "job_title": title, "location": loc,
                "window_start": pd.Timestamp(base + timedelta(days=3 * i)),
                "count": 5, "previous_count": 4.0, "rolling_avg_3": 4.0,
                "rolling_avg_5": 4.0, "growth_rate": 0.1,
                "day_of_week": 0, "month": 1, "target": 5.0,
            })
    df = pd.DataFrame(rows)
    train, val, test = time_split(df, train_frac=0.7, val_frac=0.2)
    n = 10
    assert len(train["window_start"].unique()) == int(n * 0.7)
    assert len(val["window_start"].unique()) == int(n * 0.2)
    assert len(test["window_start"].unique()) == n - int(n * 0.7) - int(n * 0.2)


def test_get_X_y_columns(dataset):
    X, y = get_X_y(dataset)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET
