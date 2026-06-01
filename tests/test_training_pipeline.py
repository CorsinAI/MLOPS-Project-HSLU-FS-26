"""
Unit tests — Training Pipeline
================================
Covers the prepare module in order of the training data flow:
  1. build_dataset       — target creation, NaN dropping, gap filtering
  2. encode_categoricals — dtype casting
  3. time_split          — temporal ordering, no leakage
  4. get_X_y             — feature / target extraction
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.training.prepare import (
    CATEGORICALS,
    FEATURES,
    HOLDOUT_WINDOWS,
    TARGET,
    build_dataset,
    encode_categoricals,
    get_X_y,
    time_split,
    trim_for_training,
)
from tests.conftest import make_features_df


# ---------------------------------------------------------------------------
# Shared helper — uses real WINDOW_DAYS (7) spacing so build_dataset keeps rows
# ---------------------------------------------------------------------------

def _raw(windows: int = 6) -> pd.DataFrame:
    """Synthetic feature DataFrame with correct 7-day window spacing."""
    return make_features_df(windows_per_pair=windows)


# ===========================================================================
# 1. Holdout Trimming
# ===========================================================================

class TestTrimForTraining:

    def test_excludes_last_n_complete_windows(self):
        features = _raw(windows=6)
        trimmed = trim_for_training(features, n_holdout=2)
        all_windows = sorted(features["window_start"].unique())
        trimmed_windows = set(trimmed["window_start"].unique())
        assert all_windows[-1] not in trimmed_windows
        assert all_windows[-2] not in trimmed_windows
        for w in all_windows[:-2]:
            assert w in trimmed_windows

    def test_default_holdout_matches_constant(self):
        """trim_for_training() with no arguments must hold out exactly HOLDOUT_WINDOWS windows."""
        features = _raw(windows=6)
        all_windows = sorted(features["window_start"].unique())
        trimmed = trim_for_training(features)
        trimmed_windows = sorted(trimmed["window_start"].unique())
        assert len(trimmed_windows) == len(all_windows) - HOLDOUT_WINDOWS

    def test_does_not_mutate_input(self):
        features = _raw()
        original_len = len(features)
        trim_for_training(features)
        assert len(features) == original_len

    def test_too_few_windows_returns_all(self):
        """With fewer windows than n_holdout, the full DataFrame must be returned unchanged."""
        features = _raw(windows=2)
        trimmed = trim_for_training(features, n_holdout=3)
        assert set(trimmed["window_start"].unique()) == set(features["window_start"].unique())

    def test_incomplete_window_not_counted_as_holdout_candidate(self):
        """A window whose period has not yet closed must not count toward the n_holdout quota."""
        today = pd.Timestamp.now().normalize()
        incomplete_start = today - pd.Timedelta(days=1)  # window still open

        features = _raw(windows=4)
        # Append rows with an incomplete window_start
        extra = features.iloc[:2].copy()
        extra["window_start"] = incomplete_start
        df = pd.concat([features, extra], ignore_index=True)

        trimmed = trim_for_training(df, n_holdout=2)
        # The incomplete window must not appear in the result
        assert incomplete_start not in set(trimmed["window_start"])
        # The 2 most recent *complete* windows must be excluded
        complete_windows = sorted(features["window_start"].unique())
        assert complete_windows[-1] not in set(trimmed["window_start"])
        assert complete_windows[-2] not in set(trimmed["window_start"])


# ===========================================================================
# 2. Dataset Construction  (operates on already-trimmed features)
# ===========================================================================

class TestBuildDataset:

    def test_target_column_is_present(self):
        ds = build_dataset(_raw())
        assert TARGET in ds.columns

    def test_drops_nan_lag_rows(self):
        """Rows where previous_count or growth_rate are NaN must be excluded."""
        ds = build_dataset(_raw())
        assert ds["previous_count"].isna().sum() == 0
        assert ds["growth_rate"].isna().sum() == 0

    def test_last_window_per_pair_is_dropped(self):
        """The last window has no successor, so it cannot produce a target."""
        ds = build_dataset(_raw(windows=4))
        # Per pair: window 0 dropped (NaN lag), windows 1-2 kept, window 3 dropped (no next)
        assert len(ds) == (4 - 2) * 2   # 2 kept × 2 pairs

    def test_target_equals_next_window_count(self):
        """Each row's target must equal the count of the immediately following window."""
        ds = build_dataset(_raw())
        de_z = (
            ds[(ds["job_title"] == "Data Engineer") & (ds["location"] == "Zurich")]
            .sort_values("window_start")
            .reset_index(drop=True)
        )
        for i in range(len(de_z) - 1):
            assert de_z.loc[i, TARGET] == de_z.loc[i + 1, "count"]

    def test_gap_rows_are_dropped(self):
        """A row followed by a non-consecutive window must not appear in the dataset."""
        base = datetime(2026, 1, 4)
        df = pd.DataFrame([
            {
                "job_title": "A", "location": "B",
                "window_start": pd.Timestamp(base),
                "count": 3, "previous_count": float("nan"),
                "rolling_avg_3": float("nan"), "rolling_avg_5": float("nan"),
                "growth_rate": float("nan"),
            },
            {
                "job_title": "A", "location": "B",
                "window_start": pd.Timestamp(base + timedelta(days=WINDOW_DAYS)),
                "count": 4, "previous_count": 3.0,
                "rolling_avg_3": 3.0, "rolling_avg_5": 3.0, "growth_rate": 0.25,
            },
            # Gap is 2× WINDOW_DAYS — next window is not consecutive
            {
                "job_title": "A", "location": "B",
                "window_start": pd.Timestamp(base + timedelta(days=WINDOW_DAYS * 3)),
                "count": 5, "previous_count": 4.0,
                "rolling_avg_3": 4.0, "rolling_avg_5": 4.0, "growth_rate": 0.2,
            },
        ])
        ds = build_dataset(df)
        # base: NaN lag → dropped; WINDOW_DAYS: gap to next = 14 ≠ 7 → dropped; WINDOW_DAYS*3: no next → dropped
        assert len(ds) == 0

    def test_empty_input_returns_empty_dataset(self):
        # Explicit dtypes are required so the datetime subtraction inside
        # build_dataset doesn't receive an object-typed window_start column.
        empty = pd.DataFrame({
            "job_title": pd.Series([], dtype=str),
            "location": pd.Series([], dtype=str),
            "window_start": pd.Series([], dtype="datetime64[ns]"),
            "count": pd.Series([], dtype=float),
            "previous_count": pd.Series([], dtype=float),
            "rolling_avg_3": pd.Series([], dtype=float),
            "rolling_avg_5": pd.Series([], dtype=float),
            "growth_rate": pd.Series([], dtype=float),
        })
        ds = build_dataset(empty)
        assert len(ds) == 0


# ===========================================================================
# 2. Categorical Encoding
# ===========================================================================

class TestEncodeCategoricals:

    @pytest.fixture
    def dataset(self):
        return build_dataset(_raw())

    def test_categorical_columns_have_category_dtype(self, dataset):
        encoded = encode_categoricals(dataset)
        for col in CATEGORICALS:
            assert encoded[col].dtype.name == "category", f"{col} should be category dtype"

    def test_numeric_columns_are_unchanged(self, dataset):
        encoded = encode_categoricals(dataset)
        assert pd.api.types.is_numeric_dtype(encoded["count"])
        assert pd.api.types.is_numeric_dtype(encoded["previous_count"])

    def test_does_not_mutate_input(self, dataset):
        original_dtype = dataset["job_title"].dtype
        encode_categoricals(dataset)
        assert dataset["job_title"].dtype == original_dtype


# ===========================================================================
# 3. Time Split
# ===========================================================================

class TestTimeSplit:

    @pytest.fixture
    def dataset(self):
        return build_dataset(_raw(windows=8))

    def test_train_precedes_val_precedes_test(self, dataset):
        train, val, test = time_split(dataset)
        assert train["window_start"].max() < val["window_start"].min()
        assert val["window_start"].max() < test["window_start"].min()

    def test_all_rows_are_preserved(self, dataset):
        train, val, test = time_split(dataset)
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_splits_are_disjoint(self, dataset):
        train, val, test = time_split(dataset)
        train_w = set(train["window_start"])
        val_w = set(val["window_start"])
        test_w = set(test["window_start"])
        assert not train_w & val_w
        assert not train_w & test_w
        assert not val_w & test_w

    def test_train_is_largest_split(self, dataset):
        train, val, test = time_split(dataset)
        assert len(train) >= len(val)
        assert len(train) >= len(test)

    def test_custom_fractions_are_respected(self):
        """With 10 unique windows and 0.7/0.2 split → train=7, val=2, test=1."""
        base = datetime(2026, 1, 4)
        rows = [
            {
                "job_title": "A", "location": "B",
                "window_start": pd.Timestamp(base + timedelta(days=WINDOW_DAYS * i)),
                "count": 5, "previous_count": 4.0,
                "rolling_avg_3": 4.0, "rolling_avg_5": 4.0,
                "growth_rate": 0.1, "target": 5.0,
            }
            for i in range(1, 11)  # windows 1–10 (window 0 has NaN lag, skip it)
        ]
        df = pd.DataFrame(rows)
        train, val, test = time_split(df, train_frac=0.7, val_frac=0.2)
        assert len(train["window_start"].unique()) == 7
        assert len(val["window_start"].unique()) == 2
        assert len(test["window_start"].unique()) == 1


# ===========================================================================
# 4. Feature / Target Extraction
# ===========================================================================

class TestGetXY:

    @pytest.fixture
    def dataset(self):
        return build_dataset(_raw())

    def test_X_has_exactly_feature_columns(self, dataset):
        X, _ = get_X_y(dataset)
        assert list(X.columns) == FEATURES

    def test_y_has_target_name(self, dataset):
        _, y = get_X_y(dataset)
        assert y.name == TARGET

    def test_target_not_in_X(self, dataset):
        X, _ = get_X_y(dataset)
        assert TARGET not in X.columns

    def test_X_and_y_have_same_length(self, dataset):
        X, y = get_X_y(dataset)
        assert len(X) == len(y)
