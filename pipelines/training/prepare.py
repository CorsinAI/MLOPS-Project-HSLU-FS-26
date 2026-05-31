"""
Prepares the feature DataFrame for LightGBM training.

Steps:
  1. Create target: count of the NEXT consecutive window for each (job_title, location)
     - Rows followed by a gap (>WINDOW_DAYS) get no target and are dropped
  2. Drop rows with NaN in lag features (first window of each series)
  3. Encode categoricals as pandas Categorical (LightGBM reads these natively)
  4. Time-based 70/20/10 train/val/test split — never shuffle across time
"""
import pandas as pd

from pipelines.feature.aggregate import WINDOW_DAYS

FEATURES = [
    "job_title",
    "location",
    "previous_count",
    "rolling_avg_3",
    "rolling_avg_5",
    "growth_rate",
]
TARGET = "target"
CATEGORICALS = ["job_title", "location"]


def build_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `target` column (next window's count) and drop unusable rows.
    Only consecutive windows (gap == WINDOW_DAYS) produce a valid target.
    """
    df = features.sort_values(["job_title", "location", "window_start"]).copy()

    grp = df.groupby(["job_title", "location"])

    # Next window count within each group
    df["target"] = grp["count"].shift(-1)

    # Next window date within each group
    df["_next_window"] = grp["window_start"].shift(-1)

    # Gap to the next row
    df["_gap_days"] = (df["_next_window"] - df["window_start"]).dt.days

    # Only keep rows where the next window is exactly one step ahead
    df = df[df["_gap_days"] == WINDOW_DAYS].drop(columns=["_next_window", "_gap_days"])

    # Drop rows missing lag features (first window of each series)
    df = df.dropna(subset=["previous_count", "growth_rate"])

    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns so LightGBM treats them as categories."""
    df = df.copy()
    for col in CATEGORICALS:
        df[col] = df[col].astype("category")
    return df


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test by window_start date (70/20/10 default).
    Cutoffs are derived from the sorted unique windows so proportions are
    respected regardless of how many rows each window has.
    Never shuffles.
    """
    windows = sorted(df["window_start"].unique())
    n = len(windows)
    train_end = windows[int(n * train_frac) - 1]
    val_end = windows[int(n * (train_frac + val_frac)) - 1]

    train = df[df["window_start"] <= train_end].copy()
    val = df[(df["window_start"] > train_end) & (df["window_start"] <= val_end)].copy()
    test = df[df["window_start"] > val_end].copy()

    return train, val, test


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset DataFrame into feature matrix X and target series y."""
    return df[FEATURES], df[TARGET]
