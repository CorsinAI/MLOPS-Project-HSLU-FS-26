import pandas as pd


def compute_features(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-group lag and rolling features.

    Expected input columns: job_title, location, window_start, count
    Added columns:
      previous_count, rolling_avg_3, rolling_avg_5, growth_rate
    """
    df = counts.sort_values(["job_title", "location", "window_start"]).copy()

    grp = df.groupby(["job_title", "location"])["count"]

    df["previous_count"] = grp.shift(1)
    df["rolling_avg_3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["rolling_avg_5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["growth_rate"] = (
        (df["count"] - df["previous_count"]) / (df["previous_count"].fillna(0) + 1)
    ).where(df["previous_count"].notna())

    return df.reset_index(drop=True)
