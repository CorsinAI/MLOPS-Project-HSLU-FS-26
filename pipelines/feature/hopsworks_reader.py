"""
Reads the feature group from Hopsworks and returns it as a DataFrame.
Credentials are read from environment variables via .env (same as the writer).
"""
import pandas as pd
from dotenv import load_dotenv

from pipelines.feature.hopsworks_writer import get_feature_store

load_dotenv()


def read_feature_group(
    feature_group_name: str = "job_postings_features",
    version: int = 1,
) -> pd.DataFrame:
    """
    Read all rows from the Hopsworks feature group and return them as a
    DataFrame with window_start parsed as datetime.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group(name=feature_group_name, version=version)
    df = fg.read()
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df
