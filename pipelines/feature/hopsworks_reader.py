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
    fs = get_feature_store()
    fg = fs.get_feature_group(name=feature_group_name, version=version)
    df = fg.read()
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df
