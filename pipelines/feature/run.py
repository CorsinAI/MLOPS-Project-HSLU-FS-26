"""
Entry point for the feature pipeline.

Usage:
    python -m pipelines.feature.run
    python -m pipelines.feature.run --no-hopsworks
"""
import argparse

from dotenv import load_dotenv
load_dotenv()

from pipelines.feature.aggregate import load_postings, assign_windows, aggregate_counts
from pipelines.feature.features import compute_features


def run(push_to_hopsworks: bool = True) -> None:
    print("Loading postings from Azure Blob Storage ...")
    df = load_postings()
    print(f"  {len(df)} postings loaded")

    df = assign_windows(df)

    counts = aggregate_counts(df)
    print(f"  {len(counts)} (job_title, location, window) combinations")

    features = compute_features(counts)

    print(f"Feature table shape: {features.shape}")
    print(features.head())

    if push_to_hopsworks:
        from pipelines.feature.hopsworks_writer import write_feature_group
        write_feature_group(features)
    else:
        print("Skipping Hopsworks write (--no-hopsworks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hopsworks", action="store_true")
    args = parser.parse_args()
    run(push_to_hopsworks=not args.no_hopsworks)
