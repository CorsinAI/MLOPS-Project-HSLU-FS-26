"""
Entry point for the feature pipeline.

Usage:
    python -m pipelines.feature.run
    python -m pipelines.feature.run --no-hopsworks
    python -m pipelines.feature.run --input data/structured_jobs_20.05_normalized_cleaned.jsonl --no-hopsworks
"""
import argparse
from pathlib import Path

from pipelines.feature.aggregate import load_postings, assign_windows, aggregate_counts
from pipelines.feature.features import compute_features

DEFAULT_INPUT = Path("data/structured_jobs_20.05_normalized_cleaned.jsonl")


def run(input_path: Path = DEFAULT_INPUT, push_to_hopsworks: bool = True) -> None:
    print(f"Loading postings from {input_path} ...")
    df = load_postings(input_path)
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--no-hopsworks", action="store_true")
    args = parser.parse_args()
    run(args.input, push_to_hopsworks=not args.no_hopsworks)
