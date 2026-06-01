"""
Entry point for the training pipeline.

Usage:
    python -m pipelines.training.run
    python -m pipelines.training.run --no-hopsworks --input data/structured_jobs_20.05_normalized_cleaned.jsonl
"""
import argparse
from pathlib import Path

from pipelines.training.prepare import (
    HOLDOUT_WINDOWS,
    build_dataset,
    encode_categoricals,
    time_split,
    trim_for_training,
)

DEFAULT_INPUT = Path("data/structured_jobs_20.05_normalized_cleaned.jsonl")


def run(input_path: Path = DEFAULT_INPUT, from_hopsworks: bool = False) -> None:
    # --- Feature data ---
    if from_hopsworks:
        from pipelines.feature.hopsworks_reader import read_feature_group
        print("Reading features from Hopsworks ...")
        features = read_feature_group()
    else:
        from pipelines.feature.aggregate import load_postings, assign_windows, aggregate_counts
        from pipelines.feature.features import compute_features
        print(f"Building features from {input_path} ...")
        df = load_postings(input_path)
        df = assign_windows(df)
        counts = aggregate_counts(df)
        features = compute_features(counts)

    print(f"  {len(features)} feature rows across {features['window_start'].nunique()} windows")

    # --- Hold out the most recent complete windows ---
    features = trim_for_training(features)
    print(f"  After holdout: {features['window_start'].nunique()} windows "
          f"(latest {HOLDOUT_WINDOWS} complete windows reserved for inference)")

    # --- Prepare training data ---
    print("Preparing training dataset ...")
    dataset = build_dataset(features)
    dataset = encode_categoricals(dataset)

    train_df, val_df, test_df = time_split(dataset)

    print(f"  Train rows: {len(train_df)}  ({train_df['window_start'].min().date()} → {train_df['window_start'].max().date()})")
    print(f"  Val rows:   {len(val_df)}  ({val_df['window_start'].min().date()} → {val_df['window_start'].max().date()})")
    print(f"  Test rows:  {len(test_df)}  ({test_df['window_start'].min().date()} → {test_df['window_start'].max().date()})")

    # --- Train ---
    from pipelines.training.train import train_model
    print("Training LightGBM ...")
    booster, metrics = train_model(train_df, val_df, test_df)

    print(f"\nDone. MLflow run logged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hopsworks", action="store_true",
                        help="Read features from a local JSONL file instead of Hopsworks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to JSONL file (used when --no-hopsworks is set)")
    args = parser.parse_args()
    run(args.input, from_hopsworks=not args.no_hopsworks)
